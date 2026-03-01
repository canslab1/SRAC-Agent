"""
evolution.py -- Evolutionary Algorithms with Self-Reputation Awareness
======================================================================

Corresponds to Java source files:
  - Evolution/EvoThread_CA.java       (CA evolution: crossover + mutation + selection)
  - Evolution/EvoThread_CA_Mix.java   (CA with self-aware agent mixing -- KEY INNOVATION)
  - Evolution/EvoThread_SW.java       (SW evolution, identical logic to CA)
  - Evolution/EvoThread_SW_Mix.java   (SW with self-aware mixing)

This module implements the complete evolutionary cycle:
  1. Fitness computation (IPD game play)
  2. Reputation evaluation (z-score based)
  3. Self-awareness mechanism (SRAC -- the paper's core contribution)
  4. Selection and reproduction (crossover, mutation, replacement)

Conversion notes:
  - Java uses separate Thread subclasses (EvoThread_CA, EvoThread_CA_Mix, etc.)
    Python unifies everything into a single EvolutionEngine class with a topology
    parameter, because the only differences between CA and SW variants are:
    (a) the network topology (handled by network.py)
    (b) the selection threshold (3/4 for CA, 7/8 for SW)
    (c) the z-score thresholds for self-awareness
  - Java runs evolution in a Thread (extends Thread); Python uses a generator
    pattern (yield per generation) to allow the GUI to update between generations.
  - The reputation evaluation algorithm from Appendix S2 of the manuscript is
    implemented in giveNeiRep() -> compute_neighbor_reputation_scores().
  - The relative fitness and self-reputation level computation from Appendix S3
    is implemented in getLowOrHigh() -> classify_z_score() and SelfAware() ->
    selfaware_selection().
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
import networkx as nx

from .agent import Agent, copy_agent_grid
from .parameters import SimParameters
from .network import get_neighbors
from .ipd_game import compute_fitness


class EvolutionEngine:
    """
    Main evolutionary simulation engine that manages the IPD game, reputation
    evaluation, self-awareness mechanism, and evolutionary selection.

    Corresponds to Java: EvoThread_CA, EvoThread_CA_Mix, EvoThread_SW, EvoThread_SW_Mix

    The Java code uses inheritance:
      EvoThread -> EvoThread_CA -> EvoThread_CA_Mix
                                -> EvoThread_SW -> EvoThread_SW_Mix

    Python unifies all variants into one class, using parameters to control behavior.
    """

    def __init__(self, agent_grid: np.ndarray, network: nx.Graph,
                 params: SimParameters, callback=None):
        """
        Initialize the evolution engine.

        Corresponds to Java: EvoThread_CA constructor + EvoThread_CA_Mix constructor

        Args:
            agent_grid: 2D numpy object array of Agent instances
                        (Java: Agent[][] myAgent)
            network: networkx.Graph representing the social interaction network
                     (Java: CAN myCAN or SWN -- the board[][] LinkedList structure)
            params: Simulation parameters
                    (Java: MyParameter parameter)
            callback: Optional callback function(generation, agent_grid, stats)
                      called after each generation for GUI updates.
                      (Java: handled via frame.pw.paintImmediately())
        """
        self.agent_grid = agent_grid
        self.network = network
        self.params = params
        self.callback = callback

        board_size = params.board_size

        # Fitness array for current generation
        # (Java: int Fitness[][] -- 2D array in EvoThread_CA)
        self.fitness = np.zeros((board_size, board_size), dtype=int)

        # Cooperation times: coop_times[i][j][k] = number of times neighbor k
        # cooperated with agent (i,j) in this generation
        # (Java: int[][][] CoopTimes in EvoThread_CA_Mix)
        self.coop_times: Dict[Tuple[int, int], List[int]] = {}

        # Neighbor-to-me reputation scores
        # (Java: int[][][] NeiToMeRep in EvoThread_CA_Mix)
        self.nei_to_me_rep: Dict[Tuple[int, int], List[int]] = {}

        # Self-aware agent mask
        # (Java: boolean[][] SelfAwareAgent in EvoThread_CA_Mix)
        self.selfaware_mask = np.zeros((board_size, board_size), dtype=bool)

        # Initialize self-aware agents if enabled
        # (Java: CreateSelfAwareAgent(ratio))
        if params.selfaware and params.selfaware_ratio > 0:
            self._create_selfaware_agents(params.selfaware_ratio)

        # Pre-compute neighbor lists for all nodes (optimization)
        self._neighbor_cache: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for i in range(board_size):
            for j in range(board_size):
                self._neighbor_cache[(i, j)] = get_neighbors(network, i, j)

        # Pre-compute reverse neighbor indices (optimization):
        # _reverse_idx[(agent_node, neighbor_node)] -> position of agent_node
        # in neighbor_node's neighbor list.  This eliminates the O(k) linear
        # search (list.index()) inside _set_nei_rep, which is called
        # O(N * k) ≈ 20 000 times per generation.
        self._reverse_idx: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        for i in range(board_size):
            for j in range(board_size):
                node = (i, j)
                for nei_node in self._neighbor_cache[node]:
                    nei_nbs = self._neighbor_cache[nei_node]
                    for rev_idx, n in enumerate(nei_nbs):
                        if n == node:
                            self._reverse_idx[(node, nei_node)] = rev_idx
                            break

        # Initialize reputation and cooperation tracking structures
        # (Java: initRepAndCoopTimes())
        self._init_rep_and_coop_times()

        # Evolution history: list of agent grid snapshots
        # (Java: EvoData.agent[][][] -- [generation][row][col])
        self.history: List[np.ndarray] = []

        # Determine selection threshold based on topology.
        # Store as (numerator, denominator) to use pure integer arithmetic,
        # exactly matching Java's `Neighbor.size() * N / D` integer math.
        # CA: 3/4 of neighbors must beat you (Java: EvoThread_CA.FindSelectionByOneAgent)
        # SW: 7/8 of neighbors must beat you (Java: EvoThread_SW_Mix.FindSelectionByOneAgent)
        if params.topology_type == "Small-World":
            self._thresh_num = 7
            self._thresh_den = 8
        else:
            self._thresh_num = 3
            self._thresh_den = 4

    def _create_selfaware_agents(self, ratio: float):
        """
        Randomly designate a fraction of agents as self-aware.

        Corresponds to Java: EvoThread_CA_Mix.CreateSelfAwareAgent(float ratio)

        Java algorithm:
            SelfAwareAgent = new boolean[BoardSize][BoardSize];
            for(int i=0; i<BoardSize; i++)
              for(int j=0; j<BoardSize; j++)
                if(Probability(ratio))
                  SelfAwareAgent[i][j] = true;

        Args:
            ratio: Fraction of agents to make self-aware (0.0 to 1.0)
        """
        bs = self.params.board_size
        # Each agent independently becomes self-aware with probability = ratio
        # (Java: if(Probability(ratio)) SelfAwareAgent[i][j] = true;)
        self.selfaware_mask = np.random.random((bs, bs)) < ratio

    def _init_rep_and_coop_times(self):
        """
        Initialize reputation and cooperation tracking data structures.

        Corresponds to Java: EvoThread_CA_Mix.initRepAndCoopTimes()

        Java:
            NeiToMeRep = new int[BoardSize][BoardSize][];
            CoopTimes = new int[BoardSize][BoardSize][];
            for each (i,j):
                NeiToMeRep[i][j] = new int[board[i][j].size()];
                CoopTimes[i][j] = new int[board[i][j].size()];
        """
        bs = self.params.board_size
        for i in range(bs):
            for j in range(bs):
                n_neighbors = len(self._neighbor_cache[(i, j)])
                self.coop_times[(i, j)] = [0] * n_neighbors
                self.nei_to_me_rep[(i, j)] = [0] * n_neighbors

    def _clear_rep_and_coop_times(self):
        """
        Reset reputation and cooperation data for the next generation.

        Corresponds to Java: EvoThread_CA_Mix.clearRepAndCoopTimes()

        Java:
            for each (i,j):
                myAgent[i][j].setReputation(0);
                for each neighbor k:
                    CoopTimes[i][j][k] = 0;
        """
        bs = self.params.board_size
        for i in range(bs):
            for j in range(bs):
                self.agent_grid[i, j].reputation = 0
                neighbors = self._neighbor_cache[(i, j)]
                self.coop_times[(i, j)] = [0] * len(neighbors)

    def _clear_real_rep(self):
        """
        Reset real reputation for all agents.

        Corresponds to Java: EvoThread_CA_Mix.clearRealRep()

        Java:
            for each (i,j):
                myAgent[i][j].setRealRep(0);
        """
        bs = self.params.board_size
        for i in range(bs):
            for j in range(bs):
                self.agent_grid[i, j].real_rep = 0

    # =========================================================================
    # Reputation Evaluation (Appendix S2 of the manuscript)
    # =========================================================================

    @staticmethod
    def compute_neighbor_reputation_scores(coop_times: List[int]) -> List[int]:
        """
        Evaluate the relative reputation scores of an agent's opponents
        based on their cooperation times.

        Corresponds to Java: EvoThread_CA_Mix.giveNeiRep(int[] X)
        Also corresponds to Appendix S2 algorithm in the manuscript:

        Manuscript pseudo-code:
            1. Ci = (c0,i, c1,i, ..., cn-1,i) -- cooperation counts
            2. Calculate avg and std of Ci
            3. Compute z-score: ri,j = (cj,i - avg) / std
            4. Return reputation list

        Java implementation maps z-scores to discrete levels:
            z <= -2  -> -3
            z <= -1  -> -2
            z <=  0  -> -1
            z <=  1  ->  1
            z <=  2  ->  2
            z >   2  ->  3

        Args:
            coop_times: List of cooperation counts from each neighbor
                        (Java: int[] X = NeiForMe_CoopTimes)

        Returns:
            List of integer reputation levels for each neighbor

        Performance note:
            Uses pure Python math instead of numpy for the typical case of
            ~8 neighbors. Creating numpy arrays and calling np.mean/np.std
            for 8-element arrays adds ~3-5μs overhead per call, which
            accumulates significantly over 2500 agents per generation.
        """
        n = len(coop_times)
        if n == 0:
            return []

        # Compute mean using pure Python (faster than numpy for small n)
        # (Java: for(int i=0;i<n;i++) Mean += X[i]/(double)n;)
        total = 0
        for x in coop_times:
            total += x
        mean = total / n

        # Compute population standard deviation using pure Python
        # (Java: for(int i=0;i<n;i++) Var += Math.pow(Mean-X[i],2)/(double)n;
        #        StdDiff = Math.sqrt(Var);)
        var_sum = 0.0
        for x in coop_times:
            diff = mean - x
            var_sum += diff * diff
        std = (var_sum / n) ** 0.5

        nei_levels = []
        if std == 0:
            # Java behavior: z = (X[i]-Mean)/0 = NaN.
            # In Java, NaN comparisons (<=, <, >, >=) all return false,
            # so the if/else-if chain falls through to the final else,
            # assigning NeiLevel[i] = 3 (highest reputation).
            # We replicate this exact Java behavior for consistency.
            for _ in range(n):
                nei_levels.append(3)
            return nei_levels

        for i in range(n):
            # Compute z-score
            # (Java: double z = (X[i] - Mean) / StdDiff;)
            z = (coop_times[i] - mean) / std

            # Map z-score to discrete reputation level
            # (Java: if(z<=-2) NeiLevel[i]=-3; else if(z<=-1) -2; etc.)
            if z <= -2:
                nei_levels.append(-3)
            elif z <= -1:
                nei_levels.append(-2)
            elif z <= 0:
                nei_levels.append(-1)
            elif z <= 1:
                nei_levels.append(1)
            elif z <= 2:
                nei_levels.append(2)
            else:
                nei_levels.append(3)

        return nei_levels

    def _compute_reputation(self):
        """
        Compute reputation scores for all agents based on their neighbors'
        cooperation evaluations.

        Corresponds to Java: EvoThread_CA_Mix.ComputeRep()

        Java algorithm:
            for each agent (i,j):
                1. Get cooperation times from neighbors -> NeiForMe_CoopTimes
                2. Compute relative reputation scores -> MeForNei_Rep = giveNeiRep(...)
                3. Set each neighbor's reputation based on this agent's evaluation

        This implements the first part of the reputation evaluation system
        described in the manuscript: each agent evaluates its opponents'
        relative reputations based on their cooperation frequency.
        """
        bs = self.params.board_size
        for i in range(bs):
            for j in range(bs):
                # 1. Get cooperation times from this agent's perspective
                # (Java: int[] NeiForMe_CoopTimes = CoopTimes[i][j];)
                nei_coop = self.coop_times[(i, j)]

                # 2. Compute relative reputation scores for neighbors
                # (Java: int[] MeForNei_Rep = giveNeiRep(NeiForMe_CoopTimes);)
                me_for_nei_rep = self.compute_neighbor_reputation_scores(nei_coop)

                # 3. Set reputation on each neighbor
                # (Java: setNeiRep(myAgent[i][j], MeForNei_Rep);)
                self._set_nei_rep(i, j, me_for_nei_rep)

    def _set_nei_rep(self, agent_row: int, agent_col: int, rep_scores: List[int]):
        """
        Distribute reputation scores from one agent to its neighbors.

        Corresponds to Java: EvoThread_CA_Mix.setNeiRep(Agent agent, int[] MeToNei_Rep)

        Java:
            for each neighbor i:
                get neighbor position from encoded index
                myAgent[row][col].setReputation(myAgent[row][col].getReputation() + MeToNei_Rep[i]);
                find this agent's index in neighbor's neighbor list
                NeiToMeRep[row][col][index] = MeToNei_Rep[i];
        """
        neighbors = self._neighbor_cache[(agent_row, agent_col)]
        agent_node = (agent_row, agent_col)

        for idx, (nei_row, nei_col) in enumerate(neighbors):
            # Accumulate reputation on neighbor
            # (Java: myAgent[row][col].setReputation(myAgent[row][col].getReputation() + MeToNei_Rep[i]);)
            self.agent_grid[nei_row, nei_col].reputation += rep_scores[idx]

            # Look up this agent's index in the neighbor's neighbor list
            # using precomputed reverse index (O(1) instead of O(k) list.index())
            # (Java: int IndexOfNeiSeeMe = myCAN.board[row][col].indexOf(...))
            nei_node = (nei_row, nei_col)
            reverse_idx = self._reverse_idx.get((agent_node, nei_node))
            if reverse_idx is not None:
                # Record the reputation score
                # (Java: NeiToMeRep[row][col][IndexOfNeiSeeMe] = MeToNei_Rep[i];)
                self.nei_to_me_rep[nei_node][reverse_idx] = rep_scores[idx]

    def _compute_real_rep(self):
        """
        Compute the "real" reputation for each agent.

        Corresponds to Java: EvoThread_CA_Mix.ComputeRealRep()

        Java algorithm:
            for each agent (i,j):
                RealRep = sum over neighbors k of:
                    NeiToMeRep[i][j][k] * CoopTimes[i][j][k] / TotalTimes
                myAgent[i][j].setRealRep((int)RealRep);

        Real reputation combines the neighbor's evaluation (NeiToMeRep)
        with the cooperation frequency (CoopTimes), weighted by total rounds.
        This gives a more nuanced reputation measure than raw reputation alone.
        """
        bs = self.params.board_size
        total_times = float(self.params.ipd_rounds)

        for i in range(bs):
            for j in range(bs):
                neighbors = self._neighbor_cache[(i, j)]
                real_rep = 0.0
                for k in range(len(neighbors)):
                    # (Java: RealRep += NeiToMeRep[i][j][k] * CoopTimes[i][j][k] / TotalTimes;)
                    real_rep += (self.nei_to_me_rep[(i, j)][k] *
                                 self.coop_times[(i, j)][k] / total_times)
                # (Java: myAgent[i][j].setRealRep((int)RealRep);)
                self.agent_grid[i, j].real_rep = int(real_rep)

    # =========================================================================
    # Self-Awareness Mechanism (Section 3, Step 4c of the manuscript)
    # =========================================================================

    @staticmethod
    def classify_z_score(values, self_value: int,
                         low_ratio: float, high_ratio: float) -> int:
        """
        Classify a value as LOW (-1), MIDDLE (0), or HIGH (1) relative to
        a group of values using z-score thresholds.

        Corresponds to Java: EvoThread_CA_Mix.getLowOrHigh(int[] X, int self,
                             double LowRatio, double HighRatio)
        Also corresponds to Appendix S3, Algorithm 1 and Algorithm 2 of the manuscript.

        Manuscript pseudo-code (Algorithm 1 -- Fitness):
            1. Collect fitness values of all opponents
            2. Calculate avg and std
            3. Fitnessi <- MIDDLE
            4. if afi < (avg - std) then LOW
               else if afi > (avg + std) then HIGH

        Java implementation uses continuous thresholds instead of +/- 1 std:
            z = (self - Mean) / StdDiff;
            if(z < LowRatio) return -1;   // LOW
            if(z > HighRatio) return 1;    // HIGH
            return 0;                       // MIDDLE

        Args:
            values: List or array of neighbor values (fitness or reputation)
                    (Java: int[] X -- NeiFitness or NeiRep)
            self_value: This agent's value (Java: int self -- MyFitness or MyRep)
            low_ratio: Z-score threshold for LOW classification
                       (Java: double LowRatio)
            high_ratio: Z-score threshold for HIGH classification
                        (Java: double HighRatio)

        Returns:
            -1 (LOW), 0 (MIDDLE), or 1 (HIGH)

        Performance note:
            Uses pure Python math instead of numpy for the typical case of
            ~8 neighbors. This avoids numpy dispatch overhead (~3-5μs per call)
            which accumulates significantly across all agents.
        """
        n = len(values)
        if n == 0:
            return 0

        # Compute mean and population standard deviation using pure Python
        # (Java: Mean, Var, StdDiff computation)
        total = 0
        for v in values:
            total += v
        mean = total / n

        var_sum = 0.0
        for v in values:
            diff = v - mean
            var_sum += diff * diff
        std = (var_sum / n) ** 0.5

        if std == 0:
            # When all neighbor values are identical, std=0.
            # Java: z = (self - Mean) / 0 produces:
            #   NaN   if self == Mean  → all comparisons false → returns 0 (MIDDLE)
            #   +Inf  if self > Mean   → +Inf > HighRatio true → returns 1 (HIGH)
            #   -Inf  if self < Mean   → -Inf < LowRatio true → returns -1 (LOW)
            if self_value > mean:
                return 1   # HIGH
            elif self_value < mean:
                return -1  # LOW
            else:
                return 0   # MIDDLE

        # Compute z-score
        # (Java: double z = (self - Mean) / StdDiff;)
        z = (self_value - mean) / std

        # Classify based on thresholds
        # (Java: if(z < LowRatio) return -1; if(z > HighRatio) return 1; return 0;)
        if z < low_ratio:
            return -1  # LOW
        elif z > high_ratio:
            return 1   # HIGH
        else:
            return 0   # MIDDLE

    def _selfaware_selection(self, agent: Agent):
        """
        Apply the self-awareness mechanism to adjust an agent's strategy.

        Corresponds to Java: EvoThread_CA_Mix.SelfAware(Agent agent)
        This is the CORE INNOVATION of the paper (Section 3, Step 4c).

        Algorithm (from manuscript and Java source):
            1. Get agent's fitness and reputation
            2. Get all neighbors' fitness and reputation values
            3. Classify agent's fitness as LOW/MIDDLE/HIGH using z-scores
            4. Classify agent's reputation as LOW/MIDDLE/HIGH using z-scores
            5. If fitness is LOW or reputation is LOW:
               a. Find "socially good" neighbors (both fitness HIGH and reputation HIGH)
               b. If socially good neighbors exist:
                  - Copy one's strategy randomly (evolutionary learning)
               c. If no socially good neighbors exist:
                  - Use standard evolutionary selection (AgentOfNextG)

        The self-awareness mechanism implements a feedback loop where agents
        can detect that their strategies are not meeting social expectations
        and actively seek to learn from successful, reputable neighbors.

        Args:
            agent: The self-aware agent to potentially adjust
        """
        row, col = agent.row, agent.col
        neighbors = self._neighbor_cache[(row, col)]

        # Get fitness and reputation thresholds for current topology
        f_low, f_high, r_low, r_high = self.params.get_active_thresholds()

        # Collect neighbor fitness and reputation values as Python lists
        # (avoiding numpy array creation overhead for ~8 elements)
        # (Java: int[] NeiFitness = new int[Neighbor.size()];
        #        int[] NeiRep = new int[Neighbor.size()];)
        nei_fitness = [self.agent_grid[nr, nc].fitness for nr, nc in neighbors]
        nei_rep = [self.agent_grid[nr, nc].reputation for nr, nc in neighbors]

        # Classify this agent's fitness and reputation relative to neighbors
        # (Java: int MyFitnessLH = getLowOrHigh(NeiFitness, MyFitness, F_LowRatio, F_HighRatio);
        #        int MyRepLH = getLowOrHigh(NeiRep, MyRep, R_LowRatio, R_HighRatio);)
        my_fitness_lh = self.classify_z_score(nei_fitness, agent.fitness, f_low, f_high)
        my_rep_lh = self.classify_z_score(nei_rep, agent.reputation, r_low, r_high)

        # If fitness is LOW or reputation is LOW, need to adjust strategy
        # (Java: if((MyFitnessLH == -1) || (MyRepLH == -1)))
        if my_fitness_lh == -1 or my_rep_lh == -1:
            # Classify each neighbor's fitness and reputation
            # (Java: int[] NeiFitnessLH, NeiRepLH for each neighbor)
            num_good = 0
            good_indices = []

            for idx, (nr, nc) in enumerate(neighbors):
                nei_f_lh = self.classify_z_score(nei_fitness, nei_fitness[idx], f_low, f_high)
                nei_r_lh = self.classify_z_score(nei_rep, nei_rep[idx], r_low, r_high)

                # A "socially good" neighbor has BOTH high fitness AND high reputation
                # (Java: if((NeiFitnessLH[i]==1) && (NeiRepLH[i]==1)) NumOfGood++;)
                if nei_f_lh == 1 and nei_r_lh == 1:
                    num_good += 1
                    good_indices.append(idx)

            if num_good == 0:
                # No socially good neighbors: fall back to evolutionary selection
                # (Java: agent.setChromosome(AgentOfNextG(agent).getChromosome());)
                new_agent = self._agent_of_next_g(agent)
                agent.chromosome = new_agent.chromosome.copy()
            else:
                # Copy a randomly selected socially good neighbor's strategy
                # (Java: int RandomOne = (int)(Math.random()*NumOfGood);
                #        agent.setChromosome(myAgent[Nei_Row][Nei_Col].getChromosome());)
                random_idx = good_indices[np.random.randint(num_good)]
                nr, nc = neighbors[random_idx]
                agent.chromosome = self.agent_grid[nr, nc].chromosome.copy()

    # =========================================================================
    # Evolutionary Operators (Java: EvoThread_CA)
    # =========================================================================

    def _mutation(self, agent: Agent) -> Agent:
        """
        Apply mutation to an agent's chromosome.

        Corresponds to Java: EvoThread_CA.Mutation(Agent agent)

        Java algorithm:
            Agent tempAgent = agent.copy();
            for(int j=0; j<StrategyLength; j++)
                if(Probability(Mutation_Rate))
                    tempChromosome[j] = (byte)(tempChromosome[j] ^ 1);  // flip bit
            return tempAgent;

        Each bit in the chromosome is independently flipped with probability = mutation_rate.

        Args:
            agent: Agent to mutate

        Returns:
            New Agent with potentially mutated chromosome
        """
        new_agent = agent.copy()
        # For each bit, flip with probability mutation_rate
        # (Java: if(Probability(Mutation_Rate)) tempChromosome[j]=(byte)(tempChromosome[j]^1);)
        mutation_mask = np.random.random(new_agent.strategy_length) < self.params.mutation_rate
        new_agent.chromosome[mutation_mask] ^= 1  # XOR with 1 to flip
        return new_agent

    def _crossover(self, agent1: Agent, agent2: Agent) -> Tuple[Agent, Agent]:
        """
        Apply single-point crossover between two agents.

        Corresponds to Java: EvoThread_CA.Crossover(Agent[] agent)

        Standard single-point crossover (evolutionary computation):
            1. Test crossover probability ONCE (not per-bit).
            2. If crossover occurs:
               a. Pick a random cut point in [1, StrategyLength].
               b. Child1 = parent1[0..cut-1] + parent2[cut..end]
               c. Child2 = parent2[0..cut-1] + parent1[cut..end]
            3. If crossover does not occur:
               Children are copies of their parents (no swap).

        Bug fix note:
            The original Java EvoThread_CA.Crossover() contained two issues:
            (a) Crossover probability was tested per-bit (inside the loop)
                instead of once per crossover operation.
            (b) In the 'else' branch (probability test failed), the code
                performed reference aliasing:
                    tempChromosome_1 = tempChromosome1;
                    tempChromosome_2 = tempChromosome2;
                This caused the output array to alias the parent's array,
                corrupting subsequent swap operations -- both children ended
                up with parent2's bit values at positions >= cut_point.
            This Python version fixes both issues to implement correct
            single-point crossover as defined in evolutionary computation.

        Args:
            agent1, agent2: Parent agents

        Returns:
            Tuple of two offspring agents
        """
        child1 = agent1.copy()
        child2 = agent2.copy()

        str_len = agent1.strategy_length

        if np.random.random() < self.params.crossover_rate:
            # Crossover occurs: apply single-point crossover
            # Java: int cut_C=(byte)(Math.random()*agent[0].StrategyLength+1);
            # For StrategyLength=4: values in {1, 2, 3, 4}
            cut_point = np.random.randint(1, str_len + 1)

            p1 = agent1.chromosome.copy()
            p2 = agent2.chromosome.copy()

            # Child1 = parent1[0..cut-1] + parent2[cut..end]
            # Child2 = parent2[0..cut-1] + parent1[cut..end]
            c1 = np.concatenate([p1[:cut_point], p2[cut_point:]])
            c2 = np.concatenate([p2[:cut_point], p1[cut_point:]])

            child1.chromosome = c1
            child2.chromosome = c2

        # else: no crossover -- children remain copies of parents

        return child1, child2

    def _find_best_neighbor(self, agent: Agent) -> Agent:
        """
        Find the neighbor with the highest fitness (including self).

        Corresponds to Java: EvoThread_CA.FindBestStrategyOfNeighbors(Agent agent)
        + the decode step in AgentOfNextG.

        Bug fix: The original Java AgentOfNextG decoded the best neighbor
        index with row/col swapped:
            int row = bestIndex % parameter.BoardSize;  // WRONG: gives col
            int col = bestIndex / parameter.BoardSize;  // WRONG: gives row
        This caused it to look up myAgent[col_of_best][row_of_best] instead
        of myAgent[row_of_best][col_of_best], returning a potentially wrong
        agent. This Python version uses correct decoding:
            row = bestIndex // BoardSize
            col = bestIndex % BoardSize

        Args:
            agent: The focal agent

        Returns:
            The neighbor (or self) with the highest fitness
        """
        row, col = agent.row, agent.col
        neighbors = self._neighbor_cache[(row, col)]
        bs = self.params.board_size

        # Start with self as best (Java: BestStrategyIndex = row*BoardSize + col)
        best_index = row * bs + col
        best_fitness = self.fitness[row, col]

        for nr, nc in neighbors:
            nf = self.fitness[nr, nc]
            if nf > best_fitness:
                best_index = nr * bs + nc
                best_fitness = nf

        # Decode linear index back to (row, col).
        # Encoding: index = row * BoardSize + col
        # Correct decoding: row = index // BoardSize, col = index % BoardSize
        #
        # Bug fix: The original Java AgentOfNextG had the decoding swapped:
        #   int row = bestIndex % parameter.BoardSize;  // WRONG: gives col
        #   int col = bestIndex / parameter.BoardSize;  // WRONG: gives row
        # This Python version uses the correct decoding.
        decoded_row = best_index // bs
        decoded_col = best_index % bs
        return self.agent_grid[decoded_row, decoded_col]

    def _agent_of_next_g(self, agent: Agent) -> Agent:
        """
        Generate a candidate agent for the next generation using evolutionary operators.

        Corresponds to Java: EvoThread_CA.AgentOfNextG(Agent agent)

        Java algorithm:
            1. Find best neighbor -> candidate[0]
            2. Pick two random neighbors, crossover -> candidates[1], [2]
            3. Mutate all three -> candidates[3], [4], [5]
            4. Randomly pick one of the 6 candidates

        This creates a pool of 6 candidate strategies:
          - 1 best neighbor (exploitation)
          - 2 crossover offspring (recombination)
          - 3 mutations of the above (exploration)
        Then randomly selects one, balancing exploration and exploitation.

        Args:
            agent: The agent being replaced

        Returns:
            A candidate Agent for the next generation
        """
        neighbors = self._neighbor_cache[(agent.row, agent.col)]
        if not neighbors:
            return self._mutation(agent)

        candidates = []

        # 1. Best neighbor's copy
        # (Java: CandidateAgent[0] = myAgent[row][col].copy();)
        best = self._find_best_neighbor(agent)
        candidates.append(best.copy())

        # 2. Crossover of two random neighbors
        # (Java: int[] C = new int[2]; random selection; Crossover(tempCrossover))
        if len(neighbors) >= 2:
            idx1, idx2 = np.random.choice(len(neighbors), 2, replace=False)
            n1r, n1c = neighbors[idx1]
            n2r, n2c = neighbors[idx2]
            child1, child2 = self._crossover(
                self.agent_grid[n1r, n1c],
                self.agent_grid[n2r, n2c]
            )
            candidates.append(child1)
            candidates.append(child2)
        else:
            # Only one neighbor: duplicate for crossover
            nr, nc = neighbors[0]
            candidates.append(self.agent_grid[nr, nc].copy())
            candidates.append(self.agent_grid[nr, nc].copy())

        # 3. Mutate all three candidates
        # (Java: for(int i=0;i<3;i++) CandidateAgent[3+i]=Mutation(CandidateAgent[i]).copy();)
        for i in range(3):
            candidates.append(self._mutation(candidates[i]))

        # 4. Randomly pick one of the 6 candidates
        # (Java: int random=(int)(Math.random()*CandidataLength); return CandidateAgent[random];)
        chosen = candidates[np.random.randint(len(candidates))]
        return chosen

    def _find_selection_needed(self, agent: Agent) -> bool:
        """
        Determine if an agent should be replaced (is in the bottom fraction).

        Corresponds to Java: EvoThread_CA.FindSelectionByOneAgent(Agent agent)

        Java algorithm:
            threshold = Neighbor.size() * 3/4;  // CA: 3/4, SW: 7/8
            counter = 0;
            for each neighbor:
                if neighbor.fitness > agent.fitness: counter++;
                if counter > threshold: return true;
            return false;

        An agent is selected for replacement if more than threshold_ratio
        of its neighbors have higher fitness, meaning it's performing poorly
        relative to its local neighborhood.

        Args:
            agent: The agent to evaluate

        Returns:
            True if the agent should be replaced
        """
        row, col = agent.row, agent.col
        neighbors = self._neighbor_cache[(row, col)]
        # Pure integer arithmetic to exactly match Java's
        # `int threshold = Neighbor.size() * N / D;`
        threshold = len(neighbors) * self._thresh_num // self._thresh_den
        counter = 0

        for nr, nc in neighbors:
            if self.fitness[nr, nc] > agent.fitness:
                counter += 1
            if counter > threshold:
                return True

        return False

    def _selection_by_sn(self):
        """
        Perform selection for all agents based on their social network.

        Corresponds to Java: EvoThread_CA_Mix.SelectionBySN() (overrides EvoThread_CA version)

        Java algorithm:
            for each agent (i,j):
                if SelfAwareAgent[i][j]:
                    SelfAware(myAgent[i][j])       // Self-awareness mechanism
                else:
                    if FindSelectionByOneAgent(myAgent[i][j]):
                        replace with AgentOfNextG   // Poor performer -> replace
                    else:
                        10% chance of mutation       // Random exploration

        This is the key selection step where self-aware and non-self-aware agents
        are treated differently, implementing the mixed-population model.
        """
        bs = self.params.board_size
        for i in range(bs):
            for j in range(bs):
                agent = self.agent_grid[i, j]

                if self.selfaware_mask[i, j]:
                    # Self-aware agent: use self-awareness mechanism
                    # (Java: SelfAware(myAgent[i][j]);)
                    self._selfaware_selection(agent)
                else:
                    # Non-self-aware agent: standard evolutionary selection
                    # (Java: if(FindSelectionByOneAgent(myAgent[i][j]))
                    #            myAgent[i][j].setChromosome(AgentOfNextG(...).getChromosome());
                    #        else if(Probability(0.1))
                    #            myAgent[i][j].setChromosome(Mutation(...).getChromosome());)
                    if self._find_selection_needed(agent):
                        new_agent = self._agent_of_next_g(agent)
                        agent.chromosome = new_agent.chromosome.copy()
                    elif np.random.random() < 0.1:
                        mutated = self._mutation(agent)
                        agent.chromosome = mutated.chromosome.copy()

    # =========================================================================
    # Main Evolution Loop
    # =========================================================================

    def run(self) -> Generator[dict, None, None]:
        """
        Run the evolutionary simulation, yielding status after each generation.

        Corresponds to Java: EvoThread_CA_Mix.run() (the Thread.run() method)

        Java algorithm:
            for(int t=0; t<Generations; t++) {
                1. Compute fitness for all agents (Fitness_Calculas)
                2. Compute reputation (ComputeRep)
                3. Compute real reputation (ComputeRealRep)
                4. Save generation data (evodata.setEvoData)
                5. Selection (SelectionBySN) -- if not last generation
                6. Repaint GUI
                7. Clear reputation and cooperation data
                8. Increment generation
            }

        Yields:
            Dict with 'generation', 'avg_fitness', 'agent_grid' per generation.
            Using a generator allows the GUI to process events between generations.
            This replaces Java's Thread + frame.pw.paintImmediately() pattern.
        """
        bs = self.params.board_size
        has_selfaware = self.params.selfaware and self.params.selfaware_ratio > 0

        for gen in range(self.params.generations):
            # Step 1: Compute fitness for all agents
            # (Java: for each (i,j): Fitness[i][j] = Fitness_Calculas(myAgent[i][j], ...))
            for i in range(bs):
                for j in range(bs):
                    agent = self.agent_grid[i, j]
                    neighbors = self._neighbor_cache[(i, j)]
                    is_sa = bool(self.selfaware_mask[i, j]) if has_selfaware else False

                    fitness_val, coop_list = compute_fitness(
                        agent, neighbors, self.agent_grid,
                        self.params.memory_length, self.params.ipd_rounds,
                        is_selfaware=is_sa
                    )

                    # Store fitness
                    # (Java: Fitness[i][j] = ...; myAgent[i][j].setFitness(Fitness[i][j]);)
                    self.fitness[i, j] = fitness_val
                    agent.fitness = fitness_val

                    # Store cooperation times
                    # (Java: CoopTimes[i][j][k]++ done inside Fitness_Calculas)
                    self.coop_times[(i, j)] = coop_list

            # Step 2: Compute reputation (if self-aware agents exist)
            # (Java: ComputeRep();)
            if has_selfaware:
                self._compute_reputation()
                # Step 3: Compute real reputation
                # (Java: ComputeRealRep();)
                self._compute_real_rep()

            # Step 4: Save generation data
            # (Java: evodata.setEvoData(CurGeneration, myAgent);)
            self.history.append(copy_agent_grid(self.agent_grid))

            # Step 5: Selection (skip on last generation)
            # (Java: if(this.CurGeneration < parameter.Generations-1) this.SelectionBySN();)
            if gen < self.params.generations - 1:
                self._selection_by_sn()

            # Compute generation statistics for callback/yield
            avg_fitness = np.mean(self.fitness)
            total_fitness = np.sum(self.fitness)

            stats = {
                'generation': gen,
                'avg_fitness': float(avg_fitness),
                'total_fitness': int(total_fitness),
            }

            # Step 6: Notify callback (GUI repaint equivalent)
            if self.callback:
                self.callback(gen, self.agent_grid, stats)

            # Step 7: Clear reputation and cooperation data for next generation
            # (Java: clearRepAndCoopTimes(); clearRealRep();)
            if has_selfaware:
                self._clear_rep_and_coop_times()
                self._clear_real_rep()

            yield stats

    def run_complete(self) -> List[dict]:
        """
        Run the complete evolution and return all generation statistics.

        Convenience method that consumes the generator from run().
        Useful for batch experiments where no GUI update is needed.

        Returns:
            List of statistics dictionaries, one per generation
        """
        return list(self.run())
