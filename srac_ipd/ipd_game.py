"""
ipd_game.py -- Iterated Prisoner's Dilemma Game Engine (Optimized)
==================================================================

Corresponds to Java source files:
  - Evolution/EvoThread_CA.java  (method: Fitness_Calculas)
  - Evolution/EvoThread_CA_Mix.java  (method: Fitness_Calculas with selfaware flag)

This module implements the core IPD game mechanics:
  1. Playing q rounds of the Prisoner's Dilemma between two agents
  2. Computing fitness (total payoff) for each agent across all neighbors
  3. Tracking cooperation times for reputation calculation

Conversion notes:
  - Java's Fitness_Calculas uses manual array indexing and bit manipulation.
    Python uses numpy arrays for clarity and potential vectorization.
  - The Java code has two versions of Fitness_Calculas:
    (a) EvoThread_CA.Fitness_Calculas(Agent, LinkedList) -- non-self-aware
    (b) EvoThread_CA_Mix.Fitness_Calculas(Agent, LinkedList, boolean) -- with selfaware flag
    Python unifies these into a single function with an is_selfaware parameter.
  - Key behavioral difference: Self-aware agents initialize memory bits to 0 (both cooperate),
    while non-self-aware agents initialize memory bits randomly. This means self-aware
    agents start each interaction with a cooperative posture.
  - The payoff matrix is defined in parameters.py as a numpy array for efficient lookup.

Performance optimization notes (Python-specific):
  - For the default memory_length=1 case (4-bit chromosome, 16 possible states),
    the original numpy-based inner loop created numpy arrays and called np.sum/np.arange
    on 2-element arrays inside a 100-round loop, which added ~1-5μs of dispatch overhead
    PER ROUND. With 2500 agents × 8 neighbors × 100 rounds = 2M iterations/generation,
    this caused each generation to take several minutes.
  - The optimized version uses:
    (a) Precomputed state-transition tables (16 entries, built once per game)
    (b) Pure Python integers for the inner loop (no numpy dispatch overhead)
    (c) Cycle detection: for deterministic strategies, the game enters a repeating
        cycle within at most 16 rounds. The remaining rounds are computed analytically.
    This reduces each generation from minutes to ~1-2 seconds.
"""

import random as _random
import numpy as np
from typing import List, Tuple, Optional

from .agent import Agent
from .parameters import PAYOFF_MATRIX

# ---------------------------------------------------------------------------
# Precomputed payoff lookup: _PAY[my_action][opponent_action]
# Cooperate = 0, Defect = 1.
# R = 3 (CC), S = 0 (CD), T = 5 (DC), P = 1 (DD)
# Using tuple-of-tuples for fastest Python access (faster than list or numpy
# for scalar lookups in tight loops).
# (Java: hardcoded if/else in Fitness_Calculas)
# ---------------------------------------------------------------------------
_PAY = ((3, 0), (5, 1))


def play_ipd_rounds(agent_chromosome: np.ndarray,
                    opponent_chromosome: np.ndarray,
                    memory_length: int,
                    num_rounds: int,
                    is_selfaware: bool = False) -> Tuple[int, int]:
    """
    Play multiple rounds of the Iterated Prisoner's Dilemma between two agents.

    Corresponds to Java: EvoThread_CA.Fitness_Calculas (the inner while loop)
    and EvoThread_CA_Mix.Fitness_Calculas (with selfaware parameter)

    Dispatches to an optimized fast path for memory_length=1 (the default
    and most common case), or a general path for other memory lengths.

    Args:
        agent_chromosome: Agent's strategy chromosome (Java: agent.Chromosome)
        opponent_chromosome: Opponent's strategy chromosome
        memory_length: Agent memory capacity (Java: parameter.MemoryLength)
        num_rounds: Number of IPD rounds to play (Java: parameter.Times)
        is_selfaware: Whether the agent has self-awareness
                      (Java: boolean selfaware in EvoThread_CA_Mix.Fitness_Calculas)

    Returns:
        Tuple of (agent_score, opponent_cooperation_count):
          - agent_score: Total payoff accumulated by the agent
          - opponent_cooperation_count: Number of times opponent cooperated
    """
    if memory_length == 1:
        return _play_ipd_m1(agent_chromosome, opponent_chromosome,
                            num_rounds, is_selfaware)
    else:
        return _play_ipd_general(agent_chromosome, opponent_chromosome,
                                 memory_length, num_rounds, is_selfaware)


def _play_ipd_m1(agent_chrom: np.ndarray, oppo_chrom: np.ndarray,
                  num_rounds: int, is_selfaware: bool) -> Tuple[int, int]:
    """
    Ultra-fast IPD for memory_length=1 with transition-table and cycle detection.

    Corresponds to Java: inner loop of Fitness_Calculas for MemoryLength=1

    For memory_length=1:
      - Memory bits = 2 (my_last_action, opponent_last_action)
      - Strategy length = 4 (chromosome encodes: Scc, Scd, Sdc, Sdd)
      - State = (my_memory_position, opponent_memory_position)
        where each position ∈ {0, 1, 2, 3}
      - Total possible states = 4 × 4 = 16

    Optimization strategy:
      1. Build a 16-entry transition table mapping each state to
         (score, opp_cooperated, next_state) -- O(16) setup
      2. Detect cycle: since there are only 16 states and transitions are
         deterministic, the game MUST enter a cycle within 16 rounds.
      3. Once cycle is detected, compute remaining rounds analytically:
         score += full_cycles * cycle_score + leftover_score
      This reduces 100 rounds to typically 3-10 rounds of actual simulation.

    Memory encoding (matching Java exactly):
      self_arr  = [mem_bit_0, mem_bit_1, chrom_0, chrom_1, chrom_2, chrom_3]
      position  = mem_bit_0 * 2 + mem_bit_1
      action    = chrom[position]
      After each round:
        self_arr[0] = my_action     (Java: self[Memory_bits-2] = Strategy1)
        self_arr[1] = opp_action    (Java: self[Memory_bits-1] = Strategy2)
        oppo_arr[0] = opp_action    (Java: oppo[Memory_bits-2] = Strategy2)
        oppo_arr[1] = my_action     (Java: oppo[Memory_bits-1] = Strategy1)

    Args:
        agent_chrom: numpy int8 array of length 4
        oppo_chrom: numpy int8 array of length 4
        num_rounds: number of IPD rounds (Java: parameter.Times)
        is_selfaware: whether agent starts with cooperative memory

    Returns:
        (total_score, opponent_cooperation_count)
    """
    # Convert numpy chromosomes to Python list for fast scalar indexing.
    # tolist() is faster than iterating with int() for small arrays.
    # (Java: agent.Chromosome[k] accessed directly)
    a = agent_chrom.tolist()  # [Scc, Scd, Sdc, Sdd] as Python ints
    o = oppo_chrom.tolist()

    # Build transition table: trans[state] = (score, coop_bit, next_state)
    # state = agent_pos * 4 + opponent_pos  (compact encoding, 0..15)
    # (Java: the inner while loop logic, precomputed for all 16 states)
    trans = [None] * 16
    for pa in range(4):
        for po in range(4):
            act1 = a[pa]       # Agent's action from chromosome lookup
            act2 = o[po]       # Opponent's action from chromosome lookup
            sc = _PAY[act1][act2]         # Payoff for agent
            cc = 1 if act2 == 0 else 0    # Did opponent cooperate?
            # Next state: agent remembers (act1, act2), opponent remembers (act2, act1)
            # (Java: self[0]=Strategy1, self[1]=Strategy2, oppo[0]=Strategy2, oppo[1]=Strategy1)
            next_pa = act1 * 2 + act2     # Agent's next memory position
            next_po = act2 * 2 + act1     # Opponent's next memory position
            trans[pa * 4 + po] = (sc, cc, next_pa * 4 + next_po)

    # Initialize memory state
    # (Java: if(selfaware) self[k]=0,oppo[k]=0; else self[k]=(int)(Math.random()*2))
    if is_selfaware:
        # Self-aware agents: both start with CC memory -> position 0
        state = 0  # pa=0, po=0
    else:
        # Non-self-aware: random independent memory bits for self and opponent
        # self_mem = (random_bit, random_bit) -> pa = bit0*2 + bit1 ∈ {0..3}
        # oppo_mem = (random_bit, random_bit) -> po = bit0*2 + bit1 ∈ {0..3}
        # Combined state = pa*4 + po ∈ {0..15}, uniform distribution
        state = _random.randint(0, 15)

    score = 0
    opp_coop = 0

    # --- Simulate with cycle detection ---
    # Since transitions are deterministic and there are only 16 states,
    # the game MUST revisit a state within at most 16 rounds.
    # Once a cycle is found, compute remaining rounds analytically.
    seen = {}  # state -> round_number when first seen
    rnd = 0

    while rnd < num_rounds:
        if state in seen:
            # ---- Cycle detected! ----
            # The game is now in a repeating loop.
            cyc_start = seen[state]
            cyc_len = rnd - cyc_start
            remaining = num_rounds - rnd
            full_cycles = remaining // cyc_len
            leftover = remaining % cyc_len

            # Sum one full cycle (starting from current state)
            cyc_score = 0
            cyc_coop = 0
            st = state
            for _ in range(cyc_len):
                sc, cc, st = trans[st]
                cyc_score += sc
                cyc_coop += cc

            # Add score for all full cycles at once (O(1) instead of O(n))
            score += full_cycles * cyc_score
            opp_coop += full_cycles * cyc_coop

            # Simulate remaining leftover rounds
            st = state
            for _ in range(leftover):
                sc, cc, st = trans[st]
                score += sc
                opp_coop += cc

            return score, opp_coop

        # No cycle yet: simulate this round normally
        seen[state] = rnd
        sc, cc, state = trans[state]
        score += sc
        opp_coop += cc
        rnd += 1

    return score, opp_coop


def _play_ipd_general(agent_chrom: np.ndarray, oppo_chrom: np.ndarray,
                       memory_length: int, num_rounds: int,
                       is_selfaware: bool) -> Tuple[int, int]:
    """
    General IPD for any memory_length (>1). Uses pure Python for speed.

    Corresponds to Java: EvoThread_CA.Fitness_Calculas inner loop (general case)

    This handles memory_length > 1 where the state space is too large for
    efficient cycle detection (e.g., memory_length=2 has 256 possible states).
    Still uses pure Python integers instead of numpy for speed.

    Java algorithm:
        1. Build extended arrays: self[] = [memory_bits | chromosome]
           and oppo[] = [memory_bits | opponent_chromosome]
        2. Initialize memory bits (randomly for non-SRAC, to 0 for SRAC)
        3. For each round:
           a. Compute memory state -> lookup position in chromosome
           b. Get actions: Strategy1 = self[Position1+Memory_bits]
                          Strategy2 = oppo[Position2+Memory_bits]
           c. Compute payoff from payoff matrix
           d. Update memory bits with actions taken
        4. Return accumulated score

    Args:
        agent_chrom: numpy int8 array
        oppo_chrom: numpy int8 array
        memory_length: agent memory capacity (>1)
        num_rounds: number of IPD rounds
        is_selfaware: whether agent starts with cooperative memory

    Returns:
        (total_score, opponent_cooperation_count)
    """
    memory_bits = memory_length * 2

    # Convert to Python lists for fast scalar access
    # (Java: directly accesses self[k] and oppo[k])
    a_chrom = agent_chrom.tolist()
    o_chrom = oppo_chrom.tolist()

    # Initialize memory bits
    # (Java: if(selfaware) bits=0; else bits=random(0,1))
    if is_selfaware:
        self_mem = [0] * memory_bits
        oppo_mem = [0] * memory_bits
    else:
        self_mem = [_random.randint(0, 1) for _ in range(memory_bits)]
        oppo_mem = [_random.randint(0, 1) for _ in range(memory_bits)]

    score = 0
    opp_coop = 0

    for _ in range(num_rounds):
        # Compute position from memory bits (binary to decimal)
        # (Java: for(int z=0; z<Memory_bits; z++)
        #    Position1 += (int)(self[z] * Math.pow(2, Memory_bits-1-z));)
        pos1 = 0
        pos2 = 0
        for b in range(memory_bits):
            bit_val = 1 << (memory_bits - 1 - b)
            pos1 += self_mem[b] * bit_val
            pos2 += oppo_mem[b] * bit_val

        # Look up actions from chromosome
        # (Java: Strategy1 = self[Position1 + Memory_bits]; ...)
        action1 = a_chrom[pos1]
        action2 = o_chrom[pos2]

        # Payoff (Java: if(Strategy1==0 && Strategy2==0) score+=3; etc.)
        score += _PAY[action1][action2]
        if action2 == 0:
            opp_coop += 1

        # Update memory: shift old bits, insert new ones
        # (Java: for(int g=2; g<Memory_bits; g++) { self[g-2]=self[g]; ... }
        #        self[Memory_bits-2] = Strategy1; self[Memory_bits-1] = Strategy2; ...)
        if memory_bits > 2:
            self_mem[:memory_bits - 2] = self_mem[2:memory_bits]
            oppo_mem[:memory_bits - 2] = oppo_mem[2:memory_bits]
        if memory_bits > 0:
            self_mem[memory_bits - 2] = action1
            self_mem[memory_bits - 1] = action2
            oppo_mem[memory_bits - 2] = action2
            oppo_mem[memory_bits - 1] = action1

    return score, opp_coop


def compute_fitness(agent: Agent, neighbors: List[Tuple[int, int]],
                    agent_grid: np.ndarray, memory_length: int,
                    num_rounds: int, is_selfaware: bool = False) -> Tuple[int, List[int]]:
    """
    Compute the total fitness of an agent by playing IPD with all its neighbors.

    Corresponds to Java: EvoThread_CA.Fitness_Calculas / EvoThread_CA_Mix.Fitness_Calculas

    Java algorithm:
        int score = 0;
        for(int i=0; i<Neighbor.size(); i++) {
            // decode neighbor position
            // play IPD rounds
            score += current_score;
        }
        return score;

    The total fitness is the sum of payoffs from all IPD interactions
    across all neighbors in one generation.

    Args:
        agent: The agent whose fitness to compute
        neighbors: List of (row, col) neighbor positions
        agent_grid: 2D grid of all agents
        memory_length: Agent memory capacity (Java: parameter.MemoryLength)
        num_rounds: IPD rounds per opponent (Java: parameter.Times)
        is_selfaware: Whether this agent has self-awareness capability

    Returns:
        Tuple of (total_fitness, coop_times_list):
          - total_fitness: Sum of payoffs across all neighbor interactions
          - coop_times_list: List of cooperation counts, one per neighbor
            (Java: CoopTimes[i][j][k] for each neighbor k)
    """
    total_score = 0
    coop_times = []

    # Cache agent's chromosome reference for slight speedup in loop
    agent_chrom = agent.chromosome

    # Iterate over all neighbors
    # (Java: for(int i=0; i<Neighbor.size(); i++) { ... })
    for nei_row, nei_col in neighbors:
        neighbor = agent_grid[nei_row, nei_col]

        # Play IPD rounds between agent and this neighbor
        game_score, opp_coop = play_ipd_rounds(
            agent_chrom,
            neighbor.chromosome,
            memory_length,
            num_rounds,
            is_selfaware
        )

        total_score += game_score
        coop_times.append(opp_coop)

    return total_score, coop_times
