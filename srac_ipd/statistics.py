"""
statistics.py -- Statistics Computation
=======================================

Corresponds to Java source files:
  - Analysis/DiagramFrame.java  (methods: Statistic1 through Statistic8)
  - Analysis/StatisticData.java (Serializable container for experiment stats)

This module computes various statistics from evolution history data:
  1. Strategy population counts per generation (16 strategies)
  2. Top/bottom 25% average fitness per generation
  3. Average fitness per generation
  8. Paper-format 4-strategy subset (ALL-C, TFT, PAVLOV, ALL-D)

Note: The following were removed because the corresponding Java menu items
are commented out (i.e. never called in the Java version):
  4. [Removed] Total fitness per generation (Java Statistic4 / Diagram4)
  5. [Removed] Per-experiment total fitness (Java Statistic5, only from disabled Experiment_CA/SW)
  7. [Removed] Paper-format 5-strategy subset (Java Statistic7 / Load Avg E_CA_5/SW_5)

Conversion notes:
  - Java computes statistics inside DiagramFrame constructors and Statistic*()
    methods, tightly coupled with GUI rendering.
    Python separates computation (this module) from visualization (visualization.py),
    following the separation-of-concerns principle.
  - Java stores stats as int[][] (generations x strategies).
    Python uses numpy arrays for efficient computation using vectorized operations.
  - scipy.stats is available for advanced statistical analysis if needed.
"""

import numpy as np
from typing import List, Optional

from .agent import Agent


def compute_strategy_counts(history: List[np.ndarray], strategy_length: int = 4) -> np.ndarray:
    """
    Count the number of agents using each strategy in each generation.

    Corresponds to Java: DiagramFrame.Statistic1(MyParameter, Agent[][][])

    Java algorithm:
        for each generation g:
            for each agent (i,j):
                compute strategy index from chromosome (binary to decimal)
                data[g][index]++;

    This produces the "evolutionary dynamics" data showing how strategy
    populations change over time -- the primary analytical output of the simulation.

    Args:
        history: List of agent grids, one per generation
                 (Java: Agent[][][] agent -- [generation][row][col])
        strategy_length: Chromosome length (default 4 for memory-1)

    Returns:
        numpy array of shape (generations, num_strategies) where
        num_strategies = 2^strategy_length (16 for memory-1)
    """
    num_generations = len(history)
    num_strategies = 2 ** strategy_length
    # (Java: int data[][] = new int[Gen][StrNum];)
    counts = np.zeros((num_generations, num_strategies), dtype=int)

    for g in range(num_generations):
        grid = history[g]
        rows, cols = grid.shape
        # Use numpy bincount for fast histogram counting (avoids Python loop per agent)
        indices = np.empty(rows * cols, dtype=int)
        k = 0
        for i in range(rows):
            for j in range(cols):
                indices[k] = grid[i, j].get_strategy_index()
                k += 1
        gen_counts = np.bincount(indices, minlength=num_strategies)
        counts[g, :] = gen_counts[:num_strategies]

    return counts


def compute_fitness_quartiles(history: List[np.ndarray]) -> np.ndarray:
    """
    Compute the average fitness of the top 25% and bottom 25% agents per generation.

    Corresponds to Java: DiagramFrame.Statistic2(MyParameter, Agent[][][])

    Java algorithm:
        1. Copy all agents into a 1D array per generation
        2. Sort by fitness
        3. Average bottom 25% -> data[g][0]
        4. Average top 25% -> data[g][1]

    This shows the fitness gap between the best and worst performing agents,
    indicating the degree of inequality in the population.

    Args:
        history: List of agent grids, one per generation

    Returns:
        numpy array of shape (generations, 2) where
        [:, 0] = average fitness of bottom 25%
        [:, 1] = average fitness of top 25%
    """
    num_generations = len(history)
    # (Java: int data[][] = new int[Gen][2];)
    quartile_data = np.zeros((num_generations, 2), dtype=int)

    for g in range(num_generations):
        grid = history[g]
        rows, cols = grid.shape

        # Collect all fitness values into a numpy array efficiently
        # (Java: tempAgent[g][i*Size+j] = agent[g][i][j].copy(); then sort by fitness)
        all_fitness = np.empty(rows * cols, dtype=int)
        k = 0
        for i in range(rows):
            for j in range(cols):
                all_fitness[k] = grid[i, j].fitness
                k += 1

        # Sort fitness values (ascending)
        # (Java: bubble sort by fitness)
        all_fitness.sort()

        n = len(all_fitness)
        percent_25 = n // 4

        # Bottom 25% average
        # (Java: for(int x=0; x<Percent25; x++) Low25 += tempAgent[g][x].getFitness();
        #        data[g][0] = (int)(Low25/Percent25);)
        quartile_data[g, 0] = int(np.mean(all_fitness[:percent_25]))

        # Top 25% average
        # (Java: for(int y=Size*Size-Percent25; y<Size*Size; y++) High25 += ...;
        #        data[g][1] = (int)(High25/Percent25);)
        quartile_data[g, 1] = int(np.mean(all_fitness[n - percent_25:]))

    return quartile_data


def _sum_fitness(grid: np.ndarray) -> int:
    """
    Sum fitness of all agents in a grid using pure Python for speed.
    Avoids creating intermediate arrays for a simple summation.
    """
    rows, cols = grid.shape
    total = 0
    for i in range(rows):
        for j in range(cols):
            total += grid[i, j].fitness
    return total


def compute_avg_fitness(history: List[np.ndarray], num_agents: int = 2500) -> np.ndarray:
    """
    Compute the average fitness per generation.

    Corresponds to Java: DiagramFrame.Statistic3(MyParameter, Agent[][][])

    Java algorithm:
        for each generation g:
            sum all agent fitness values
            data[g][0] = total_fitness / 2500

    This is the key metric for measuring overall societal benefit,
    as described in the paper's results section.

    Args:
        history: List of agent grids, one per generation
        num_agents: Total number of agents (default 2500 = 50*50)

    Returns:
        numpy array of shape (generations, 1)
    """
    num_generations = len(history)
    # (Java: int data[][] = new int[Gen][1];)
    avg_data = np.zeros((num_generations, 1), dtype=int)

    for g in range(num_generations):
        # (Java: data[g][0] /= 2500;)
        avg_data[g, 0] = _sum_fitness(history[g]) // num_agents

    return avg_data


# [Removed] compute_total_fitness()
# Corresponded to Java DiagramFrame.Statistic4 (Total Fitness per generation).
# Removed because Java version also commented out this chart:
#   MyMenuBar.java line 76: //setMenuItem(analysis,"Diagram4",ml);
# If needed in the future, simply sum all agent fitness per generation
# without dividing by num_agents (unlike compute_avg_fitness which divides).


def extract_four_strategies(counts: np.ndarray) -> np.ndarray:
    """
    Extract the four key strategies (ALL-C, TFT, PAVLOV, ALL-D) from full counts.

    Corresponds to Java: DiagramFrame.Statistic8(MyParameter, int[][])

    Java algorithm:
        for each generation:
            tdata[g][0] = data[g][0]   // ALL-C (index 0)
            tdata[g][1] = data[g][5]   // TFT (index 5)
            tdata[g][2] = data[g][6]   // PAVLOV (index 6)
            tdata[g][3] = data[g][15]  // ALL-D (index 15)

    These are the four most important strategies analyzed in the paper:
    - ALL-C (0000): Always cooperate -- vulnerable but socially beneficial
    - TFT (0101): Tit-for-tat -- reciprocal strategy
    - PAVLOV (0110): Win-stay, lose-shift -- adaptive strategy
    - ALL-D (1111): Always defect -- exploitative strategy

    Args:
        counts: Strategy counts array of shape (generations, 16)

    Returns:
        numpy array of shape (generations, 4) for [ALL-C, TFT, PAVLOV, ALL-D]
    """
    num_gen = counts.shape[0]
    # (Java: int[][] tdata = new int[Gen][4];)
    four = np.zeros((num_gen, 4), dtype=int)
    four[:, 0] = counts[:, 0]   # ALL-C (index 0)
    four[:, 1] = counts[:, 5]   # TFT   (index 5)
    four[:, 2] = counts[:, 6]   # PAVLOV (index 6)
    four[:, 3] = counts[:, 15]  # ALL-D  (index 15)
    return four


# [Removed] extract_five_strategies()
# Corresponded to Java DiagramFrame.Statistic7 (5-strategy paper-format chart).
# Extracted indices [0, 3, 5, 6, 7] = [ALL-C, S3, TFT, PAVLOV, S7]
# with Java Parameter.myColor4 (5 colors: Black, Green, Yellow, Cyan, Magenta).
# Removed because Java version also commented out the menu items:
#   MyMenuBar.java line 62: //setMenuItem(experiment,"Load Avg E_CA_5",ml);
#   MyMenuBar.java line 63: //setMenuItem(experiment,"Load Avg E_SW_5",ml);
# If needed in the future, extract counts[:, [0, 3, 5, 6, 7]] from
# the full 16-strategy counts array.
