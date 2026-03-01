"""
agent.py -- Agent Model
========================

Corresponds to Java source file:
  - SimModel/Agent.java

Each agent in the simulation has:
  - A position (row, col) on the 2D grid
  - A binary strategy chromosome of length 2^(2*memory_length)
  - Fitness, reputation, and real-reputation scores

Conversion notes:
  - Java's Agent uses byte[] for the chromosome; Python uses a numpy int8 array
    for efficient vectorized operations.
  - Java's static NumberOfAgent counter is replaced by a class variable.
    Note: In the Java code this counter was mainly used for debugging.
  - Java's Serializable interface is replaced by pickle compatibility
    (Python dataclasses are pickle-compatible by default).
  - Java's copy() method (via copy constructor) is replaced by a dedicated
    copy() method using numpy array copy.
"""

import numpy as np
from typing import Optional


class Agent:
    """
    Represents a single IPD agent with a binary strategy chromosome.

    Corresponds to Java: SimModel/Agent.java

    Attributes:
        agent_id (int): Unique identifier (Java: Agent.ID)
        row (int): Row position on the 2D grid (Java: Agent.row)
        col (int): Column position on the 2D grid (Java: Agent.col)
        memory_length (int): How many past rounds the agent remembers (Java: Agent.memory_length)
        strategy_length (int): Chromosome length = 2^(2*memory_length) (Java: Agent.StrategyLength)
        chromosome (np.ndarray): Binary strategy array, dtype int8 (Java: Agent.Chromosome)
        fitness (int): Accumulated fitness score (Java: Agent.Fitness)
        reputation (int): Reputation score from neighbors (Java: Agent.Reputation)
        real_rep (int): Real reputation = weighted by cooperation times (Java: Agent.RealRep)
    """

    # Class-level agent counter (Java: Agent.NumberOfAgent -- static field)
    _next_id: int = 0

    # Precomputed power arrays for get_strategy_index(), keyed by strategy_length.
    # For strategy_length=4: _powers_cache[4] = (8, 4, 2, 1)
    # Avoids creating 2**np.arange(...) on every call (~3-5μs overhead per call).
    _powers_cache: dict = {}

    def __init__(self, memory_length: int, row: int, col: int,
                 chromosome: Optional[np.ndarray] = None):
        """
        Create a new agent with a random or specified chromosome.

        Corresponds to Java: Agent(int memory_length, int row, int column)

        Args:
            memory_length: Number of past rounds to remember (Java: memory_length param)
            row: Grid row position (Java: row param)
            col: Grid column position (Java: column param)
            chromosome: If provided, use this chromosome instead of random.
                        If None, generate a random binary chromosome.

        Java equivalent:
            public Agent(int memory_length, int row, int column) {
                // ... computes StrategyLength = 2^(2*memory_length)
                // ... randomly generates Chromosome[i] = (byte)(Math.random()*2)
            }
        """
        # Assign unique ID (Java: ID = this.NumberOfAgent; this.NumberOfAgent++;)
        self.agent_id = Agent._next_id
        Agent._next_id += 1

        # Grid position (Java: this.row = row; this.col = column;)
        self.row = row
        self.col = col

        # Memory capacity (Java: this.memory_length = memory_length;)
        self.memory_length = memory_length

        # Strategy chromosome length: 2^(2*memory_length)
        # (Java: StrategyLength = (int)Math.pow(2, 2*memory_length);)
        # For memory_length=1: 2^2 = 4 bits (the 16 deterministic strategies)
        self.strategy_length = int(2 ** (2 * memory_length))

        # Ensure power array is cached for this strategy length (optimization)
        if self.strategy_length not in Agent._powers_cache:
            Agent._powers_cache[self.strategy_length] = tuple(
                2 ** (self.strategy_length - 1 - k) for k in range(self.strategy_length)
            )

        # Fitness, reputation, real reputation -- all start at 0
        # (Java: Fitness=0; Reputation=0; RealRep=0;)
        self.fitness = 0
        self.reputation = 0
        self.real_rep = 0

        # Strategy chromosome: binary array
        # (Java: this.Chromosome = new byte[StrategyLength];
        #        for(int i=0;i<StrategyLength;i++) Chromosome[i]=(byte)(Math.random()*2);)
        if chromosome is not None:
            self.chromosome = chromosome.copy()
        else:
            # Random binary chromosome (0 or 1 for each bit)
            self.chromosome = np.random.randint(0, 2, size=self.strategy_length, dtype=np.int8)

    def copy(self) -> 'Agent':
        """
        Create a deep copy of this agent.

        Corresponds to Java: Agent.copy() -> new Agent(this)
        The Java copy constructor copies all fields and deep-copies the chromosome.

        Returns:
            A new Agent with identical state but independent chromosome array.
        """
        new_agent = Agent.__new__(Agent)
        new_agent.agent_id = self.agent_id
        new_agent.row = self.row
        new_agent.col = self.col
        new_agent.memory_length = self.memory_length
        new_agent.strategy_length = self.strategy_length
        new_agent.fitness = self.fitness
        new_agent.reputation = self.reputation
        new_agent.real_rep = self.real_rep
        # Deep copy of chromosome (Java: for(int i=0;i<StrategyLength;i++) this.Chromosome[i]=agent.Chromosome[i];)
        new_agent.chromosome = self.chromosome.copy()
        return new_agent

    def get_strategy_index(self) -> int:
        """
        Convert the binary chromosome to a decimal strategy index (0-15 for memory-1).

        Corresponds to Java: PaintWorld.paintComponent() and Agent.paintAgent()
            int sIdx = 0;
            for(int k=0; k<StrategyLength; k++)
                sIdx += (int)(Chromosome[k]) * (int)Math.pow(2, StrategyLength-k-1);

        This is used for color-mapping: each of the 16 strategies gets a unique color.

        Returns:
            Integer index (0 to 2^strategy_length - 1)

        Performance note:
            Uses precomputed powers tuple and pure Python sum instead of numpy
            operations, avoiding ~3-5μs numpy dispatch overhead per call.
            Called O(N * generations) times from statistics and visualization.
        """
        # Convert binary array to decimal using precomputed powers
        # chromosome = [b0, b1, b2, b3] -> b0*8 + b1*4 + b2*2 + b3*1
        powers = Agent._powers_cache[self.strategy_length]
        chrom = self.chromosome
        idx = 0
        for k in range(self.strategy_length):
            if chrom[k]:
                idx += powers[k]
        return idx

    @staticmethod
    def reset_id_counter():
        """Reset the global agent ID counter. Useful when starting new simulations."""
        Agent._next_id = 0

    def __repr__(self) -> str:
        """String representation for debugging (Java: Agent.paintAgent())."""
        chrom_str = ''.join(str(b) for b in self.chromosome)
        return (f"Agent(id={self.agent_id}, pos=({self.row},{self.col}), "
                f"chr={chrom_str}, fit={self.fitness}, rep={self.reputation})")


def create_agent_grid(board_size: int, memory_length: int) -> np.ndarray:
    """
    Create a board_size x board_size grid of agents with random chromosomes.

    Corresponds to Java code in SimFrame constructor and Setting dialog:
        Agent[][] myAgent = new Agent[BoardSize][BoardSize];
        for(int i=0; i<BoardSize; i++)
            for(int j=0; j<BoardSize; j++)
                myAgent[i][j] = new Agent(MemoryLength, i, j);

    Args:
        board_size: Grid dimension (Java: Parameter.BoardSize)
        memory_length: Agent memory capacity (Java: Parameter.MemoryLength)

    Returns:
        2D numpy object array of Agent instances, shape (board_size, board_size)

    Why numpy object array instead of Python list-of-lists:
        Using np.ndarray for the grid allows us to leverage numpy's indexing
        capabilities, while still storing heterogeneous Agent objects.
        The Java version uses Agent[][] which is essentially the same concept.
    """
    Agent.reset_id_counter()
    grid = np.empty((board_size, board_size), dtype=object)
    for i in range(board_size):
        for j in range(board_size):
            grid[i, j] = Agent(memory_length, i, j)
    return grid


def copy_agent_grid(grid: np.ndarray) -> np.ndarray:
    """
    Deep-copy an entire agent grid.

    Corresponds to Java: EvoData.setEvoData() which deep-copies agent arrays
    using Agent.copy() for each element.

    Args:
        grid: 2D numpy object array of Agent instances

    Returns:
        New 2D numpy object array with deep-copied Agent instances
    """
    rows, cols = grid.shape
    new_grid = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            new_grid[i, j] = grid[i, j].copy()
    return new_grid
