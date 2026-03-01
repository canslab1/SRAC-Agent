"""
network.py -- Network Topology Generation
==========================================

Corresponds to Java source files:
  - SimModel/CAN.java  (Cellular Automata Network)
  - SimModel/SWN.java  (Small-World Network, extends CAN)

This module creates the social interaction networks on which agents play IPD games.
Two network types are supported:
  1. Cellular Automata (CAN): Regular 2D toroidal grid with Moore neighborhood
  2. Small-World (SWN): CAN base + random long-range shortcuts

Conversion notes:
  - Java uses LinkedList[][] where each cell stores neighbor indices as Integer objects.
    Python uses networkx.Graph which provides a clean, well-tested graph API with
    built-in support for neighbor queries, graph algorithms, and visualization.
  - Java encodes neighbors as single integers (row * BoardSize + col).
    Python stores neighbors as (row, col) tuples, which is more readable.
    Helper functions convert between the two representations.
  - networkx provides efficient adjacency queries via G.neighbors(node), replacing
    the Java LinkedList traversal pattern.
  - The toroidal boundary condition (periodic boundary) is preserved exactly:
    Java: if(row<0) row=BoardSize+row; ... (row%BoardSize)*BoardSize+(col%BoardSize)
    Python: row % board_size, col % board_size
"""

import networkx as nx
import numpy as np
from typing import List, Tuple


def create_can(board_size: int, radius: int = 1) -> nx.Graph:
    """
    Create a Cellular Automata Network (regular 2D toroidal grid).

    Corresponds to Java: CAN.java constructor and CAN.CreateCA()

    Java algorithm (CAN.CreateCA):
        for each cell (i,j):
            for k in range(-Radius, Radius+1):
                for l in range(-Radius, Radius+1):
                    if not (k==0 and l==0):
                        compute row/col with toroidal wrapping
                        add neighbor to LinkedList if not already present

    This creates a Moore neighborhood (8 neighbors for radius=1) with
    periodic (toroidal) boundary conditions, meaning the grid wraps around
    so that edge cells connect to the opposite side.

    Args:
        board_size: Grid dimension W = H (Java: CAN.BoardSize)
        radius: k-neighborhood radius (Java: CAN.Radius)
                radius=1 -> Moore neighborhood (8 neighbors per cell)

    Returns:
        networkx.Graph with nodes as (row, col) tuples and edges representing
        neighbor relationships.

    Why networkx instead of LinkedList[][]:
        networkx provides efficient graph operations (neighbor lookup, shortest
        path, clustering coefficient, etc.) that would require manual implementation
        with raw LinkedLists. It also enables easy visualization and analysis
        of the network structure using standard graph algorithms.
    """
    G = nx.Graph()
    G.graph['topology_type'] = 'Cellular Automata'
    G.graph['board_size'] = board_size
    G.graph['radius'] = radius

    # Add all nodes (Java: implicitly created when adding to LinkedList)
    for i in range(board_size):
        for j in range(board_size):
            G.add_node((i, j))

    # Add edges based on k-neighborhood with toroidal boundary
    # (Java: CAN.CreateCA() -- nested loop over offsets -Radius to +Radius)
    for i in range(board_size):
        for j in range(board_size):
            for dk in range(-radius, radius + 1):
                for dl in range(-radius, radius + 1):
                    # Skip self-connection (Java: if(!(k==0 && l==0)))
                    if dk == 0 and dl == 0:
                        continue
                    # Toroidal wrapping
                    # (Java: if(row<0) row=BoardSize+row;
                    #        ... (row%BoardSize)*BoardSize + (col%BoardSize))
                    ni = (i + dk) % board_size
                    nj = (j + dl) % board_size
                    # Add edge if not already present
                    # (Java: if(!board[i][j].contains(...)) board[i][j].addLast(...))
                    # networkx.Graph automatically prevents duplicate edges
                    G.add_edge((i, j), (ni, nj))

    return G


def create_swn(board_size: int, radius: int = 1, shortcuts: int = 1) -> nx.Graph:
    """
    Create a Small-World Network (CAN base + random shortcuts).

    Corresponds to Java: SWN.java constructor and SWN.CreateShortcuts()

    Java algorithm (SWN.CreateShortcuts):
        1. Start with a CAN (super(BoardSize, Radius))
        2. For each cell (i,j), add 'Shortcuts' random long-range connections:
           a. Randomly pick a target cell
           b. If target is not self and not already a neighbor:
              - Add bidirectional link
              - Increment shortcut counter for both cells
           c. Limit: each cell can have at most Shortcuts+3 added shortcuts

    This preserves the high local clustering of the CAN while adding
    long-range connections that dramatically reduce average path length,
    creating the "small-world" property described by Watts-Strogatz.

    Args:
        board_size: Grid dimension (Java: SWN.BoardSize, inherited from CAN)
        radius: k-neighborhood radius (Java: SWN.Radius, inherited from CAN)
        shortcuts: Number of shortcuts per node (Java: SWN.Shortcuts)

    Returns:
        networkx.Graph with CAN base topology plus random shortcut edges.

    Why a custom small-world implementation instead of nx.watts_strogatz_graph:
        The Java code uses a specific algorithm that adds a fixed number of
        shortcuts per node (rather than rewiring existing edges with probability p).
        To maintain exact algorithmic equivalence with the Java simulator,
        we replicate the same shortcut-adding logic.
    """
    # Step 1: Create the base CAN topology (Java: super(BoardSize, Radius))
    G = create_can(board_size, radius)
    G.graph['topology_type'] = 'Small-World'
    G.graph['shortcuts'] = shortcuts

    # Step 2: Add random shortcuts
    # (Java: SWN.CreateShortcuts(int Shortcuts))

    # Track how many shortcuts have been added to each node
    # (Java: int[][] CurShortcuts = new int[BoardSize][BoardSize])
    shortcut_count = np.zeros((board_size, board_size), dtype=int)

    # Maximum shortcuts per node (Java: CurShortcuts[row_idx][col_idx] < this.Shortcuts+3)
    max_shortcuts_per_node = shortcuts + 3

    # Safety limit: maximum random attempts per cell before giving up.
    # Prevents an infinite loop when the network is too densely connected
    # for additional shortcuts to be found.  The Java version has the same
    # latent infinite-loop bug; this limit is a Python improvement.
    max_attempts_per_shortcut = board_size * board_size * 2

    for i in range(board_size):
        for j in range(board_size):
            # For each cell, add up to 'shortcuts' random connections
            # (Java: for(int k=CurShortcuts[i][j]; k<Shortcuts; k++))
            attempts = 0
            while shortcut_count[i, j] < shortcuts:
                attempts += 1
                if attempts > max_attempts_per_shortcut:
                    break   # Give up — no valid target can be found

                # Randomly pick a target cell
                # (Java: int row_idx=(int)(Math.random()*this.BoardSize);
                #        int col_idx=(int)(Math.random()*this.BoardSize);)
                ri = np.random.randint(0, board_size)
                rj = np.random.randint(0, board_size)

                # Check: not self (Java: if(idx != i*BoardSize+j))
                if (ri, rj) == (i, j):
                    continue

                # Check: not already a neighbor
                # (Java: if(!CheckNeighbor(board[i][j], idx)))
                if G.has_edge((i, j), (ri, rj)):
                    continue

                # Check: target hasn't exceeded its shortcut limit
                # (Java: if(CurShortcuts[row_idx][col_idx] < this.Shortcuts+3))
                if shortcut_count[ri, rj] >= max_shortcuts_per_node:
                    continue

                # Add bidirectional shortcut
                # (Java: board[i][j].addLast(new Integer(idx));
                #        board[row_idx][col_idx].addLast(new Integer(i*BoardSize+j));)
                G.add_edge((i, j), (ri, rj))
                shortcut_count[i, j] += 1
                shortcut_count[ri, rj] += 1
                attempts = 0    # Reset counter for the next shortcut of this cell

    return G


def get_neighbors(G: nx.Graph, row: int, col: int) -> List[Tuple[int, int]]:
    """
    Get the list of neighbor positions for a given node.

    Corresponds to Java: myCAN.board[i][j] (LinkedList of encoded neighbor indices)

    In Java, neighbors are stored as encoded integers:
        int NeiIndex = Integer.parseInt(Neighbor.get(i).toString());
        int row = NeiIndex / parameter.BoardSize;
        int col = NeiIndex % parameter.BoardSize;

    In Python with networkx, neighbors are directly available as (row, col) tuples.

    Args:
        G: The network graph
        row: Node row position
        col: Node column position

    Returns:
        List of (row, col) tuples for all neighbors
    """
    return list(G.neighbors((row, col)))


def get_neighbor_count(G: nx.Graph, row: int, col: int) -> int:
    """
    Get the number of neighbors for a given node.

    Corresponds to Java: myCAN.board[i][j].size()

    Args:
        G: The network graph
        row: Node row position
        col: Node column position

    Returns:
        Number of neighbors (degree of the node)
    """
    return G.degree((row, col))


def get_network_info(G: nx.Graph) -> dict:
    """
    Get summary statistics about the network topology.

    This has no direct Java equivalent but is useful for analysis.
    The manuscript discusses network properties like separation (avg path length)
    and clustering coefficient.

    Returns:
        Dictionary with topology_type, num_nodes, num_edges, avg_degree,
        and avg_clustering_coefficient.
    """
    info = {
        'topology_type': G.graph.get('topology_type', 'Unknown'),
        'board_size': G.graph.get('board_size', 0),
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': sum(d for _, d in G.degree()) / G.number_of_nodes(),
    }
    # Clustering coefficient (computationally expensive for large graphs)
    if G.number_of_nodes() <= 2500:
        info['avg_clustering'] = nx.average_clustering(G)
    return info
