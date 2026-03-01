"""
visualization.py -- Visualization Module
=========================================

Corresponds to Java source files:
  - SimModel/PaintWorld.java   (2D lattice color grid visualization)
  - Analysis/Diagram.java      (multi-series line chart)
  - Analysis/Diagram1.java     (styled line chart with dashes/dots)

This module provides matplotlib-based visualization functions that replicate
the Java Swing custom painting in PaintWorld and Diagram classes.

Conversion notes:
  - Java's PaintWorld.paintComponent(Graphics g) manually draws filled rectangles
    for each cell with AWT Graphics2D. Python uses matplotlib's imshow() for the
    lattice grid, which is more efficient and provides built-in zoom/pan/save.
  - Java's Diagram.paintComponent(Graphics g) manually draws axes, tick marks,
    and line segments. Python uses matplotlib's standard plot() and axis formatting,
    which provides publication-quality output with minimal code.
  - Java's Diagram1 adds dashed/dotted stroke styles for paper figures.
    matplotlib supports these natively via linestyle parameter.
  - All visualization functions return matplotlib Figure objects that can be
    embedded in tkinter (via FigureCanvasTkAgg) or displayed standalone.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for GUI integration
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from typing import List, Optional, Tuple

from .parameters import (STRATEGY_COLORS, STRATEGY_LABELS, PAPER_4_COLORS,
                          FITNESS_QUARTILE_COLORS,
                          AVG_FITNESS_COLORS, EXPERIMENT_COLORS)


def create_lattice_figure(agent_grid: np.ndarray, strategy_length: int = 4,
                          title: str = "", figsize: Tuple[float, float] = (8, 8)) -> Figure:
    """
    Create a 2D lattice visualization showing agent strategy distribution.

    Corresponds to Java: PaintWorld.paintComponent(Graphics g)

    Java algorithm:
        for each cell (i,j):
            compute sIdx = binary-to-decimal(chromosome)
            fill rectangle at (i,j) with StrategyColor[sIdx]
        draw legend with all 16 strategy colors and labels

    The lattice view is the primary spatial visualization of the simulation,
    showing strategy clustering and evolution patterns across the 2D grid.
    Colored cells represent the 16 possible memory-1 deterministic strategies.

    Args:
        agent_grid: 2D numpy object array of Agent instances
                    (Java: Agent[][] agent in PaintWorld)
        strategy_length: Chromosome length (4 for memory-1)
        title: Optional title string
        figsize: Figure size in inches

    Returns:
        matplotlib Figure with the lattice visualization

    Why imshow instead of manual rectangle drawing:
        matplotlib's imshow() is optimized for rendering large 2D grids and
        provides anti-aliased output, colorbar support, and interactive zoom.
        The Java version draws individual rectangles with Graphics2D.fillRect(),
        which is the Swing equivalent but much more verbose.
    """
    rows, cols = agent_grid.shape
    # Build strategy index matrix (flat array + reshape for speed)
    # (Java: for each cell, compute sIdx from chromosome)
    flat = np.empty(rows * cols, dtype=int)
    k = 0
    for i in range(rows):
        for j in range(cols):
            flat[k] = agent_grid[i, j].get_strategy_index()
            k += 1
    strategy_indices = flat.reshape(rows, cols)

    # Create custom colormap from the 16 strategy colors
    # (Java: StrategyColor[sIdx] mapped to Color objects)
    num_strategies = 2 ** strategy_length
    cmap = ListedColormap(STRATEGY_COLORS[:num_strategies])

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    # Use pcolormesh with edgecolors so that each cell is drawn as a
    # separate quad with its own fill colour AND border.  This avoids
    # the sub-pixel misalignment that imshow + grid-overlay causes
    # (some cells showing a sliver of a neighbour's colour).
    # (Java: PaintWorld draws black grid lines first, then fills each
    #  cell at +1 pixel offset with BoardPieceSize-1, leaving 1-pixel
    #  black borders around every cell.)
    ax.pcolormesh(strategy_indices, cmap=cmap, vmin=0,
                  vmax=num_strategies - 1,
                  edgecolors='black', linewidth=0.5)
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)              # flip y so row-0 is at top
    ax.set_aspect('equal')

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Add legend showing key strategies
    # (Java: PaintWorld draws a legend with colored squares and labels
    #        for ALL-C (0), TFT (5), PAVLOV (6), ALL-D (15))
    from matplotlib.patches import Patch
    key_strategies = [0, 5, 6, 15]
    legend_elements = []
    for idx in key_strategies:
        label = STRATEGY_LABELS.get(idx, f"S{idx}")
        legend_elements.append(
            Patch(facecolor=STRATEGY_COLORS[idx], edgecolor='black',
                  label=f"{label} ({idx:04b})")
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              framealpha=0.8)

    fig.tight_layout()
    return fig


def create_strategy_dynamics_chart(counts: np.ndarray,
                                   colors: Optional[List[str]] = None,
                                   labels: Optional[List[str]] = None,
                                   title: str = "Evolutionary Dynamics",
                                   ylabel: str = "Number of Agents",
                                   figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Create a multi-series line chart showing strategy population over generations.

    Corresponds to Java: Diagram.paintComponent(Graphics g) used by Statistic1

    Java algorithm:
        1. Draw X axis (Generations) and Y axis (Numbers)
        2. For each strategy (color series):
            draw connected line segments from (g, count[g]) to (g+1, count[g+1])

    This is the primary chart for analyzing evolutionary dynamics, showing
    how the population of each strategy changes over time.

    Args:
        counts: Array of shape (generations, num_series)
                (Java: int[][] data in Diagram)
        colors: List of hex color strings, one per series
                (Java: Color[] color in Diagram)
        labels: List of label strings, one per series
        title: Chart title
        ylabel: Y-axis label
        figsize: Figure size in inches

    Returns:
        matplotlib Figure with the line chart
    """
    num_gen, num_series = counts.shape
    generations = np.arange(num_gen)

    if colors is None:
        colors = STRATEGY_COLORS[:num_series]
    if labels is None:
        labels = [STRATEGY_LABELS.get(i, f"S{i}") for i in range(num_series)]

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    # Plot each strategy series
    # (Java: for each series, draw line segments with corresponding color)
    for s in range(num_series):
        ax.plot(generations, counts[:, s], color=colors[s % len(colors)],
                label=labels[s] if s < len(labels) else f"S{s}",
                linewidth=1.5)

    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, num_gen - 1)
    # (Java: Diagram uses max_y * 1.5 for Y-axis upper bound)
    ax.set_ylim(0, max(1, int(np.max(counts) * 1.5)))
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_four_strategy_chart(four_counts: np.ndarray,
                               title: str = "Four Key Strategies",
                               figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Create a styled line chart for the four key strategies (paper format).

    Corresponds to Java: Diagram1 used by DiagramFrame.Statistic8()

    Java:
        tdata[g][0] = data[g][0]   // ALL-C
        tdata[g][1] = data[g][5]   // TFT
        tdata[g][2] = data[g][6]   // PAVLOV
        tdata[g][3] = data[g][15]  // ALL-D
        Diagram1 uses different stroke styles (dashed, dotted)

    Uses distinct line styles for publication-quality figures, matching the
    paper's Figure 7 (evolutionary dynamics of four strategies).

    Args:
        four_counts: Array of shape (generations, 4) for [ALL-C, TFT, PAVLOV, ALL-D]
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    labels = ['ALL-C', 'TFT', 'PAVLOV', 'ALL-D']
    # (Java: Diagram1 uses different BasicStroke styles)
    line_styles = ['-', '--', '-.', ':']

    num_gen = four_counts.shape[0]
    generations = np.arange(num_gen)

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    for s in range(4):
        ax.plot(generations, four_counts[:, s],
                color=PAPER_4_COLORS[s],
                linestyle=line_styles[s],
                label=labels[s],
                linewidth=2.0)

    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Number of Agents', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, num_gen - 1)
    # (Java: Diagram1 uses max_y * 1.5 for Y-axis upper bound)
    ax.set_ylim(0, max(1, int(np.max(four_counts) * 1.5)))
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_avg_fitness_chart(avg_fitness: np.ndarray,
                             title: str = "Average Fitness",
                             ylabel: str = "Average Fitness",
                             label: str = "Average Fitness",
                             figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Create a single-series fitness chart over generations.

    Corresponds to Java: Diagram used by DiagramFrame.Statistic3() / Statistic4()

    Args:
        avg_fitness: Array of shape (generations, 1) or (generations,)
        title: Chart title
        ylabel: Y-axis label (default "Average Fitness";
                use "Total Fitness" for Statistic4)
        label: Legend label for the line
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    if avg_fitness.ndim == 2:
        avg_fitness = avg_fitness[:, 0]

    num_gen = len(avg_fitness)
    generations = np.arange(num_gen)

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    # (Java: single green line — myColor2[0] = Green(0,255,0))
    ax.plot(generations, avg_fitness, color=AVG_FITNESS_COLORS[0],
            linewidth=2.0, label=label)

    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, num_gen - 1)
    ax.set_ylim(0, max(1, int(np.max(avg_fitness) * 1.5)))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_fitness_quartile_chart(quartile_data: np.ndarray,
                                  title: str = "Top/Bottom 25% Fitness",
                                  figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Create a chart comparing top 25% and bottom 25% average fitness.

    Corresponds to Java: Diagram used by DiagramFrame.Statistic2()

    Args:
        quartile_data: Array of shape (generations, 2)
                       [:, 0] = bottom 25%, [:, 1] = top 25%
        title: Chart title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    num_gen = quartile_data.shape[0]
    generations = np.arange(num_gen)

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    # (Java: Statistic2 data[g][0]=bottom 25% with myColor1[0]=Red,
    #                     data[g][1]=top 25%    with myColor1[1]=Green)
    ax.plot(generations, quartile_data[:, 0], color=FITNESS_QUARTILE_COLORS[0],
            linewidth=2.0, label='Bottom 25%')
    ax.plot(generations, quartile_data[:, 1], color=FITNESS_QUARTILE_COLORS[1],
            linewidth=2.0, label='Top 25%')

    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel('Average Fitness', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, num_gen - 1)
    ax.set_ylim(0, max(1, int(np.max(quartile_data) * 1.5)))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_experiment_comparison_chart(fitness_data: np.ndarray,
                                       ratio_labels: List[str],
                                       title: str = "Experiment Comparison",
                                       ylabel: str = "Average Fitness",
                                       figsize: Tuple[float, float] = (10, 6)) -> Figure:
    """
    Create a comparison chart for multiple experiment runs with different SRAC ratios.

    Corresponds to Java: DiagramFrame.Statistic5/Statistic6 (experiment comparison charts)

    This produces the charts shown in Figures 5 and 6 of the paper, comparing
    average payoff curves for 0%, 10%, 30%, 50%, and 100% SRAC agent mixes.

    Args:
        fitness_data: Array of shape (generations, num_experiments)
        ratio_labels: List of labels for each experiment (e.g., ["0%", "10%", "30%"])
        title: Chart title
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    num_gen, num_exp = fitness_data.shape
    generations = np.arange(num_gen)

    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)

    for e in range(num_exp):
        ax.plot(generations, fitness_data[:, e],
                color=EXPERIMENT_COLORS[e % len(EXPERIMENT_COLORS)],
                linewidth=2.0, label=ratio_labels[e] if e < len(ratio_labels) else f"Exp {e}")

    ax.set_xlabel('Generations', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, num_gen - 1)
    ax.set_ylim(0, max(1, int(np.max(fitness_data) * 1.5)))
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
