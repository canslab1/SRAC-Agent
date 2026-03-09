# SRAC-Agent

**Self-Reputation Awareness Component in Evolutionary Spatial IPD Game**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python 3 simulator for studying the influence of self-reputation awareness on agent behavior in evolutionary spatial Iterated Prisoner's Dilemma (IPD) games. This project is a faithful port of the original Java simulator developed at NCTU CIS Learning Technique Lab (2004-2005).

## Overview

This simulator implements the model described in:

> Huang, C.-Y. & Lee, C.-L. (2014). Influences of Agents with a Self-Reputation Awareness Component in an Evolutionary Spatial IPD Game. *PLOS ONE*, 9(6), e99841. https://doi.org/10.1371/journal.pone.0099841

The system simulates a population of agents arranged on a 2D spatial network, playing the Iterated Prisoner's Dilemma with their neighbors. Agents evolve over generations through genetic crossover and mutation. The key innovation is the **Self-Reputation Awareness Component (SRAC)**, where agents can evaluate their own fitness and reputation relative to neighbors and adapt their strategies accordingly.

## Features

- **Multi-Agent Spatial Simulation** — N×N grid of agents (default 50×50 = 2,500 agents) playing IPD
- **Network Topologies** — Cellular Automata (CA) regular grids and Small-World (SW) networks
- **Evolutionary Dynamics** — Selection, crossover, and mutation based on neighborhood fitness
- **SRAC Mechanism** — Self-aware agents detect poor fitness/reputation and learn from "socially good" neighbors
- **16 Deterministic Strategies** — Memory-1 binary chromosomes including ALL-C, TFT, PAVLOV, ALL-D
- **Optimized IPD Engine** — Cycle detection reduces computation from O(n) to O(1) per game
- **Interactive GUI** — Tkinter-based interface with real-time visualization and generation scrubbing
- **CLI Mode** — Headless execution for scripted batch experiments
- **Batch Experiments** — Run multiple SRAC ratios with automatic result aggregation, CSV/PNG export
- **Rich Visualization** — Strategy dynamics, fitness trends, quartile analysis, spatial lattice display, cross-ratio comparison charts

## Installation

### Prerequisites

- Python 3.9 or higher
- tkinter (usually included with Python; on Linux: `sudo apt install python3-tk`)

### Setup

```bash
git clone https://github.com/canslab1/SRAC-Agent.git
cd SRAC-Agent
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Usage

### GUI Mode (Default)

```bash
python main.py
```

This launches the interactive Tkinter interface where you can:
- Configure all simulation parameters
- Run evolution with real-time progress
- Scrub through generations with a slider
- View multiple analysis charts
- Run batch experiments across SRAC ratios

### CLI Mode

```bash
python main.py --cli --board-size 30 --generations 50 --sa-ratio 0.1
```

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--board-size` | 50 | Grid dimension (N×N) |
| `--generations` | 100 | Number of evolutionary generations |
| `--memory-length` | 1 | Agent memory capacity |
| `--ipd-rounds` | 100 | IPD rounds per opponent pair |
| `--mutation-rate` | 0.01 | Bit-flip mutation probability |
| `--crossover-rate` | 0.7 | Single-point crossover probability |
| `--topology` | CA | Network topology: `CA` or `SW` |
| `--radius` | 1 | Neighborhood radius |
| `--shortcuts` | 1 | SW shortcuts per node |
| `--sa-ratio` | 0.0 | Self-aware agent fraction (0.0–1.0) |
| `--f-low` / `--f-high` | -1.0 / 1.0 | Fitness z-score thresholds |
| `--r-low` / `--r-high` | -1.0 / 1.0 | Reputation z-score thresholds |
| `--seed` | None | Random seed for reproducibility |
| `--output` | None | Save results to pickle file |
| `--plot` | False | Show matplotlib plots after run |

#### Example: SW Network with Self-Aware Agents

```bash
python main.py --cli \
    --board-size 30 \
    --generations 100 \
    --topology SW --shortcuts 2 \
    --sa-ratio 0.3 \
    --f-low -0.5 --f-high 0.5 \
    --r-low -0.5 --r-high 0.5 \
    --seed 42 \
    --output results.pkl \
    --plot
```

### Batch Experiments (GUI)

The GUI provides a batch experiment mode for systematic comparison across multiple SRAC mixing ratios:

1. Go to **Experiment → Run Batch Experiment**
2. Select network topology (CA or SW)
3. Enter SRAC ratios (e.g., `0, 0.1, 0.3, 0.5, 1.0`)
4. Set z-score thresholds and number of runs per ratio
5. Choose an output directory for results

#### Batch Output Files

The batch experiment automatically generates the following in the selected output directory:

| File | Description |
|------|-------------|
| `batch_results.pkl` | Complete results in pickle format (for further analysis) |
| `avg_fitness_comparison.csv` | Average fitness per generation across all ratios |
| `four_strategies_ratio_*.csv` | ALL-C, TFT, PAVLOV, ALL-D counts per ratio |
| `fitness_quartiles_ratio_*.csv` | Top/Bottom 25% fitness per ratio |
| `chart_avg_fitness_comparison.png` | Fitness comparison chart across ratios |
| `chart_four_strategies_ratio_*.png` | Four key strategy dynamics per ratio |
| `chart_fitness_quartiles_ratio_*.png` | Fitness quartile chart per ratio |
| `chart_strategy_*_comparison.png` | Per-strategy comparison across all ratios |

## Core Algorithms

### IPD Game Engine
Agents play the Iterated Prisoner's Dilemma using memory-1 deterministic strategies encoded as 4-bit binary chromosomes. The engine includes cycle detection that reduces per-game computation from O(n) to O(1) for typical configurations.

### Evolutionary Selection
Each generation, agents with low relative fitness in their neighborhood are replaced. Replacement candidates are generated through:
1. Copying the best neighbor's strategy
2. Crossover between random neighbors
3. Mutation of the above candidates

### SRAC Mechanism
Self-aware agents evaluate their fitness and reputation using z-score classification:
- Agents with **LOW** fitness or reputation seek out "socially good" neighbors (high fitness + high reputation)
- They copy the strategy of a randomly selected socially good neighbor
- This mechanism enables agents to escape poor strategies without relying solely on evolutionary pressure

## Visualization

The simulator provides the following chart types:

- **Strategy Dynamics** — Population trends of all 16 strategies over generations
- **Four Key Strategies** — Focused view on ALL-C, TFT, PAVLOV, and ALL-D with distinct line styles
- **Average Fitness** — Mean fitness trend across generations
- **Fitness Quartiles** — Top 25% vs. Bottom 25% fitness comparison
- **Spatial Lattice** — Color-coded N×N grid showing spatial strategy distribution
- **Cross-Ratio Comparison** — Per-strategy population trends across different SRAC ratios

## Origin

This Python implementation is a faithful port of the Java simulator originally developed by:

- **Original Author:** HikiChen, NCTU CIS Learning Technique Lab (2004–2005)
- **Python Conversion:** Assisted by Claude (Anthropic)

## Project Structure

```
SRAC-Agent/
├── main.py                # Entry point (GUI & CLI modes)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata & build config
├── LICENSE                # MIT License
├── CONTRIBUTING.md        # Contribution guidelines
└── srac_ipd/              # Main package
    ├── __init__.py        # Package metadata
    ├── parameters.py      # Configuration & constants
    ├── agent.py           # Agent model with chromosomes
    ├── network.py         # Network topologies (CA, SW)
    ├── ipd_game.py        # IPD game engine (optimized)
    ├── evolution.py       # Evolutionary algorithms & SRAC
    ├── statistics.py      # Analysis computations
    ├── visualization.py   # matplotlib charts & lattice display
    └── gui.py             # Tkinter GUI & batch experiment runner
```

## Authors

- **Chung-Yuan Huang** (黃崇源) — Department of Computer Science and Information Engineering, Chang Gung University, Taiwan (gscott@mail.cgu.edu.tw)
- **Chun-Liang Lee** — National Chiao Tung University, Taiwan

## References

1. Huang, C.-Y. & Lee, C.-L. (2014). Influences of Agents with a Self-Reputation Awareness Component in an Evolutionary Spatial IPD Game. *PLOS ONE*, 9(6), e99841. https://doi.org/10.1371/journal.pone.0099841

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
