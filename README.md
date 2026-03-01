# SRAC-Agent

**Self-Reputation Awareness Component in Evolutionary Spatial IPD Game**

A Python 3 simulator for studying the influence of self-reputation awareness on agent behavior in evolutionary spatial Iterated Prisoner's Dilemma (IPD) games. This project is a faithful port of the original Java simulator developed at NCTU CIS Learning Technique Lab (2004-2005).

## Overview

This simulator implements the model described in:

> Chung-Yuan Huang and Chun-Liang Lee, "Influences of Agents with a Self-Reputation Awareness Component in an Evolutionary Spatial IPD Game," *PLoS ONE*.

The system simulates a population of agents arranged on a 2D spatial network, playing the Iterated Prisoner's Dilemma with their neighbors. Agents evolve over generations through genetic crossover and mutation. The key innovation is the **Self-Reputation Awareness Component (SRAC)**, where agents can evaluate their own fitness and reputation relative to neighbors and adapt their strategies accordingly.

## Features

- **Multi-Agent Spatial Simulation** -- N×N grid of agents (default 50×50 = 2,500 agents) playing IPD
- **Network Topologies** -- Cellular Automata (CA) regular grids and Small-World (SW) networks
- **Evolutionary Dynamics** -- Selection, crossover, and mutation based on neighborhood fitness
- **SRAC Mechanism** -- Self-aware agents detect poor fitness/reputation and learn from "socially good" neighbors
- **16 Deterministic Strategies** -- Memory-1 binary chromosomes including ALL-C, TFT, PAVLOV, ALL-D
- **Optimized IPD Engine** -- Cycle detection reduces computation from O(n) to O(1) per game
- **Interactive GUI** -- Tkinter-based interface with real-time visualization and generation scrubbing
- **CLI Mode** -- Headless execution for scripted batch experiments
- **Rich Visualization** -- Strategy dynamics charts, fitness trends, spatial lattice display

## Project Structure

```
SRAC-Agent/
├── main.py                # Entry point (GUI & CLI modes)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata & build config
└── srac_ipd/              # Main package
    ├── __init__.py        # Package metadata
    ├── parameters.py      # Configuration & constants
    ├── agent.py           # Agent model with chromosomes
    ├── network.py         # Network topologies (CA, SW)
    ├── ipd_game.py        # IPD game engine (optimized)
    ├── evolution.py       # Evolutionary algorithms & SRAC
    ├── statistics.py      # Analysis computations
    ├── visualization.py   # matplotlib charts & lattice display
    └── gui.py             # Tkinter GUI interface
```

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

## Origin

This Python implementation is a faithful port of the Java simulator originally developed by:

- **Original Author:** HikiChen, NCTU CIS Learning Technique Lab (2004–2005)
- **Python Conversion:** Assisted by Claude (Anthropic)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
