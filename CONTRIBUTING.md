# Contributing to SRAC-Agent

Thank you for your interest in contributing to SRAC-Agent! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/canslab1/SRAC-Agent/issues) page to report bugs or request features.
- When reporting a bug, please include:
  - Python version (`python --version`)
  - Operating system
  - Steps to reproduce the issue
  - Expected vs. actual behavior
  - Relevant error messages or screenshots

### Submitting Changes

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SRAC-Agent.git
   cd SRAC-Agent
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make your changes** and test them.
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: brief description of your change"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a **Pull Request** on GitHub.

## Development Setup

```bash
git clone https://github.com/canslab1/SRAC-Agent.git
cd SRAC-Agent
pip install -r requirements.txt
python main.py  # Verify the GUI launches correctly
```

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints where practical.
- Keep functions focused and reasonably sized.
- Document non-obvious algorithms with comments referencing the paper or Java source.
- Visualization functions should return `matplotlib.figure.Figure` objects for GUI embedding.

## Project Architecture

| Module | Responsibility |
|--------|---------------|
| `parameters.py` | All configurable constants and the `SimParameters` dataclass |
| `agent.py` | Agent model with binary chromosome strategy |
| `network.py` | CA and Small-World network topology construction |
| `ipd_game.py` | IPD game engine with cycle detection optimization |
| `evolution.py` | Evolutionary selection, crossover, mutation, and SRAC mechanism |
| `statistics.py` | Post-simulation statistical analysis |
| `visualization.py` | All matplotlib chart generation functions |
| `gui.py` | Tkinter GUI, batch experiment runner, and result export |

## Areas for Contribution

- Performance optimizations for large grid sizes
- Additional network topologies (e.g., scale-free networks)
- New visualization modes and chart types
- Unit tests and integration tests
- Documentation improvements
- Support for extended memory lengths (memory > 1)
- Additional batch experiment output formats (e.g., JSON, HDF5)
- Parallel execution for batch experiments

## Questions?

Feel free to open an issue for any questions about the codebase or contribution process.
