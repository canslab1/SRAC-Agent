"""
main.py -- Entry Point for SRAC-IPD Simulator
===============================================

Corresponds to Java: Project.java

Java:
    public class Project {
        public static void main(String[] args) {
            SimWorld frame = new SimWorld();
            frame.setVisible(true);
        }
    }

This script provides two usage modes:
  1. GUI mode (default): Launch the tkinter graphical interface
  2. CLI mode (--cli): Run a headless simulation with command-line arguments

The GUI mode replicates the Java Swing interface using tkinter.
The CLI mode is a Python-specific addition for scripted batch experiments.

Usage:
    python main.py             # Launch GUI
    python main.py --cli       # Run headless simulation with defaults
    python main.py --cli --board-size 30 --generations 50 --sa-ratio 0.1
"""

import argparse
import numpy as np


def run_gui():
    """
    Launch the GUI application.

    Corresponds to Java: Project.main() -> new SimWorld(); frame.setVisible(true);
    """
    from srac_ipd.gui import SimulationApp
    app = SimulationApp()
    app.mainloop()


def run_cli(args):
    """
    Run a headless simulation from command-line arguments.

    This has no direct Java equivalent -- it's a Python-specific convenience
    for scripted experiments and reproducibility.

    The simulation flow follows the same steps as the Java EvoThread_CA_Mix.run():
        1. Initialize parameters
        2. Create agent grid and network topology
        3. Run evolutionary simulation
        4. Compute and display statistics
    """
    from srac_ipd.parameters import SimParameters
    from srac_ipd.agent import create_agent_grid
    from srac_ipd.network import create_can, create_swn, get_network_info
    from srac_ipd.evolution import EvolutionEngine
    from srac_ipd.statistics import (compute_strategy_counts, compute_avg_fitness,
                                     extract_four_strategies)

    # Set random seed for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)

    # Configure parameters
    # (Java: Parameter defaults and Setting dialog)
    params = SimParameters(
        sim_name=args.name,
        board_size=args.board_size,
        memory_length=args.memory_length,
        ipd_rounds=args.ipd_rounds,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
    )
    params.update_derived()

    # Set topology
    radius = args.radius
    shortcuts = args.shortcuts
    if args.topology == "CA":
        params.set_topology_ca(radius)
    else:
        params.set_topology_sw(radius, shortcuts)

    # Set self-awareness
    if args.sa_ratio > 0:
        params.set_selfaware_params(
            True, args.sa_ratio,
            args.f_low, args.f_high, args.r_low, args.r_high
        )

    # Print configuration
    print("=" * 60)
    print("SRAC-IPD Simulator (Python 3)")
    print("=" * 60)
    print(f"  Simulation: {params.sim_name}")
    print(f"  Board Size: {params.board_size} x {params.board_size} ({params.num_agents} agents)")
    print(f"  Memory Length: {params.memory_length}")
    print(f"  Strategy Length: {params.strategy_length}")
    print(f"  IPD Rounds: {params.ipd_rounds}")
    print(f"  Generations: {params.generations}")
    print(f"  Mutation Rate: {params.mutation_rate}")
    print(f"  Crossover Rate: {params.crossover_rate}")
    print(f"  Topology: {params.topology_type} (radius={radius})")
    if args.topology == "SW":
        print(f"  Shortcuts: {shortcuts}")
    if args.sa_ratio > 0:
        print(f"  Self-Aware Ratio: {args.sa_ratio:.0%}")
        print(f"  F Thresholds: [{args.f_low}, {args.f_high}]")
        print(f"  R Thresholds: [{args.r_low}, {args.r_high}]")
    if args.seed is not None:
        print(f"  Random Seed: {args.seed}")
    print("=" * 60)

    # Create agent grid (Java: Agent[][] in SimFrame constructor)
    print("\nInitializing agents...")
    agent_grid = create_agent_grid(params.board_size, params.memory_length)

    # Create network (Java: CAN/SWN constructor)
    print("Creating network topology...")
    if args.topology == "CA":
        network = create_can(params.board_size, radius)
    else:
        network = create_swn(params.board_size, radius, shortcuts)

    net_info = get_network_info(network)
    print(f"  Nodes: {net_info['num_nodes']}, Edges: {net_info['num_edges']}, "
          f"Avg Degree: {net_info['avg_degree']:.1f}")

    # Run evolution (Java: EvoThread_CA_Mix.run())
    print("\nRunning evolution...")
    engine = EvolutionEngine(agent_grid, network, params)

    for stats in engine.run():
        gen = stats['generation']
        avg_f = stats['avg_fitness']
        if gen % 10 == 0 or gen == params.generations - 1:
            print(f"  Generation {gen:3d}: avg_fitness = {avg_f:.2f}")

    print("\nEvolution complete.")

    # Compute and display final statistics
    history = engine.history
    counts = compute_strategy_counts(history, params.strategy_length)
    four = extract_four_strategies(counts)
    avg = compute_avg_fitness(history, params.num_agents)

    print("\nFinal Strategy Distribution:")
    last_gen = counts[-1]
    from srac_ipd.parameters import STRATEGY_LABELS
    for idx in [0, 5, 6, 15]:
        label = STRATEGY_LABELS[idx]
        print(f"  {label:8s} ({idx:04b}): {last_gen[idx]:5d} agents")

    print(f"\nFinal Average Fitness: {avg[-1, 0]}")

    # Save results if requested
    if args.output:
        import pickle
        data = {
            'params': params,
            'history': history,
            'strategy_counts': counts,
            'avg_fitness': avg,
        }
        with open(args.output, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nResults saved to {args.output}")

    # Show matplotlib plots if requested
    if args.plot:
        import matplotlib.pyplot as plt
        from srac_ipd.visualization import (create_four_strategy_chart,
                                            create_avg_fitness_chart,
                                            create_lattice_figure)

        fig1 = create_four_strategy_chart(four,
                                          title=f"Four Key Strategies ({params.topology_type})")
        fig2 = create_avg_fitness_chart(avg, title="Average Fitness per Generation")
        fig3 = create_lattice_figure(engine.history[-1], params.strategy_length,
                                     title=f"Final Strategy Distribution (Gen {params.generations-1})")

        # Convert Figure objects to pyplot figures for display
        for i, fig in enumerate([fig1, fig2, fig3], 1):
            new_fig = plt.figure(figsize=fig.get_size_inches())
            new_manager = new_fig.canvas.manager
            new_fig.canvas.figure = fig
            fig.set_canvas(new_fig.canvas)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="SRAC-IPD Simulator - Self-Reputation Awareness in Spatial IPD Game"
    )
    parser.add_argument('--cli', action='store_true',
                        help='Run in command-line mode (no GUI)')

    # Simulation parameters (Java: Parameter defaults)
    parser.add_argument('--name', default='CLI_Sim', help='Simulation name')
    parser.add_argument('--board-size', type=int, default=50, help='Grid size (default: 50)')
    parser.add_argument('--memory-length', type=int, default=1, help='Agent memory capacity')
    parser.add_argument('--ipd-rounds', type=int, default=100, help='IPD rounds per opponent')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations')
    parser.add_argument('--mutation-rate', type=float, default=0.01, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.7, help='Crossover rate')

    # Topology parameters
    parser.add_argument('--topology', choices=['CA', 'SW'], default='CA',
                        help='Network topology (CA or SW)')
    parser.add_argument('--radius', type=int, default=1, help='Neighborhood radius')
    parser.add_argument('--shortcuts', type=int, default=1, help='SW shortcuts per node')

    # Self-awareness parameters
    parser.add_argument('--sa-ratio', type=float, default=0.0,
                        help='Self-aware agent ratio (0.0-1.0)')
    parser.add_argument('--f-low', type=float, default=-1.0, help='Fitness LOW threshold')
    parser.add_argument('--f-high', type=float, default=1.0, help='Fitness HIGH threshold')
    parser.add_argument('--r-low', type=float, default=-1.0, help='Reputation LOW threshold')
    parser.add_argument('--r-high', type=float, default=1.0, help='Reputation HIGH threshold')

    # Output options
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--output', type=str, default=None, help='Output pickle file path')
    parser.add_argument('--plot', action='store_true', help='Show matplotlib plots after run')

    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        run_gui()


if __name__ == '__main__':
    main()
