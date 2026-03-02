"""
SRAC-IPD Simulator (Python 3 Version)
======================================
Self-Reputation Awareness Component in Evolutionary Spatial IPD Game

This package is a faithful Python 3 conversion of the Java-based IPD simulator
originally developed by HikiChen, NCTU CIS Learning Technique Lab (2004-2005).

The simulator implements the model described in:
"Influences of Agents with a Self-Reputation Awareness Component
 in an Evolutionary Spatial IPD Game"
by Chung-Yuan Huang and Chun-Liang Lee (PLoS ONE)

Modules:
    parameters    -- Simulation parameters (Java: Parameter.java, MyParameter.java)
    agent         -- Agent model with strategy chromosome (Java: Agent.java)
    network       -- Network topologies CAN/SWN (Java: CAN.java, SWN.java)
    ipd_game      -- IPD game engine and fitness calculation (Java: EvoThread_CA.Fitness_Calculas)
    evolution     -- Evolutionary algorithms with SRAC (Java: EvoThread_CA*.java, EvoThread_SW*.java)
    statistics    -- Statistics computation (Java: DiagramFrame.java)
    visualization -- matplotlib charts and lattice display (Java: PaintWorld.java, Diagram.java)
    gui           -- Tkinter GUI (Java: SimWorld.java, SimFrame.java, etc.)
"""

__version__ = "1.1.0"
__author__ = "Converted from Java by Claude; Original: HikiChen, NCTU CIS"
