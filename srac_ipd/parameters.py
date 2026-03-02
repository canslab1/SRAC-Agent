"""
parameters.py -- Simulation Parameter Management
=================================================

Corresponds to Java source files:
  - Parameter.java  (static global defaults)
  - MyParameter.java (per-simulation mutable copy)

This module defines all configurable parameters for the SRAC-IPD simulation,
including IPD game settings, network topology settings, evolutionary computation
settings, self-awareness thresholds, and display color mappings.

Conversion notes:
  - Java's static fields in Parameter.java are replaced by module-level constants
    and a default dictionary, because Python has no concept of static-only classes.
  - Java's MyParameter (Serializable) is replaced by a Python dataclass with
    copy support via copy.deepcopy, and pickle for serialization.
  - Color arrays (Java AWT Color) are replaced by matplotlib-compatible hex strings.
"""

import copy
from dataclasses import dataclass

# =============================================================================
# Strategy Labels
# (Java: Parameter.StrategyValue -- 16 memory-1 deterministic strategies)
# With memory_length=1, each strategy is a 4-bit chromosome [Scc, Scd, Sdc, Sdd]
# where 0 = Cooperate, 1 = Defect.
# The index is the decimal value of the binary chromosome.
# =============================================================================
STRATEGY_LABELS = {
    0:  "ALL-C",    # 0000 -- Always Cooperate (Yes-Man, YM)
    1:  "S1",       # 0001
    2:  "S2",       # 0010
    3:  "S3",       # 0011
    4:  "S4",       # 0100
    5:  "TFT",      # 0101 -- Tit-For-Tat
    6:  "PAVLOV",   # 0110 -- Win-Stay, Lose-Shift (WS/LS)
    7:  "S7",       # 0111
    8:  "S8",       # 1000
    9:  "S9",       # 1001
    10: "S10",      # 1010
    11: "S11",      # 1011
    12: "S12",      # 1100
    13: "S13",      # 1101
    14: "S14",      # 1110
    15: "ALL-D",    # 1111 -- Always Defect (Scoundrel, S)
}

# =============================================================================
# Color Palettes
# (Java: Parameter.myColor, myColor1, ..., myColor5)
# Converted from java.awt.Color to matplotlib hex colors.
# =============================================================================

# 16 colors for the 16 possible memory-1 deterministic strategies
# (Java: Parameter.myColor -- indices 0 to 15)
# Converted from Java's new Color(r,g,b) to hex strings.
STRATEGY_COLORS = [
    '#FFFFC0',  #  0: Light Yellow  -- ALL-C   (Java: 255,255,192)
    '#FFFF80',  #  1: Yellow-Light           (Java: 255,255,128)
    '#FFFF40',  #  2: Yellow-Medium          (Java: 255,255,64)
    '#FFFF00',  #  3: Yellow                 (Java: 255,255,0)
    '#FFC0FF',  #  4: Light Pink             (Java: 255,192,255)
    '#00FF00',  #  5: Green         -- TFT   (Java: 0,255,0)
    '#0000FF',  #  6: Blue          -- PAVLOV(Java: 0,0,255)
    '#FF00FF',  #  7: Magenta                (Java: 255,0,255)
    '#C0FFFF',  #  8: Light Cyan             (Java: 192,255,255)
    '#80FFFF',  #  9: Cyan-Light             (Java: 128,255,255)
    '#40FFFF',  # 10: Cyan-Medium            (Java: 64,255,255)
    '#00FFFF',  # 11: Cyan                   (Java: 0,255,255)
    '#C0C0C0',  # 12: Silver                 (Java: 192,192,192)
    '#808080',  # 13: Gray                   (Java: 128,128,128)
    '#404040',  # 14: Dark Gray              (Java: 64,64,64)
    '#000000',  # 15: Black         -- ALL-D (Java: 0,0,0)
]

# Colors for top/bottom 25% fitness chart (Java: Parameter.myColor1)
# Java: myColor1[0]=Red(255,0,0), myColor1[1]=Green(0,255,0)
# Java Statistic2: data[g][0]=bottom 25%, data[g][1]=top 25%
# Java Diagram renders StrategyColor[j] for Data[*][j], so:
#   data[g][0] (bottom 25%) -> myColor1[0] = Red
#   data[g][1] (top 25%)    -> myColor1[1] = Green
# Python data layout is the same: quartile_data[:,0]=bottom 25%, [:,1]=top 25%
# So colors must be [Red, Green] to match Java's mapping.
FITNESS_QUARTILE_COLORS = ['#FF0000', '#00FF00']  # [0]=Red (bottom 25%), [1]=Green (top 25%)

# Color for average fitness chart (Java: Parameter.myColor2)
AVG_FITNESS_COLORS = ['#00FF00']  # Green (Java: 0,255,0)

# Colors for experiment mixing ratios (Java: Parameter.myColor3 -- 8 colors)
EXPERIMENT_COLORS = [
    '#000000', '#FF0000', '#00FF00', '#0000FF',
    '#FFFF00', '#00FFFF', '#FF00FF', '#FFFFFF',
]

# Colors for paper-format 4-strategy chart (Java: Parameter.myColor5)
# ALL-C (Yellow), TFT (Green), PAVLOV (Blue), ALL-D (Black)
PAPER_4_COLORS = ['#FFFF00', '#00FF00', '#0000FF', '#000000']


# =============================================================================
# SimParameters dataclass
# (Java: MyParameter.java -- mutable per-simulation parameter set)
#
# Conversion notes:
#   - Java's MyParameter is Serializable; Python uses pickle-compatible dataclass.
#   - Java copies from static Parameter fields in constructor; Python uses
#     field defaults matching the Java defaults.
#   - Java has separate fields for CA and SW thresholds; Python unifies them
#     with a topology_type flag to select which thresholds to use.
# =============================================================================
@dataclass
class SimParameters:
    """
    Mutable per-simulation parameters.

    Corresponds to Java: MyParameter.java
    Each simulation instance gets its own copy of parameters, which can be
    modified without affecting other simulations or the global defaults.
    """

    # --- Simulation Identity ---
    # (Java: MyParameter.SimName)
    sim_name: str = "Sim1"

    # --- IPD Game Parameters ---
    # (Java: MyParameter.BoardSize, NumOfAgent, MemoryLength, StrLength, Times, Generations)
    board_size: int = 50           # Grid dimension W = H (Java: BoardSize)
    num_agents: int = 2500         # Total agents = board_size^2 (Java: NumOfAgent)
    memory_length: int = 1         # Agent memory capacity c (Java: MemoryLength)
    strategy_length: int = 4       # Chromosome length = 2^(2*c) (Java: StrLength)
    ipd_rounds: int = 100          # IPD rounds per opponent per generation q (Java: Times)
    generations: int = 100         # Total generations MAX_G (Java: Generations)

    # --- Evolutionary Computation Parameters ---
    # (Java: MyParameter.Mutation_Rate, Crossover_Rate)
    mutation_rate: float = 0.01    # Pm -- probability of flipping each chromosome bit
    crossover_rate: float = 0.7    # Pc -- probability of applying crossover

    # --- Network Topology Parameters ---
    # (Java: MyParameter.TopologyType, Radius, Shortcuts)
    topology_type: str = ""        # "Cellular Automata" or "Small-World"
    radius: int = 1                # k-neighborhood radius (default Moore neighborhood)
    shortcuts: int = 0             # Number of small-world shortcuts per node

    # --- Self-Awareness Parameters ---
    # (Java: MyParameter.Selfaware, SelfawareRatio, F_LowRatio, F_HighRatio, etc.)
    selfaware: bool = False        # Whether SRAC agents are enabled
    selfaware_ratio: float = 0.0   # Fraction of agents that are self-aware (0.0 to 1.0)

    # CA-specific z-score thresholds for fitness and reputation classification
    # (Java: Parameter.F_LowRatio, F_HighRatio, R_LowRatio, R_HighRatio)
    f_low_ratio: float = -1.0      # Fitness z-score below which fitness is "LOW"
    f_high_ratio: float = 1.0      # Fitness z-score above which fitness is "HIGH"
    r_low_ratio: float = -1.0      # Reputation z-score below which reputation is "LOW"
    r_high_ratio: float = 1.0      # Reputation z-score above which reputation is "HIGH"

    # SW-specific z-score thresholds (may differ from CA thresholds)
    # (Java: Parameter.SW_F_LowRatio, SW_F_HighRatio, SW_R_LowRatio, SW_R_HighRatio)
    sw_f_low_ratio: float = -1.0
    sw_f_high_ratio: float = 1.0
    sw_r_low_ratio: float = -1.0
    sw_r_high_ratio: float = 1.0

    # --- Batch Experiment Parameters ---
    # (Java: Parameter.Runs)
    runs: int = 1                  # Number of experiment repetitions

    def copy(self) -> 'SimParameters':
        """
        Create a deep copy of this parameter set.
        (Java: MyParameter was Serializable and copied field-by-field)
        """
        return copy.deepcopy(self)

    def set_topology_ca(self, radius: int):
        """
        Configure for Cellular Automata topology.
        (Java: MyParameter.setTopologyParameter(String, int))
        """
        self.topology_type = "Cellular Automata"
        self.radius = radius
        self.shortcuts = 0

    def set_topology_sw(self, radius: int, shortcuts: int):
        """
        Configure for Small-World Network topology.
        (Java: MyParameter.setTopologyParameter(String, int, int))
        """
        self.topology_type = "Small-World"
        self.radius = radius
        self.shortcuts = shortcuts

    def set_selfaware_params(self, enabled: bool, ratio: float,
                             f_low: float, f_high: float,
                             r_low: float, r_high: float):
        """
        Set all self-awareness parameters at once.
        (Java: MyParameter.setSelfawareParameter)
        """
        self.selfaware = enabled
        self.selfaware_ratio = ratio
        self.f_low_ratio = f_low
        self.f_high_ratio = f_high
        self.r_low_ratio = r_low
        self.r_high_ratio = r_high

    def get_active_thresholds(self):
        """
        Return the fitness/reputation thresholds appropriate for the current topology.
        For Small-World, uses SW-specific thresholds; for CA, uses standard thresholds.

        Returns:
            tuple: (f_low, f_high, r_low, r_high)
        """
        if self.topology_type == "Small-World":
            return (self.sw_f_low_ratio, self.sw_f_high_ratio,
                    self.sw_r_low_ratio, self.sw_r_high_ratio)
        else:
            return (self.f_low_ratio, self.f_high_ratio,
                    self.r_low_ratio, self.r_high_ratio)

    def update_derived(self):
        """
        Recompute derived parameters after changing board_size or memory_length.
        (Java: these were computed in constructors)
        """
        self.num_agents = self.board_size * self.board_size
        self.strategy_length = int(2 ** (2 * self.memory_length))
