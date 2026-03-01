"""
gui.py -- Tkinter Graphical User Interface (Integrated Single-Window Design)
=============================================================================

Corresponds to Java source files:
  - UI/SimWorld.java           (Main MDI window -- JFrame)
  - UI/SimFrame.java           (Simulation child window -- JInternalFrame)
  - UI/MyMenuBar.java          (Application menu bar)
  - UI/Setting.java            (Parameter setting dialog)
  - UI/MyStatusBar.java        (Status bar with progress)
  - SimModel/EvoStatus.java    (Evolution parameter display)
  - SimModel/WorldStatus.java  (World parameter display)
  - Evolution/SelfAware_ParaSetting.java (Self-awareness parameter dialog)
  - EventHandler.java          (Button event handling)
  - MyMenuListener.java        (Menu action dispatcher)

Conversion notes:
  - The Java version uses Swing MDI (JDesktopPane + JInternalFrame) with separate
    windows for settings, simulation, and analysis. This Python version consolidates
    everything into a single main window for usability:
      Left panel  : lattice visualization (Java: PaintWorld)
      Right panel : parameters + evolution controls + analysis buttons
      Bottom      : status bar with progress (Java: MyStatusBar)
  - Java's JFrame       -> tkinter.Tk (main window)
  - Java's JMenuBar     -> tkinter.Menu
  - Java's JOptionPane  -> tkinter.simpledialog
  - Java's Thread       -> threading.Thread + tkinter.after() polling for GUI updates
  - matplotlib figures are embedded in tkinter via FigureCanvasTkAgg.
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import threading
import pickle
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from .parameters import SimParameters, STRATEGY_COLORS, STRATEGY_LABELS
from .agent import Agent, create_agent_grid, copy_agent_grid
from .network import create_can, create_swn, get_neighbors, get_network_info
from .evolution import EvolutionEngine
from .statistics import (compute_strategy_counts, compute_fitness_quartiles,
                         compute_avg_fitness, extract_four_strategies)
from .visualization import (create_lattice_figure, create_strategy_dynamics_chart,
                            create_four_strategy_chart, create_avg_fitness_chart,
                            create_fitness_quartile_chart)


class SimulationApp(tk.Tk):
    """
    Main application window -- all-in-one integrated interface.

    Corresponds to Java: UI/SimWorld.java + UI/SimFrame.java combined.

    Layout:
    +--[Menu Bar]------------------------------------------------------+
    |                         |                                         |
    |   Lattice Visualization |  [1] Simulation Parameters              |
    |   (PaintWorld)          |  [2] Network Topology Settings          |
    |                         |  [3] Self-Awareness (SRAC) Settings     |
    |                         |  [4] Evolution Control Buttons          |
    |                         |  [5] Progress Bar                       |
    |                         |  [6] Analysis Buttons                   |
    |                         |                                         |
    +-------------------------+-----------------------------------------+
    |  [Status Bar with generation info]                                |
    +-----------------------------------------------------------------+

    This consolidates the Java version's separate SimWorld (main frame),
    SimFrame (simulation internal frame), Setting (parameter dialog),
    and SelfAware_ParaSetting (SRAC dialog) into a single usable window.
    """

    def __init__(self):
        super().__init__()
        self.title("SRAC-IPD Simulator (Python 3)")
        # Java SimWorld uses 1024x768; we use similar dimensions
        self.geometry("1100x750")
        self.minsize(900, 600)

        # ---- State ----
        self.params = SimParameters()
        self.agent_grid = None      # Created when user initializes
        self.evo_engine = None
        self.evo_thread = None
        self.evo_running = False
        self.history = []
        self._latest_stats = None
        self._evo_error = None      # Stores traceback if background thread crashes

        # Slider fast-path state (avoids full matplotlib redraw per tick)
        self._lattice_mesh = None          # pcolormesh QuadMesh handle
        self._gen_strategy_cache = None    # list[np.ndarray|None] lazy cache
        self._slider_programmatic = False  # suppress callback during programmatic set

        # ---- Build UI ----
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

        # ---- Initialise first simulation world ----
        self._initialize_world()

    # =====================================================================
    #  Menu Bar  (Java: UI/MyMenuBar.java)
    # =====================================================================
    def _create_menu(self):
        """
        Create the application menu bar.

        Corresponds to Java: UI/MyMenuBar.java
        Menu items are also duplicated as visible buttons in the right panel,
        so the menu serves as a secondary access path.
        """
        menubar = tk.Menu(self)

        # --- File Menu ---
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New World (Reset)", command=self._initialize_world)
        file_menu.add_command(label="Load Data...", command=self._load_data)
        file_menu.add_command(label="Save Data...", command=self._save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        # --- Experiment Menu ---
        exp_menu = tk.Menu(menubar, tearoff=0)
        exp_menu.add_command(label="Batch CA Experiment",
                             command=lambda: self._batch_experiment("CA"))
        exp_menu.add_command(label="Batch SW Experiment",
                             command=lambda: self._batch_experiment("SW"))
        menubar.add_cascade(label="Batch Experiment", menu=exp_menu)

        # --- About ---
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="About", menu=about_menu)

        self.config(menu=menubar)

    # =====================================================================
    #  Main Layout
    # =====================================================================
    def _create_main_layout(self):
        """Build the two-panel main layout: lattice (left) + controls (right)."""

        # Outer container that fills the whole window
        self.main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=6)
        self.main_pane.pack(fill='both', expand=True)

        # --- Left Panel: Lattice Visualization ---
        # (Java: PaintWorld pw -- custom JPanel with paintComponent)
        left_frame = tk.Frame(self.main_pane, bg='#2b2b2b')
        self.main_pane.add(left_frame, width=650, stretch='always')

        self.lattice_fig = Figure(figsize=(6, 6), dpi=100, facecolor='#2b2b2b')
        self.lattice_ax = self.lattice_fig.add_subplot(111)
        self.lattice_ax.set_facecolor('#1e1e1e')
        self.lattice_canvas = FigureCanvasTkAgg(self.lattice_fig, master=left_frame)
        self.lattice_canvas.get_tk_widget().pack(fill='both', expand=True)

        # --- Right Panel: All Controls ---
        right_frame = tk.Frame(self.main_pane)
        self.main_pane.add(right_frame, width=400, stretch='never')

        # Make right panel scrollable for smaller screens
        canvas_scroll = tk.Canvas(right_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(right_frame, orient='vertical', command=canvas_scroll.yview)
        self.scroll_frame = tk.Frame(canvas_scroll)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        )
        canvas_scroll.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side='right', fill='y')
        canvas_scroll.pack(side='left', fill='both', expand=True)

        # Populate right panel with control groups
        self._create_param_panel(self.scroll_frame)
        self._create_topology_panel(self.scroll_frame)
        self._create_srac_panel(self.scroll_frame)
        self._create_evolution_panel(self.scroll_frame)
        self._create_progress_panel(self.scroll_frame)
        self._create_analysis_panel(self.scroll_frame)

    # =====================================================================
    #  [1] Simulation Parameters  (Java: UI/Setting.java)
    # =====================================================================
    def _create_param_panel(self, parent):
        """
        Create the simulation parameter input panel.

        Corresponds to Java: UI/Setting.java -- JTextField[] for each parameter.
        Instead of a separate dialog window, parameters are always visible and
        editable in the main window's right panel.
        """
        frame = ttk.LabelFrame(parent, text=" 1. Simulation Parameters ")
        frame.pack(fill='x', padx=8, pady=4)

        fields = [
            ("Board Size (W=H)", "board_size", "50"),
            ("Memory Length (c)", "memory_length", "1"),
            ("IPD Rounds (q)", "ipd_rounds", "100"),
            ("Generations (MAX_G)", "generations", "100"),
            ("Mutation Rate (Pm)", "mutation_rate", "0.01"),
            ("Crossover Rate (Pc)", "crossover_rate", "0.7"),
        ]

        self.param_entries = {}
        for row_idx, (label, key, default) in enumerate(fields):
            tk.Label(frame, text=label, anchor='w').grid(
                row=row_idx, column=0, padx=5, pady=1, sticky='w')
            entry = tk.Entry(frame, width=12)
            entry.insert(0, default)
            entry.grid(row=row_idx, column=1, padx=5, pady=1, sticky='e')
            self.param_entries[key] = entry

        # Computed info label
        self.agents_label = tk.Label(frame, text="Agents: 2500", fg='grey')
        self.agents_label.grid(row=len(fields), column=0, columnspan=2, pady=2)

        # Bind board_size change to update agents count display
        self.param_entries['board_size'].bind(
            '<KeyRelease>', lambda e: self._update_agents_label())

    def _update_agents_label(self):
        try:
            bs = int(self.param_entries['board_size'].get())
            self.agents_label.config(text=f"Agents: {bs*bs}")
        except ValueError:
            pass

    # =====================================================================
    #  [2] Network Topology  (Java: CAN.java, SWN.java + JOptionPane prompts)
    # =====================================================================
    def _create_topology_panel(self, parent):
        """
        Network topology configuration panel.

        Corresponds to Java: JOptionPane prompts in MyMenuListener for
        k-neighborhood, network type, and shortcuts.
        """
        frame = ttk.LabelFrame(parent, text=" 2. Network Topology ")
        frame.pack(fill='x', padx=8, pady=4)

        # Topology type radio buttons
        self.topology_var = tk.StringVar(value="CA")
        tk.Label(frame, text="Type:", anchor='w').grid(row=0, column=0, padx=5, sticky='w')
        rb_frame = tk.Frame(frame)
        rb_frame.grid(row=0, column=1, sticky='w')
        tk.Radiobutton(rb_frame, text="Cellular Automata",
                       variable=self.topology_var, value="CA",
                       command=self._on_topology_change).pack(side='left')
        tk.Radiobutton(rb_frame, text="Small-World",
                       variable=self.topology_var, value="SW",
                       command=self._on_topology_change).pack(side='left')

        # k-neighborhood radius
        tk.Label(frame, text="Radius (k):", anchor='w').grid(
            row=1, column=0, padx=5, sticky='w')
        self.radius_entry = tk.Entry(frame, width=12)
        self.radius_entry.insert(0, "1")
        self.radius_entry.grid(row=1, column=1, padx=5, sticky='e')

        # Shortcuts (Small-World only)
        tk.Label(frame, text="Shortcuts:", anchor='w').grid(
            row=2, column=0, padx=5, sticky='w')
        self.shortcuts_entry = tk.Entry(frame, width=12)
        self.shortcuts_entry.insert(0, "1")
        self.shortcuts_entry.grid(row=2, column=1, padx=5, sticky='e')
        self.shortcuts_label_widget = frame.grid_slaves(row=2, column=0)[0]
        self.shortcuts_entry.config(state='disabled')

    def _on_topology_change(self):
        if self.topology_var.get() == "SW":
            self.shortcuts_entry.config(state='normal')
        else:
            self.shortcuts_entry.config(state='disabled')

    # =====================================================================
    #  [3] Self-Awareness (SRAC)  (Java: SelfAware_ParaSetting.java)
    # =====================================================================
    def _create_srac_panel(self, parent):
        """
        Self-Reputation Awareness Component configuration.

        Corresponds to Java: SelfAware_ParaSetting.java dialog and the
        JOptionPane prompts in EvoThread_CA_Mix constructor for ratio
        and four threshold values.
        """
        frame = ttk.LabelFrame(parent, text=" 3. Self-Awareness (SRAC) ")
        frame.pack(fill='x', padx=8, pady=4)

        # Enable toggle
        self.srac_enabled = tk.BooleanVar(value=False)
        tk.Checkbutton(frame, text="Enable SRAC Agents",
                       variable=self.srac_enabled,
                       command=self._on_srac_toggle).grid(
                           row=0, column=0, columnspan=2, padx=5, sticky='w')

        # SRAC ratio (Java: SelfAware_ParaSetting.paraValue[4] = 0.1)
        # IMPORTANT: Entry must be created in 'normal' state FIRST, then insert()
        # the default text, THEN set to 'disabled'. In tkinter, insert() on a
        # disabled Entry is silently ignored -- the widget would appear empty.
        tk.Label(frame, text="SRAC Ratio (0.0~1.0):", anchor='w').grid(
            row=1, column=0, padx=5, sticky='w')
        self.srac_ratio_entry = tk.Entry(frame, width=12)
        self.srac_ratio_entry.insert(0, "0.1")
        self.srac_ratio_entry.config(state='disabled')
        self.srac_ratio_entry.grid(row=1, column=1, padx=5, sticky='e')

        # Thresholds (F_Low, F_High, R_Low, R_High)
        # (Java: Parameter.F_LowRatio=-1, F_HighRatio=1, R_LowRatio=-1, R_HighRatio=1)
        # (Java: EvoThread_CA_Mix JOptionPane default: "-1.0,1.0,-1.0,1.0")
        # These are z-score thresholds used in getLowOrHigh() to classify an
        # agent's fitness/reputation relative to its neighbors:
        #   z < F_Low  -> LOW fitness   (agent performs poorly)
        #   z > F_High -> HIGH fitness  (agent performs well)
        #   otherwise  -> MIDDLE        (average performance)
        thresh_labels = [
            ("F_Low Ratio:", "f_low", "-1.0"),
            ("F_High Ratio:", "f_high", "1.0"),
            ("R_Low Ratio:", "r_low", "-1.0"),
            ("R_High Ratio:", "r_high", "1.0"),
        ]
        self.thresh_entries = {}
        for idx, (label, key, default) in enumerate(thresh_labels, start=2):
            tk.Label(frame, text=label, anchor='w').grid(
                row=idx, column=0, padx=5, sticky='w')
            # Create Entry in 'normal' state first, insert default, then disable
            e = tk.Entry(frame, width=12)
            e.insert(0, default)
            e.config(state='disabled')
            e.grid(row=idx, column=1, padx=5, sticky='e')
            self.thresh_entries[key] = e

    def _on_srac_toggle(self):
        state = 'normal' if self.srac_enabled.get() else 'disabled'
        self.srac_ratio_entry.config(state=state)
        for e in self.thresh_entries.values():
            e.config(state=state)

    # =====================================================================
    #  [4] Evolution Controls  (Java: EventHandler / MyMenuListener buttons)
    # =====================================================================
    def _create_evolution_panel(self, parent):
        """
        Evolution start/reset buttons.

        Corresponds to Java: JButton[] functionButton in SimFrame and the
        menu items "EvoThread_CA", "EvoThread_CA_Mix", "EvoThread_SW",
        "EvoThread_SW_Mix" in MyMenuListener.
        """
        frame = ttk.LabelFrame(parent, text=" 4. Evolution Control ")
        frame.pack(fill='x', padx=8, pady=4)

        # Big prominent "Start Evolution" button
        self.start_btn = tk.Button(
            frame, text="▶  Start Evolution", font=('Helvetica', 13, 'bold'),
            bg='#4CAF50', fg='white', activebackground='#388E3C',
            height=2, command=self._start_evolution)
        self.start_btn.pack(fill='x', padx=8, pady=6)

        # Secondary button
        self.reset_btn = tk.Button(frame, text="Reset World",
                                    command=self._initialize_world)
        self.reset_btn.pack(fill='x', padx=8, pady=2)

    # =====================================================================
    #  [5] Progress Display  (Java: UI/MyStatusBar.java + JProgressBar)
    # =====================================================================
    def _create_progress_panel(self, parent):
        """Progress bar and generation info."""
        frame = ttk.LabelFrame(parent, text=" 5. Progress ")
        frame.pack(fill='x', padx=8, pady=4)

        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var,
                                             maximum=100)
        self.progress_bar.pack(fill='x', padx=8, pady=4)

        self.progress_label = tk.Label(frame, text="Not started",
                                        font=('Helvetica', 10))
        self.progress_label.pack(pady=2)

        # --- Generation Slider ---
        # (Java: SimFrame.setStatusPanel1() creates JLabel + JScrollBar)
        # Allows scrubbing through each generation's lattice after evolution
        # completes or after loading saved data.
        self.gen_slider_label = tk.Label(frame, text="Generation => 0",
                                          font=('Helvetica', 10))
        self.gen_slider_label.pack(pady=(8, 2))

        self.gen_slider = tk.Scale(frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                    command=self._on_gen_slider_change,
                                    state='disabled', showvalue=False)
        self.gen_slider.pack(fill='x', padx=8, pady=4)

    # =====================================================================
    #  [6] Analysis Buttons  (Java: EventHandler / DiagramFrame buttons)
    # =====================================================================
    def _create_analysis_panel(self, parent):
        """
        Analysis buttons to display various charts.

        Corresponds to Java: EventHandler buttons in SimFrame that trigger
        DiagramFrame creation (Statistic1 .. Statistic8).
        """
        frame = ttk.LabelFrame(parent, text=" 6. Analysis (after evolution) ")
        frame.pack(fill='x', padx=8, pady=4)

        buttons = [
            ("📊 Strategy Dynamics (16 strategies)", self._show_strategy_dynamics),
            ("📈 4 Key Strategies (ALL-C, TFT, PAVLOV, ALL-D)", self._show_four_strategies),
            ("📉 Average Fitness per Generation", self._show_avg_fitness),
            ("📋 Fitness Quartiles (Top/Bottom 25%)", self._show_fitness_quartiles),
            ("💾 Save Data...", self._save_data),
            ("📂 Load Data...", self._load_data),
        ]
        self.save_btn = None
        self.load_btn = None
        for text, cmd in buttons:
            btn = tk.Button(frame, text=text, anchor='w', command=cmd)
            btn.pack(fill='x', padx=8, pady=2)
            if cmd == self._save_data:
                self.save_btn = btn
            elif cmd == self._load_data:
                self.load_btn = btn

    # =====================================================================
    #  Status Bar  (Java: UI/MyStatusBar.java)
    # =====================================================================
    def _create_status_bar(self):
        """Bottom status bar."""
        self.status_bar = tk.Label(
            self, text="  Ready — configure parameters on the right, then click Start Evolution.",
            anchor='w', relief='sunken', bg='#f0f0f0', font=('Helvetica', 10))
        self.status_bar.pack(side='bottom', fill='x')

    # =====================================================================
    #  World Initialization  (Java: SimFrame constructor + Agent[][] creation)
    # =====================================================================
    def _initialize_world(self):
        """
        Read current parameters and create a fresh agent grid.

        Corresponds to Java: Setting "Yes" handler -> new SimFrame(...)
            Agent[][] myAgent = new Agent[BoardSize][BoardSize];
            for each (i,j): myAgent[i][j] = new Agent(MemoryLength, i, j);
        """
        if self.evo_running:
            messagebox.showwarning("Running", "Cannot reset while evolution is running.")
            return

        # Parse parameters from entry fields
        try:
            p = self.params
            p.board_size = int(self.param_entries['board_size'].get())
            p.memory_length = int(self.param_entries['memory_length'].get())
            p.ipd_rounds = int(self.param_entries['ipd_rounds'].get())
            p.generations = int(self.param_entries['generations'].get())
            p.mutation_rate = float(self.param_entries['mutation_rate'].get())
            p.crossover_rate = float(self.param_entries['crossover_rate'].get())
            p.update_derived()
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter value: {e}")
            return

        # Create fresh agent grid
        # (Java: Agent[][] myAgent = new Agent[BoardSize][BoardSize]; ...)
        self.agent_grid = create_agent_grid(p.board_size, p.memory_length)
        self.history = []
        self.evo_engine = None
        self._latest_stats = None

        # Update progress bar maximum
        self.progress_var.set(0)
        self.progress_bar.config(maximum=p.generations)
        self.progress_label.config(text="Not started")

        # Reset generation slider (history is empty after re-init)
        self._slider_programmatic = True
        self.gen_slider.config(from_=0, to=0, state='disabled')
        self._slider_programmatic = False
        self.gen_slider_label.config(text="Generation => 0")
        self._gen_strategy_cache = None     # Invalidate slider fast-path cache

        # Draw initial lattice
        self._update_lattice(title="Initial Random Strategies")

        # Re-enable all controls (start, reset, save, load)
        self._enable_controls_after_evolution()

        self.status_bar.config(
            text=f"  World created: {p.board_size}×{p.board_size} = {p.num_agents} agents  |  "
                 f"Ready to start evolution.")

    # ----- Button state management during evolution -----
    def _disable_controls_for_evolution(self):
        """Disable buttons that should not be used while evolution is running."""
        self.start_btn.config(state='disabled', text="Running...", bg='grey')
        self.reset_btn.config(state='disabled')
        if self.save_btn:
            self.save_btn.config(state='disabled')
        if self.load_btn:
            self.load_btn.config(state='disabled')

    def _enable_controls_after_evolution(self):
        """Re-enable buttons after evolution finishes or errors."""
        self.start_btn.config(state='normal', text="▶  Start Evolution",
                              bg='#4CAF50')
        self.reset_btn.config(state='normal')
        if self.save_btn:
            self.save_btn.config(state='normal')
        if self.load_btn:
            self.load_btn.config(state='normal')

    # =====================================================================
    #  Lattice Visualization Update  (Java: PaintWorld.paintComponent)
    # =====================================================================
    def _update_lattice(self, title: str = ""):
        """
        Redraw the 2D lattice showing agent strategy colours.

        Corresponds to Java: PaintWorld.paintComponent(Graphics g)
            for each cell (i,j):
                sIdx = binary-to-decimal(chromosome)
                g.setColor(StrategyColor[sIdx])
                g.fillRect(...)
        """
        if self.agent_grid is None:
            return

        rows, cols = self.agent_grid.shape
        num_strategies = 2 ** self.params.strategy_length

        # Build strategy index matrix (flat array + reshape for speed)
        flat = np.empty(rows * cols, dtype=int)
        k = 0
        for i in range(rows):
            for j in range(cols):
                flat[k] = self.agent_grid[i, j].get_strategy_index()
                k += 1
        strategy_indices = flat.reshape(rows, cols)

        cmap = ListedColormap(STRATEGY_COLORS[:num_strategies])

        ax = self.lattice_ax
        ax.clear()

        # Use pcolormesh with edgecolors so that each cell is drawn as a
        # separate quad with its own fill colour AND border.  This avoids
        # the sub-pixel misalignment that imshow + grid-overlay causes
        # (some cells showing a sliver of a neighbour's colour).
        # (Java: PaintWorld draws black grid lines first, then fills each
        #  cell at +1 pixel offset with BoardPieceSize-1, leaving 1-pixel
        #  black borders around every cell.)
        self._lattice_mesh = ax.pcolormesh(
            strategy_indices, cmap=cmap, vmin=0,
            vmax=num_strategies - 1,
            edgecolors='black', linewidth=0.5)
        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)          # flip y so row-0 is at top
        ax.set_aspect('equal')

        if title:
            ax.set_title(title, fontsize=11, color='white')
        ax.tick_params(which='major', colors='grey', labelsize=7)

        # Mini legend for 4 key strategies
        from matplotlib.patches import Patch
        legend_items = [
            Patch(facecolor=STRATEGY_COLORS[0], edgecolor='grey', label='ALL-C(0000)'),
            Patch(facecolor=STRATEGY_COLORS[5], edgecolor='grey', label='TFT(0101)'),
            Patch(facecolor=STRATEGY_COLORS[6], edgecolor='grey', label='PAVLOV(0110)'),
            Patch(facecolor=STRATEGY_COLORS[15], edgecolor='grey', label='ALL-D(1111)'),
        ]
        ax.legend(handles=legend_items, loc='lower right', fontsize=7,
                  framealpha=0.8, facecolor='#EEEEEE', labelcolor='black')

        self.lattice_fig.tight_layout(pad=1.0)
        self.lattice_canvas.draw_idle()

    # =====================================================================
    #  Generation Slider — Fast Update Path
    # =====================================================================
    #
    #  Java's PaintWorld.paintComponent uses hardware-accelerated fillRect
    #  calls (~0 ms per frame), but matplotlib's full redraw cycle takes
    #  ~70 ms per frame (tight_layout 28 ms + canvas.draw 33 ms + clear/
    #  pcolormesh/legend 8 ms + build indices 2 ms).
    #
    #  To make the slider feel responsive we use a **fast path** that
    #  skips tight_layout / clear / pcolormesh recreation / legend and
    #  instead calls QuadMesh.set_array() to update colours in-place.
    #  This brings per-frame cost down to ~35-40 ms (~25 FPS), and
    #  draw_idle() further coalesces rapid slider events so that only
    #  the most recent frame actually renders.
    # =====================================================================

    def _get_gen_strategy_flat(self, gen: int) -> np.ndarray:
        """Return a pre-flattened strategy-index array for *gen*, with lazy caching.

        On first access for a given generation the 50×50 agent grid is
        scanned (~2 ms); subsequent accesses return the cached result
        instantly.  The cache is invalidated when a new evolution starts
        or the world is re-initialised.
        """
        if self._gen_strategy_cache is None:
            self._gen_strategy_cache = [None] * len(self.history)
        # Grow cache if history grew (shouldn't happen for slider, but safe)
        while len(self._gen_strategy_cache) < len(self.history):
            self._gen_strategy_cache.append(None)
        if self._gen_strategy_cache[gen] is None:
            gen_grid = self.history[gen]
            rows, cols = gen_grid.shape
            flat = np.empty(rows * cols, dtype=int)
            k = 0
            for i in range(rows):
                for j in range(cols):
                    flat[k] = gen_grid[i, j].get_strategy_index()
                    k += 1
            self._gen_strategy_cache[gen] = flat
        return self._gen_strategy_cache[gen]

    def _update_lattice_fast(self, gen: int):
        """Fast lattice colour update — used exclusively by the slider.

        Instead of the full ``_update_lattice`` cycle (clear → pcolormesh →
        legend → tight_layout → draw), this method:
        1. Looks up the cached flat strategy-index array  (~0 ms)
        2. Calls ``QuadMesh.set_array()`` to swap colours   (~0.1 ms)
        3. Updates the title text                            (~0.2 ms)
        4. Calls ``draw_idle()``                             (~35 ms, coalesced)

        Falls back to the full path if the mesh handle is not available
        (e.g. the first draw has not happened yet).
        """
        if self._lattice_mesh is None:
            # No mesh yet — do a full draw (sets self._lattice_mesh)
            self.agent_grid = self.history[gen]
            self._update_lattice(title=f"Generation {gen}")
            return

        flat = self._get_gen_strategy_flat(gen)
        self._lattice_mesh.set_array(flat)
        self.lattice_ax.set_title(f"Generation {gen}",
                                   fontsize=11, color='white')
        self.lattice_canvas.draw_idle()

    # =====================================================================
    #  Generation Slider Callback  (Java: SimFrame.adjustmentValueChanged)
    # =====================================================================
    def _on_gen_slider_change(self, value):
        """
        Called when the user drags the generation slider.

        Corresponds to Java: SimFrame.adjustmentValueChanged(AdjustmentEvent e)
            gen.setText("Generation => " + e.getValue());
            pw.setAgent(data.agent[e.getValue()]);
            pw.paintImmediately(...);

        Uses the fast update path (set_array) instead of a full redraw
        to keep slider scrubbing smooth.
        """
        # Guard: ignore programmatic changes (e.g. _enable_gen_slider, reset)
        if self._slider_programmatic:
            return
        # tk.Scale command passes value as a string; some platforms use
        # float format ("49.0"), so int(float(...)) is safer than int().
        gen = int(float(value))
        if not self.history or gen >= len(self.history):
            return
        self.gen_slider_label.config(text=f"Generation => {gen}")
        self.agent_grid = self.history[gen]
        self._update_lattice_fast(gen)

    def _enable_gen_slider(self):
        """Enable the generation slider and set its range to match history.

        Uses ``_slider_programmatic`` flag to suppress the Scale command
        callback so that callers retain control over what title is shown
        and avoid a redundant ``_update_lattice`` redraw.  Also initialises
        the lazy strategy-index cache.
        """
        if self.history:
            max_gen = len(self.history) - 1
            # Reset lazy cache (will be populated on demand during scrubbing)
            self._gen_strategy_cache = [None] * len(self.history)
            self._slider_programmatic = True
            self.gen_slider.config(from_=0, to=max_gen, state='normal')
            self.gen_slider.set(max_gen)
            self._slider_programmatic = False
            self.gen_slider_label.config(text=f"Generation => {max_gen}")

    # =====================================================================
    #  Start Evolution  (Java: EvoThread_CA / EvoThread_CA_Mix .run())
    # =====================================================================
    def _start_evolution(self):
        """
        Parse all settings and launch the evolutionary simulation.

        Corresponds to Java: MyMenuListener creating one of
        EvoThread_CA / EvoThread_CA_Mix / EvoThread_SW / EvoThread_SW_Mix
        and calling evo.start().

        Detailed flow (matching Java):
            1. Read and validate all parameters on the main thread
            2. Create fresh agent grid on the main thread
            3. Launch background thread which:
               a. Creates CAN or SWN network  (Java: new CAN(...) / new SWN(...))
               b. Creates EvolutionEngine     (Java: new EvoThread_CA_Mix(...))
               c. Runs evolution loop          (Java: evo.start())
            4. Begin GUI polling              (Java: frame.pw.paintImmediately inside loop)

        Network creation and engine initialization are done in the background
        thread to keep the GUI responsive during setup (which can take a few
        seconds for large grids).
        """
        if self.evo_running:
            messagebox.showwarning("Running", "Evolution is already in progress.")
            return

        # Ensure world exists
        if self.agent_grid is None:
            self._initialize_world()

        params = self.params

        # --- Re-read simulation parameters from Entry widgets ---
        # This ensures that if the user edited entries (e.g. board_size,
        # generations) after the last Reset World, the new values are used
        # instead of stale ones.
        try:
            params.board_size = int(self.param_entries['board_size'].get())
            params.memory_length = int(self.param_entries['memory_length'].get())
            params.ipd_rounds = int(self.param_entries['ipd_rounds'].get())
            params.generations = int(self.param_entries['generations'].get())
            params.mutation_rate = float(self.param_entries['mutation_rate'].get())
            params.crossover_rate = float(self.param_entries['crossover_rate'].get())
            params.update_derived()
        except ValueError as e:
            messagebox.showerror("Parameter Error",
                                 f"Invalid parameter value: {e}")
            return

        # --- Read topology settings (validation on main thread) ---
        topology = self.topology_var.get()
        try:
            radius = int(self.radius_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid radius value."); return

        shortcuts = 0
        if topology == "SW":
            try:
                shortcuts = int(self.shortcuts_entry.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid shortcuts value."); return

        # --- Read SRAC settings (validation on main thread) ---
        if self.srac_enabled.get():
            try:
                sa_ratio = float(self.srac_ratio_entry.get())
                f_low = float(self.thresh_entries['f_low'].get())
                f_high = float(self.thresh_entries['f_high'].get())
                r_low = float(self.thresh_entries['r_low'].get())
                r_high = float(self.thresh_entries['r_high'].get())
            except ValueError:
                messagebox.showerror("Error", "Invalid SRAC parameter values."); return
            params.set_selfaware_params(True, sa_ratio, f_low, f_high, r_low, r_high)
        else:
            params.set_selfaware_params(False, 0.0, -1.0, 1.0, -1.0, 1.0)

        # --- Store topology params for background thread ---
        self._topo_type = topology
        self._topo_radius = radius
        self._topo_shortcuts = shortcuts

        # --- Re-initialise agent grid (fresh random chromosomes) ---
        self.agent_grid = create_agent_grid(params.board_size, params.memory_length)
        self.history = []
        self._latest_stats = None
        self._evo_error = None          # Clear any previous error
        self.progress_var.set(0)
        self.progress_bar.config(maximum=params.generations)
        self.evo_running = True

        # Reset & disable generation slider during evolution
        # Suppress callback to avoid a spurious _on_gen_slider_change when
        # the range narrows and tk.Scale clamps the current value to 0.
        self._slider_programmatic = True
        self.gen_slider.config(from_=0, to=0, state='disabled')
        self._slider_programmatic = False
        self.gen_slider_label.config(text="Generation => 0")
        self._gen_strategy_cache = None     # Invalidate slider fast-path cache

        # Disable controls that should not be used during evolution
        self._disable_controls_for_evolution()

        # Build descriptive status text
        desc = f"{topology} | radius={radius}"
        if topology == "SW":
            desc += f" | shortcuts={shortcuts}"
        if self.srac_enabled.get():
            desc += f" | SRAC={sa_ratio:.0%}"
        self.status_bar.config(text=f"  Initializing: {desc}")
        self.progress_label.config(text="Creating network & computing generation 0...")
        self.update_idletasks()

        # --- Launch background thread ---
        # Network creation, engine init, and evolution all run in background
        # to keep the GUI fully responsive.
        # (Java: EvoThread extends Thread; evo.start();)
        self.evo_thread = threading.Thread(target=self._run_evolution, daemon=True)
        self.evo_thread.start()

        # Begin polling for GUI updates (250ms interval, consistent with
        # subsequent polls in _poll_evolution)
        self.after(250, self._poll_evolution)

    def _run_evolution(self):
        """
        Execute the evolution loop in a background thread.

        Corresponds to Java: EvoThread_CA_Mix.run() -- the Thread's run() body.

        This method now also handles network creation and engine initialization
        (moved from main thread to keep GUI responsive) and catches all exceptions
        so they can be displayed to the user via the GUI polling mechanism.
        """
        try:
            params = self.params

            # --- Create network (in background to avoid blocking GUI) ---
            # (Java: CAN myCAN = new CAN(BoardSize, k); or SWN(...))
            if self._topo_type == "CA":
                network = create_can(params.board_size, self._topo_radius)
                params.set_topology_ca(self._topo_radius)
            else:
                network = create_swn(params.board_size, self._topo_radius,
                                     self._topo_shortcuts)
                params.set_topology_sw(self._topo_radius, self._topo_shortcuts)

            # --- Create evolution engine ---
            # (Java: new EvoThread_CA_Mix(data, agent, can, frame, ...))
            self.evo_engine = EvolutionEngine(self.agent_grid, network, params)

            # --- Run evolution loop ---
            for stats in self.evo_engine.run():
                self._latest_stats = stats
                # Update reference to history (engine builds it incrementally)
                self.history = self.evo_engine.history
        except Exception:
            # Store the full traceback so _poll_evolution can display it
            import traceback
            self._evo_error = traceback.format_exc()
        finally:
            self.evo_running = False

    def _poll_evolution(self):
        """
        Periodically update the GUI while evolution runs.

        Corresponds to Java: frame.pw.paintImmediately() calls inside
        EvoThread_CA.run() that force Swing repaints between generations.

        tkinter is NOT thread-safe. We use after() polling on the main thread
        instead of updating the GUI directly from the worker thread.

        This method also checks for errors from the background thread and
        displays them to the user, preventing silent failures.
        """
        # --- Check for background thread errors ---
        if self._evo_error is not None:
            self._enable_controls_after_evolution()
            if self.evo_thread is not None:
                self.evo_thread.join(timeout=1.0)
            self.status_bar.config(text="  Error during evolution — see error message.")
            self.progress_label.config(text="❌  Error occurred")
            # If partial history was accumulated before the error, enable
            # the slider so the user can still browse completed generations.
            if self.history:
                self.agent_grid = self.history[-1]  # Point to snapshot, not live grid
                self._enable_gen_slider()
            self.evo_engine = None          # Release engine to free memory
            messagebox.showerror(
                "Evolution Error",
                f"An error occurred during evolution:\n\n{self._evo_error}")
            self._evo_error = None  # Clear so it's not shown again
            return

        # --- Update progress if new stats available ---
        if self._latest_stats is not None:
            gen = self._latest_stats['generation']
            avg_f = self._latest_stats['avg_fitness']
            self.progress_var.set(gen + 1)
            self.progress_label.config(
                text=f"Generation {gen} / {self.params.generations - 1}   |   "
                     f"Avg Fitness: {avg_f:.1f}")
            # Wrap lattice update in try-except to handle transient thread-safety
            # issues (e.g., reading agent grid while it's being modified)
            try:
                self._update_lattice(title=f"Generation {gen}")
            except Exception:
                pass  # Will be correct on next poll
            self.status_bar.config(
                text=f"  Evolving... Gen {gen}  |  Avg Fitness = {avg_f:.1f}")

        # --- Schedule next poll or finalize ---
        if self.evo_running:
            self.after(250, self._poll_evolution)
        else:
            # Evolution finished successfully
            # Point agent_grid to the last history snapshot (immutable deep
            # copy) instead of the live grid that was mutated during selection.
            if self.history:
                self.agent_grid = self.history[-1]
            try:
                self._update_lattice(
                    title=f"Final — Generation {self.params.generations - 1}")
            except Exception:
                pass
            self.progress_label.config(text="✅  Evolution Complete")
            self._enable_controls_after_evolution()
            # Join the background thread to ensure clean shutdown
            if self.evo_thread is not None:
                self.evo_thread.join(timeout=1.0)
            # Enable generation slider for post-evolution replay
            self._enable_gen_slider()
            # Release engine reference to free network/cache memory
            self.evo_engine = None
            self.status_bar.config(text="  Evolution complete.  Use slider or Analysis buttons to view results.")
            messagebox.showinfo("Complete",
                                "Evolution finished!\n\n"
                                "Use the generation slider to browse each time step,\n"
                                "or the Analysis buttons to view charts.")

    # =====================================================================
    #  Analysis Methods  (Java: DiagramFrame.Statistic1 ~ Statistic8)
    # =====================================================================
    def _check_history(self) -> bool:
        if not self.history:
            messagebox.showwarning("No Data",
                                   "No evolution data available.\n"
                                   "Please run an evolution first.")
            return False
        return True

    def _show_chart_window(self, title: str, fig: Figure):
        """Open a Toplevel window displaying a matplotlib Figure."""
        win = tk.Toplevel(self)
        win.title(title)
        win.geometry("850x550")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _show_strategy_dynamics(self):
        """
        16-strategy evolutionary dynamics chart.
        (Java: DiagramFrame type=1 -> Statistic1)
        """
        if not self._check_history(): return
        counts = compute_strategy_counts(self.history, self.params.strategy_length)
        fig = create_strategy_dynamics_chart(
            counts, title="Evolutionary Dynamics — All 16 Strategies")
        self._show_chart_window("Strategy Dynamics", fig)

    def _show_four_strategies(self):
        """
        Four key strategies chart: ALL-C, TFT, PAVLOV, ALL-D.
        (Java: DiagramFrame type=11/12 -> Statistic8)
        """
        if not self._check_history(): return
        counts = compute_strategy_counts(self.history, self.params.strategy_length)
        four = extract_four_strategies(counts)
        topo = self.params.topology_type or "N/A"
        fig = create_four_strategy_chart(
            four, title=f"Four Key Strategies — {topo}")
        self._show_chart_window("Four Key Strategies", fig)

    def _show_avg_fitness(self):
        """
        Average fitness per generation.
        (Java: DiagramFrame type=3 -> Statistic3)
        """
        if not self._check_history(): return
        avg = compute_avg_fitness(self.history, self.params.num_agents)
        fig = create_avg_fitness_chart(avg, title="Average Fitness per Generation")
        self._show_chart_window("Average Fitness", fig)

    def _show_fitness_quartiles(self):
        """
        Top / Bottom 25% fitness comparison.
        (Java: DiagramFrame type=2 -> Statistic2)
        """
        if not self._check_history(): return
        quartiles = compute_fitness_quartiles(self.history)
        fig = create_fitness_quartile_chart(quartiles)
        self._show_chart_window("Fitness Quartiles", fig)

    # =====================================================================
    #  Save / Load  (Java: IO/EvoDataOutput.java, IO/EvoDataInput.java)
    # =====================================================================
    def _save_data(self):
        if not self._check_history(): return
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not filepath: return
        data = {'params': self.params, 'history': self.history}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        messagebox.showinfo("Saved", f"Data saved to:\n{filepath}")

    def _sync_entries_from_params(self):
        """
        Update all GUI Entry widgets to reflect current self.params values.

        Called after loading saved data so that the Entry fields display the
        loaded parameter values instead of stale prior values.  This ensures
        that a subsequent 'Start Evolution' will correctly re-read the
        matching values from the entries.
        """
        p = self.params

        # Helper: overwrite an Entry widget's text (handles disabled state)
        def _set_entry(entry, value):
            prev_state = str(entry.cget('state'))
            entry.config(state='normal')
            entry.delete(0, tk.END)
            entry.insert(0, str(value))
            entry.config(state=prev_state)

        # [1] Simulation parameters
        _set_entry(self.param_entries['board_size'], p.board_size)
        _set_entry(self.param_entries['memory_length'], p.memory_length)
        _set_entry(self.param_entries['ipd_rounds'], p.ipd_rounds)
        _set_entry(self.param_entries['generations'], p.generations)
        _set_entry(self.param_entries['mutation_rate'], p.mutation_rate)
        _set_entry(self.param_entries['crossover_rate'], p.crossover_rate)
        self._update_agents_label()

        # [2] Network topology
        if p.topology_type == "Small-World":
            self.topology_var.set("SW")
        else:
            self.topology_var.set("CA")
        self._on_topology_change()  # enable/disable shortcuts entry
        _set_entry(self.radius_entry, p.radius)
        _set_entry(self.shortcuts_entry, p.shortcuts)

        # [3] SRAC settings
        self.srac_enabled.set(p.selfaware)
        self._on_srac_toggle()  # enable/disable SRAC entries
        _set_entry(self.srac_ratio_entry, p.selfaware_ratio)
        _set_entry(self.thresh_entries['f_low'], p.f_low_ratio)
        _set_entry(self.thresh_entries['f_high'], p.f_high_ratio)
        _set_entry(self.thresh_entries['r_low'], p.r_low_ratio)
        _set_entry(self.thresh_entries['r_high'], p.r_high_ratio)

    def _load_data(self):
        # Guard: cannot load while evolution is running
        if self.evo_running:
            messagebox.showwarning("Running",
                                   "Cannot load data while evolution is running.")
            return

        filepath = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not filepath: return
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.params = data['params']
            self.history = data['history']

            # Sync GUI Entry widgets with loaded parameters so the user
            # sees the correct values (and a subsequent Start Evolution
            # will use these values).  Without this, the Entry fields
            # would still show the OLD values from before loading.
            self._sync_entries_from_params()

            if self.history:
                self.agent_grid = self.history[-1]
                self._update_lattice(
                    title=f"Loaded — {len(self.history)} generations")
                # Enable generation slider so user can scrub through
                # all loaded generations (Java: setStatusPanel1 + JScrollBar)
                self._enable_gen_slider()
            messagebox.showinfo("Loaded",
                                f"Loaded {len(self.history)} generations of data.\n"
                                "Use the generation slider to browse each time step,\n"
                                "or Analysis buttons to view charts.")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load:\n{e}")

    # =====================================================================
    #  Batch Experiments  (Java: Experiment_CA.java / Experiment_SW.java)
    # =====================================================================
    def _batch_experiment(self, topology: str):
        """
        Launch a batch experiment window with multiple SRAC ratios.

        Corresponds to Java: Analysis/Experiment_CA.java (extends Thread)
        which runs N independent simulations for each SRAC ratio and
        aggregates strategy statistics and fitness data.
        """
        # Ensure current parameters are parsed
        try:
            p = self.params
            p.board_size = int(self.param_entries['board_size'].get())
            p.memory_length = int(self.param_entries['memory_length'].get())
            p.ipd_rounds = int(self.param_entries['ipd_rounds'].get())
            p.generations = int(self.param_entries['generations'].get())
            p.mutation_rate = float(self.param_entries['mutation_rate'].get())
            p.crossover_rate = float(self.param_entries['crossover_rate'].get())
            p.update_derived()
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values."); return

        radius = simpledialog.askinteger("Neighborhood",
                                         "k-neighborhood radius:", initialvalue=1)
        if radius is None: return

        shortcuts = 0
        if topology == "SW":
            shortcuts = simpledialog.askinteger("Shortcuts",
                                                "Shortcuts per node:", initialvalue=1)
            if shortcuts is None: return

        ratios_str = simpledialog.askstring(
            "Mixing Ratios",
            "Enter SRAC ratios (comma-separated):",
            initialvalue="0, 0.1, 0.3, 0.5, 1.0")
        if ratios_str is None: return
        ratios = [float(x.strip()) for x in ratios_str.split(",")]

        thresholds = simpledialog.askstring(
            "Thresholds", "F_Low, F_High, R_Low, R_High:",
            initialvalue="-1.0,1.0,-1.0,1.0")
        if thresholds is None: return
        f_low, f_high, r_low, r_high = [float(x.strip()) for x in thresholds.split(",")]

        runs = simpledialog.askinteger("Runs",
                                        "Number of runs per ratio:", initialvalue=1)
        if runs is None: return

        BatchExperimentWindow(self, self.params, topology, radius, shortcuts,
                              ratios, f_low, f_high, r_low, r_high, runs)

    # =====================================================================
    #  About  (Java: UI/Version.java)
    # =====================================================================
    def _show_about(self):
        messagebox.showinfo(
            "About SRAC-IPD Simulator",
            "SRAC-IPD Simulator (Python 3)\n\n"
            "Self-Reputation Awareness Component in\n"
            "Evolutionary Spatial IPD Game\n\n"
            "Based on: Huang & Lee, PLoS ONE\n"
            "Original Java: HikiChen, NCTU CIS Lab\n\n"
            "Python libraries: networkx, numpy, scipy, matplotlib")


# =========================================================================
#  Batch Experiment Window  (Java: Experiment_CA.java / Experiment_SW.java)
# =========================================================================
class BatchExperimentWindow(tk.Toplevel):
    """
    Batch experiment runner that tests multiple SRAC ratios.

    Corresponds to Java: Analysis/Experiment_CA.java (extends Thread)
    """

    def __init__(self, parent, base_params: SimParameters,
                 topology: str, radius: int, shortcuts: int,
                 ratios: list, f_low: float, f_high: float,
                 r_low: float, r_high: float, runs: int):
        super().__init__(parent)
        self.title(f"Batch Experiment ({topology})")
        self.geometry("500x300")
        # Make this window modal so the user cannot interact with the
        # main window while the batch experiment is running.
        self.transient(parent)
        self.grab_set()

        self.base_params = base_params
        self.topology = topology
        self.radius = radius
        self.shortcuts = shortcuts
        self.ratios = ratios
        self.f_low, self.f_high = f_low, f_high
        self.r_low, self.r_high = r_low, r_high
        self.runs = runs

        tk.Label(self, text="Batch Experiment Running...",
                 font=('Helvetica', 12, 'bold')).pack(pady=10)
        self.status_label = tk.Label(self, text="Initializing...", font=('Helvetica', 10))
        self.status_label.pack(pady=5)
        self.progress_var = tk.IntVar(value=0)
        total_sims = len(ratios) * runs
        self.progress_bar = ttk.Progressbar(self, variable=self.progress_var,
                                             maximum=total_sims)
        self.progress_bar.pack(fill='x', padx=20, pady=5)

        self.all_avg_fitness = None
        self._experiment_done = False
        self._experiment_error = None   # Stores traceback if background thread crashes
        self._sim_count = 0
        self._current_status = "Starting..."

        self.thread = threading.Thread(target=self._run_experiment, daemon=True)
        self.thread.start()
        self.after(500, self._poll_experiment)

    def _run_experiment(self):
        try:
            params = self.base_params.copy()
            gen = params.generations
            n_ratios = len(self.ratios)
            self.all_avg_fitness = np.zeros((gen, n_ratios), dtype=float)

            sim_count = 0
            for r_idx, ratio in enumerate(self.ratios):
                ratio_total_fitness = np.zeros(gen, dtype=float)
                for run_idx in range(self.runs):
                    self._current_status = f"Ratio {ratio:.0%}, Run {run_idx+1}/{self.runs}"
                    sim_count += 1
                    self._sim_count = sim_count

                    agent_grid = create_agent_grid(params.board_size, params.memory_length)
                    if self.topology == "CA":
                        network = create_can(params.board_size, self.radius)
                        params.set_topology_ca(self.radius)
                    else:
                        network = create_swn(params.board_size, self.radius, self.shortcuts)
                        params.set_topology_sw(self.radius, self.shortcuts)

                    params.set_selfaware_params(
                        ratio > 0, ratio,
                        self.f_low, self.f_high, self.r_low, self.r_high)

                    engine = EvolutionEngine(agent_grid, network, params)
                    for s in engine.run():
                        g = s['generation']
                        ratio_total_fitness[g] += s['avg_fitness']

                self.all_avg_fitness[:, r_idx] = ratio_total_fitness / self.runs
        except Exception:
            import traceback
            self._experiment_error = traceback.format_exc()
        finally:
            self._experiment_done = True

    def _poll_experiment(self):
        self.status_label.config(text=self._current_status)
        self.progress_var.set(self._sim_count)

        if self._experiment_done:
            # Check if an error occurred in the background thread
            if self._experiment_error is not None:
                self.status_label.config(text="❌  Experiment Error")
                messagebox.showerror(
                    "Experiment Error",
                    f"An error occurred during the batch experiment:\n\n"
                    f"{self._experiment_error}")
                self._experiment_error = None
                return
            self.status_label.config(text="✅  Experiment Complete!")
            self._show_results()
        else:
            self.after(500, self._poll_experiment)

    def _show_results(self):
        from .visualization import create_experiment_comparison_chart
        ratio_labels = [f"{r:.0%}" for r in self.ratios]
        fig = create_experiment_comparison_chart(
            self.all_avg_fitness, ratio_labels,
            title=f"Average Fitness Comparison ({self.topology})",
            ylabel="Average Fitness")
        win = tk.Toplevel(self)
        win.title("Experiment Results")
        win.geometry("850x550")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)
