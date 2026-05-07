from __future__ import annotations

import threading
import tkinter as tk
from importlib import util as importlib_util
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import TYPE_CHECKING

import constants
from synergie import operations
from synergie.config import (
    DEFAULT_COMBINATION_GAP_FRAMES,
    DEFAULT_DETECTION_THRESHOLD,
    DEFAULT_SMOOTHING_SIGMA,
    GYRO_SATURATION_WARNING_THRESHOLD,
    SUCCESS_WINDOW_START,
    TYPE_WINDOW_FRAMES,
)

if TYPE_CHECKING:
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession


class SynergieToolsApp:
    TRAIN_ARCHITECTURES = {
        "type": ["inceptiontime", "transformer"],
        "success": ["tcn", "lstm"],
    }
    ANNOTATION_SHORTCUTS = {
        "t": "toe_loop",
        "f": "flip",
        "z": "lutz",
        "s": "salchow",
        "l": "loop",
        "a": "axel",
    }

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Synergie Tools")
        self.root.geometry("1280x860")

        self.status_var = tk.StringVar(value="Ready")
        self.csv_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.session_var = tk.StringVar(value=sorted(constants.sessions)[0])
        self.session_files_var = tk.StringVar(value=[])
        self.session_folder_summary_var = tk.StringVar(value="")
        new_data_directories = [item["relative_path"] for item in operations.list_new_data_directories()]
        self.new_data_directory_var = tk.StringVar(value=new_data_directories[0] if new_data_directories else "")
        self.new_data_files_var = tk.StringVar(value=[])
        self.new_data_selected_file_var = tk.StringVar()
        self.new_data_output_var = tk.StringVar()
        self.new_data_summary_var = tk.StringVar(value="No file selected.")
        self.annotation_files_var = tk.StringVar(value=[])
        self.annotation_summary_var = tk.StringVar(value="No annotation file selected.")
        self.annotation_type_var = tk.StringVar(value="toe_loop")
        self.annotation_turn_var = tk.StringVar(value="")
        self.annotation_success_var = tk.StringVar(value="2")
        self.annotation_review_status_var = tk.StringVar(value="normal")
        self.annotation_athlete_var = tk.StringVar(value="")
        self.annotation_exclusion_hint_var = tk.StringVar(value="")
        self.annotation_video_path_var = tk.StringVar()
        self.annotation_video_info_var = tk.StringVar(value="No video loaded.")
        self.annotation_sensor_sync_var = tk.StringVar(value="No sync offset saved for current sensor.")
        self.annotation_video_time_var = tk.StringVar(value="00:00.000")
        self.annotation_video_slider_var = tk.DoubleVar(value=0.0)
        self.new_session_id_var = tk.StringVar()
        self.new_session_path_var = tk.StringVar()
        self.new_session_synchro_var = tk.StringVar()
        self.train_task_var = tk.StringVar(value="type")
        self.train_architecture_var = tk.StringVar(value=self.TRAIN_ARCHITECTURES["type"][0])
        self.dataset_var = tk.StringVar(value="data/annotated/total")
        self.epochs_var = tk.StringVar(value="10")
        self.train_dataset_stats_var = tk.StringVar(value="Dataset stats not loaded yet.")
        self.train_quality_summary_var = tk.StringVar(value="No training run yet.")
        self.use_pretrained_var = tk.BooleanVar(value=False)
        self.pretrained_model_var = tk.StringVar()
        self.pretrained_models_summary_var = tk.StringVar(value="No pretrained model scan yet.")
        self.quality_dataset_var = tk.StringVar(value="data/annotated/total")
        self.quality_summary_var = tk.StringVar(value="Run the quality control analysis to inspect suspicious jumps.")
        self.quality_details_var = tk.StringVar(value="No suspicious jump selected.")
        self.quality_suspicious_var = tk.StringVar(value=[])

        self.inspect_csv_path_var = tk.StringVar()
        self.inspect_session_var = tk.StringVar(value=sorted(constants.sessions)[0])
        self.inspect_session_files_var = tk.StringVar(value=[])
        self.inspect_folder_summary_var = tk.StringVar(value="")
        self.threshold_var = tk.DoubleVar(value=DEFAULT_DETECTION_THRESHOLD)
        self.threshold_label_var = tk.StringVar()
        self.sigma_var = tk.DoubleVar(value=DEFAULT_SMOOTHING_SIGMA)
        self.sigma_label_var = tk.StringVar()
        self.gap_var = tk.IntVar(value=DEFAULT_COMBINATION_GAP_FRAMES)
        self.gap_label_var = tk.StringVar()
        self.process_selected_file_info_var = tk.StringVar(value="No file selected.")
        self.inspect_selected_file_info_var = tk.StringVar(value="No file selected.")

        self.inspect_dataframe = None
        self.inspect_session = None
        self.detected_jumps: list = []
        self._new_data_files_cache: list[dict] = []
        self._annotation_files_cache: list[Path] = []
        self.figure = None
        self.axes = None
        self.canvas = None
        self.annotation_figure = None
        self.annotation_ax = None
        self.annotation_acc_ax = None
        self.annotation_canvas = None
        self.train_figure = None
        self.train_axes = None
        self.train_canvas = None
        self.quality_figure = None
        self.quality_axes = None
        self.quality_canvas = None
        self.annotation_dataframe = None
        self.annotation_file_path: Path | None = None
        self.quality_analysis: dict | None = None
        self.annotation_type_buttons: list[ttk.Radiobutton] = []
        self.annotation_turn_buttons: list[ttk.Radiobutton] = []
        self.annotation_success_buttons: list[ttk.Radiobutton] = []
        self.annotation_metadata: dict = {}
        self.annotation_video_capture = None
        self.annotation_video_fps = 0.0
        self.annotation_video_frame_count = 0
        self.annotation_video_duration_ms = 0.0
        self.annotation_video_current_ms = 0.0
        self._annotation_video_photo = None
        self._annotation_playback_after_id = None

        self._build_layout()
        self._populate_sessions()
        self._sync_slider_labels()
        self._refresh_session_file_lists()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Synergie Tools",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Petite interface pour explorer les sessions, traiter un CSV IMU, inspecter les sauts et lancer un entrainement.",
        ).grid(row=1, column=0, sticky="w")

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.notebook = notebook

        sessions_tab = ttk.Frame(notebook, padding=12)
        process_tab = ttk.Frame(notebook, padding=12)
        new_data_tab = ttk.Frame(notebook, padding=12)
        annotate_tab = ttk.Frame(notebook, padding=12)
        inspect_tab = ttk.Frame(notebook, padding=12)
        train_tab = ttk.Frame(notebook, padding=12)
        quality_tab = ttk.Frame(notebook, padding=12)
        notes_tab = ttk.Frame(notebook, padding=12)
        notebook.add(sessions_tab, text="Sessions")
        notebook.add(process_tab, text="Process CSV")
        notebook.add(new_data_tab, text="New Data")
        notebook.add(annotate_tab, text="Annotate")
        notebook.add(inspect_tab, text="Inspect IMU")
        notebook.add(train_tab, text="Train")
        notebook.add(quality_tab, text="Quality Control")
        notebook.add(notes_tab, text="Algo Notes")

        sessions_tab.columnconfigure(0, weight=1)
        sessions_tab.columnconfigure(1, weight=0)
        sessions_tab.rowconfigure(0, weight=1)

        self.sessions_text = scrolledtext.ScrolledText(sessions_tab, height=18, wrap=tk.WORD)
        self.sessions_text.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_sessions_side_panel(sessions_tab)
        ttk.Button(sessions_tab, text="Refresh sessions", command=self._refresh_all_sessions).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._build_process_tab(process_tab)
        self._build_new_data_tab(new_data_tab)
        self._build_annotate_tab(annotate_tab)
        self._build_inspect_tab(inspect_tab)
        self._build_train_tab(train_tab)
        self._build_quality_tab(quality_tab)
        self._build_notes_tab(notes_tab)

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.root.bind_all("<KeyPress>", self._on_global_keypress)

    def _build_process_tab(self, parent: ttk.Frame) -> None:
        for index in range(3):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)

        ttk.Label(parent, text="Input CSV").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.csv_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse", command=self._pick_csv).grid(row=0, column=2, sticky="e")

        ttk.Label(parent, text="Output CSV").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.output_path_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Save as", command=self._pick_output).grid(row=1, column=2, sticky="e")

        ttk.Label(parent, text="Session").grid(row=2, column=0, sticky="w", pady=4)
        self.process_session_box = ttk.Combobox(parent, textvariable=self.session_var, values=operations.list_sessions(), state="readonly")
        self.process_session_box.grid(row=2, column=1, sticky="w", padx=8)
        self.process_session_box.bind("<<ComboboxSelected>>", self._on_process_session_changed)

        ttk.Label(parent, text="Session CSV files").grid(row=3, column=0, sticky="nw", pady=4)
        self.process_session_files = tk.Listbox(parent, listvariable=self.session_files_var, height=6, exportselection=False)
        self.process_session_files.grid(row=3, column=1, columnspan=2, sticky="nsew", padx=8)
        self.process_session_files.bind("<<ListboxSelect>>", self._on_process_file_selected)

        ttk.Label(parent, textvariable=self.session_folder_summary_var, justify=tk.LEFT).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(parent, textvariable=self.process_selected_file_info_var, justify=tk.LEFT).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Button(parent, text="Process file", command=self._run_process_file).grid(row=6, column=0, sticky="w", pady=(12, 12))

        self.process_log = scrolledtext.ScrolledText(parent, height=18, wrap=tk.WORD)
        self.process_log.grid(row=7, column=0, columnspan=3, sticky="nsew")
        parent.rowconfigure(7, weight=1)

    def _build_inspect_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Detection Controls", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Input CSV").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.inspect_csv_path_var, width=44).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse", command=self._pick_inspect_csv).grid(row=0, column=2, sticky="e")

        ttk.Label(controls, text="Session").grid(row=1, column=0, sticky="w", pady=4)
        self.inspect_session_box = ttk.Combobox(
            controls,
            textvariable=self.inspect_session_var,
            values=operations.list_sessions(),
            state="readonly",
            width=16,
        )
        self.inspect_session_box.grid(row=1, column=1, sticky="w", padx=8)
        self.inspect_session_box.bind("<<ComboboxSelected>>", self._on_inspect_session_changed)

        ttk.Label(controls, text="Session CSV files").grid(row=2, column=0, sticky="nw", pady=(12, 0))
        self.inspect_session_files = tk.Listbox(controls, listvariable=self.inspect_session_files_var, height=5, exportselection=False)
        self.inspect_session_files.grid(row=2, column=1, columnspan=2, sticky="ew", padx=8)
        self.inspect_session_files.bind("<<ListboxSelect>>", self._on_inspect_file_selected)
        self.inspect_session_files.bind("<Double-1>", self._on_inspect_file_double_clicked)

        ttk.Label(controls, textvariable=self.inspect_folder_summary_var, justify=tk.LEFT).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(controls, textvariable=self.inspect_selected_file_info_var, justify=tk.LEFT).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Label(controls, text="2nd derivative threshold").grid(row=5, column=0, sticky="w", pady=(12, 0))
        tk.Scale(
            controls,
            from_=-2.0,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        ).grid(row=5, column=1, sticky="ew", padx=8)
        ttk.Label(controls, textvariable=self.threshold_label_var).grid(row=5, column=2, sticky="w")

        ttk.Label(controls, text="Smoothing sigma").grid(row=6, column=0, sticky="w", pady=(12, 0))
        tk.Scale(
            controls,
            from_=1,
            to=60,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.sigma_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        ).grid(row=6, column=1, sticky="ew", padx=8)
        ttk.Label(controls, textvariable=self.sigma_label_var).grid(row=6, column=2, sticky="w")

        ttk.Label(controls, text="Combination gap (frames)").grid(row=7, column=0, sticky="w", pady=(12, 0))
        tk.Scale(
            controls,
            from_=60,
            to=360,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.gap_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        ).grid(row=7, column=1, sticky="ew", padx=8)
        ttk.Label(controls, textvariable=self.gap_label_var).grid(row=7, column=2, sticky="w")

        actions = ttk.Frame(controls)
        actions.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(actions, text="Load and detect", command=self._run_inspection).grid(row=0, column=0, sticky="w")
        ttk.Button(actions, text="Refresh plots", command=self._redraw_plots).grid(row=0, column=1, sticky="w", padx=(8, 0))

        jump_frame = ttk.LabelFrame(parent, text="Detected Jumps", padding=12)
        jump_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        jump_frame.columnconfigure(0, weight=1)
        jump_frame.rowconfigure(0, weight=1)
        self.jump_listbox = tk.Listbox(jump_frame, exportselection=False, width=42)
        self.jump_listbox.grid(row=0, column=0, sticky="nsew")
        self.jump_listbox.bind("<<ListboxSelect>>", self._on_jump_selected)

        plots_frame = ttk.Frame(parent)
        plots_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        plots_frame.columnconfigure(0, weight=1)
        plots_frame.rowconfigure(0, weight=1)
        self.plot_container = ttk.Frame(plots_frame)
        self.plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_plot_canvas()

    def _build_new_data_tab(self, parent: ttk.Frame) -> None:
        for index in range(3):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)
        parent.rowconfigure(4, weight=1)

        ttk.Label(parent, text="Folder").grid(row=0, column=0, sticky="w", pady=4)
        self.new_data_directory_box = ttk.Combobox(parent, textvariable=self.new_data_directory_var, state="readonly")
        self.new_data_directory_box.grid(row=0, column=1, sticky="ew", padx=8)
        self.new_data_directory_box.bind("<<ComboboxSelected>>", self._on_new_data_directory_changed)

        ttk.Button(parent, text="Refresh folders", command=self._refresh_new_data_directories).grid(row=0, column=2, sticky="e")

        ttk.Label(parent, text="IMU files").grid(row=1, column=0, sticky="nw", pady=4)
        self.new_data_files = tk.Listbox(parent, listvariable=self.new_data_files_var, height=8, exportselection=False)
        self.new_data_files.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=8)
        self.new_data_files.bind("<<ListboxSelect>>", self._on_new_data_file_selected)

        ttk.Label(parent, text="For annotation CSV").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.new_data_output_var).grid(row=2, column=1, sticky="ew", padx=8)

        ttk.Button(parent, text="Process for annotation", command=self._run_process_new_data_file).grid(row=3, column=0, sticky="w", pady=(12, 8))
        ttk.Label(parent, textvariable=self.new_data_summary_var, justify=tk.LEFT).grid(row=3, column=1, columnspan=2, sticky="w", padx=8)

        self.new_data_log = scrolledtext.ScrolledText(parent, height=14, wrap=tk.WORD)
        self.new_data_log.grid(row=4, column=0, columnspan=3, sticky="nsew")
        self._refresh_new_data_directories()

    def _build_annotate_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(parent)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=0)
        left_panel.rowconfigure(2, weight=1)

        ttk.Button(left_panel, text="Refresh annotation files", command=self._refresh_annotation_files).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.annotation_files_listbox = tk.Listbox(left_panel, listvariable=self.annotation_files_var, exportselection=False, height=5, width=34)
        self.annotation_files_listbox.grid(row=1, column=0, sticky="ew")
        self.annotation_files_listbox.bind("<<ListboxSelect>>", self._on_annotation_file_selected)

        jumps_panel = ttk.LabelFrame(left_panel, text="Session Timeline", padding=8)
        jumps_panel.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        jumps_panel.columnconfigure(0, weight=1)
        jumps_panel.rowconfigure(1, weight=1)
        ttk.Label(jumps_panel, textvariable=self.annotation_summary_var, justify=tk.LEFT).grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.annotation_jump_listbox = tk.Listbox(jumps_panel, exportselection=False, height=16)
        self.annotation_jump_listbox.grid(row=1, column=0, sticky="nsew")
        self.annotation_jump_listbox.bind("<<ListboxSelect>>", self._on_annotation_jump_selected)

        right_panel = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        right_panel.grid(row=0, column=1, sticky="nsew")

        video_frame = ttk.LabelFrame(right_panel, text="Video Review", padding=8)
        video_frame.columnconfigure(1, weight=1)
        video_frame.rowconfigure(2, weight=1)

        ttk.Label(video_frame, text="Video file").grid(row=0, column=0, sticky="w")
        ttk.Entry(video_frame, textvariable=self.annotation_video_path_var).grid(row=0, column=1, sticky="ew", padx=8)
        video_buttons = ttk.Frame(video_frame)
        video_buttons.grid(row=0, column=2, sticky="e")
        ttk.Button(video_buttons, text="Browse", command=self._browse_annotation_video).grid(row=0, column=0, sticky="w")
        ttk.Button(video_buttons, text="Load", command=self._load_annotation_video_from_entry).grid(row=0, column=1, sticky="w", padx=(6, 0))

        ttk.Label(video_frame, textvariable=self.annotation_video_info_var, justify=tk.LEFT, wraplength=360).grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(6, 8),
        )

        self.annotation_video_label = ttk.Label(
            video_frame,
            text="Load a session video to review jumps.",
            anchor="center",
            relief="sunken",
        )
        self.annotation_video_label.grid(row=2, column=0, columnspan=3, sticky="nsew")

        video_timeline = ttk.Frame(video_frame)
        video_timeline.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        video_timeline.columnconfigure(0, weight=1)
        self.annotation_video_slider = tk.Scale(
            video_timeline,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.annotation_video_slider_var,
            command=lambda _value: self.annotation_video_time_var.set(self._format_video_ms(self.annotation_video_slider_var.get())),
        )
        self.annotation_video_slider.grid(row=0, column=0, sticky="ew")
        self.annotation_video_slider.bind("<ButtonRelease-1>", self._on_annotation_video_slider_released)
        ttk.Label(video_timeline, textvariable=self.annotation_video_time_var, width=12).grid(row=0, column=1, sticky="e", padx=(8, 0))

        video_controls = ttk.Frame(video_frame)
        video_controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(video_controls, text="-1s", command=lambda: self._seek_annotation_video_relative(-1000)).grid(row=0, column=0, sticky="w")
        ttk.Button(video_controls, text="+1s", command=lambda: self._seek_annotation_video_relative(1000)).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Button(video_controls, text="Go to jump", command=self._seek_annotation_video_to_current_jump).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(video_controls, text="Play x5 to jump", command=self._play_annotation_to_current_jump).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Button(video_controls, text="Stop", command=self._stop_annotation_playback).grid(row=0, column=4, sticky="w", padx=(6, 0))

        ttk.Label(video_frame, textvariable=self.annotation_sensor_sync_var, justify=tk.LEFT, wraplength=360).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(8, 4),
        )
        sync_buttons = ttk.Frame(video_frame)
        sync_buttons.grid(row=6, column=0, columnspan=3, sticky="w")
        ttk.Button(sync_buttons, text="Sync current IMU at this frame", command=self._sync_current_sensor_to_video).grid(row=0, column=0, sticky="w")
        ttk.Button(sync_buttons, text="Clear current IMU sync", command=self._clear_current_sensor_sync).grid(row=0, column=1, sticky="w", padx=(6, 0))

        plot_column = ttk.Frame(right_panel)
        plot_column.columnconfigure(0, weight=1)
        plot_column.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(plot_column, text="Annotation", padding=8)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        controls.columnconfigure(0, weight=1)

        plot_frame = ttk.LabelFrame(plot_column, text="Jump Signals", padding=8)
        plot_frame.grid(row=1, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.annotation_plot_container = ttk.Frame(plot_frame)
        self.annotation_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_annotation_plot_canvas()
        right_panel.add(video_frame, weight=3)
        right_panel.add(plot_column, weight=2)

        ttk.Label(controls, text="Athlete ID").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.annotation_athlete_var).grid(row=1, column=0, sticky="w", pady=(0, 8))

        ttk.Label(controls, text="Jump type").grid(row=2, column=0, sticky="w")
        type_frame = ttk.Frame(controls)
        type_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        type_frame.columnconfigure(0, weight=1)
        type_frame.columnconfigure(1, weight=1)
        shortcut_labels = {
            "toe_loop": ("Toe loop", 0),
            "flip": ("Flip", 0),
            "lutz": ("Lutz", 2),
            "salchow": ("Salchow", 0),
            "loop": ("Loop", 0),
            "axel": ("Axel", 0),
        }
        toe_frame = ttk.LabelFrame(type_frame, text="Piques", padding=6)
        toe_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        edge_frame = ttk.LabelFrame(type_frame, text="De carre", padding=6)
        edge_frame.grid(row=0, column=1, sticky="nsew")
        self.annotation_type_buttons = []
        for index, (key, _label, _value) in enumerate(operations.ANNOTATION_TOE_JUMP_OPTIONS):
            display_label, underline_index = shortcut_labels[key]
            button = ttk.Radiobutton(
                toe_frame,
                text=display_label,
                value=key,
                variable=self.annotation_type_var,
                command=self._sync_annotation_turn_options,
                underline=underline_index,
            )
            button.grid(row=index, column=0, sticky="w")
            self.annotation_type_buttons.append(button)
        for index, (key, _label, _value) in enumerate(operations.ANNOTATION_EDGE_JUMP_OPTIONS):
            display_label, underline_index = shortcut_labels[key]
            button = ttk.Radiobutton(
                edge_frame,
                text=display_label,
                value=key,
                variable=self.annotation_type_var,
                command=self._sync_annotation_turn_options,
                underline=underline_index,
            )
            button.grid(row=index, column=0, sticky="w")
            self.annotation_type_buttons.append(button)

        ttk.Label(controls, text="Turns").grid(row=4, column=0, sticky="w")
        turns_frame = ttk.Frame(controls)
        turns_frame.grid(row=5, column=0, sticky="w", pady=(0, 8))
        self.annotation_turn_buttons = []
        for column_index, turn_value in enumerate(["1", "2", "3", "4"]):
            button = ttk.Radiobutton(
                turns_frame,
                text=turn_value,
                value=turn_value,
                variable=self.annotation_turn_var,
                underline=0,
            )
            button.grid(row=0, column=column_index, sticky="w", padx=(0, 8 if column_index < 3 else 0))
            self.annotation_turn_buttons.append(button)

        ttk.Label(controls, text="Success").grid(row=6, column=0, sticky="w")
        success_frame = ttk.Frame(controls)
        success_frame.grid(row=7, column=0, sticky="w", pady=(0, 8))
        success_frame.columnconfigure(0, weight=1)
        success_frame.columnconfigure(1, weight=1)
        self.annotation_success_buttons = []
        for label, value in [("Fall", "0"), ("Success", "1"), ("Unknown", "2")]:
            column_index = 0 if value == "0" else 1 if value == "1" else 0
            row_index = 0 if value in {"0", "1"} else 1
            button = ttk.Radiobutton(success_frame, text=label, value=value, variable=self.annotation_success_var)
            button.grid(
                row=row_index,
                column=column_index,
                sticky="w",
                padx=(0, 12),
            )
            self.annotation_success_buttons.append(button)

        ttk.Label(controls, text="Review status").grid(row=8, column=0, sticky="w")
        review_frame = ttk.Frame(controls)
        review_frame.grid(row=9, column=0, sticky="w", pady=(0, 8))
        review_frame.columnconfigure(0, weight=1)
        review_frame.columnconfigure(1, weight=1)
        for index, (value, label) in enumerate(operations.ANNOTATION_REVIEW_STATUS_OPTIONS):
            ttk.Radiobutton(
                review_frame,
                text=label,
                value=value,
                variable=self.annotation_review_status_var,
                command=self._sync_annotation_review_controls,
            ).grid(
                row=index // 2,
                column=index % 2,
                sticky="w",
                padx=(0, 12),
            )

        ttk.Label(
            controls,
            textvariable=self.annotation_exclusion_hint_var,
            justify=tk.LEFT,
            wraplength=360,
            foreground="firebrick",
        ).grid(row=10, column=0, sticky="w", pady=(0, 8))

        ttk.Button(controls, text="Save current annotation", command=self._save_current_annotation).grid(row=11, column=0, sticky="w", pady=(8, 0))
        self._sync_annotation_review_controls()
        self._refresh_annotation_files()

    def _build_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(9, 7), dpi=100)
        overview_ax = figure.add_subplot(211)
        zoom_ax = figure.add_subplot(212)
        self.figure = figure
        self.axes = (overview_ax, zoom_ax)
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_plots()

    def _build_train_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8, 3.8), dpi=100)
        loss_ax = figure.add_subplot(121)
        accuracy_ax = figure.add_subplot(122)
        self.train_figure = figure
        self.train_axes = (loss_ax, accuracy_ax)
        self.train_canvas = FigureCanvasTkAgg(figure, master=self.train_plot_container)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_training_plot()

    def _build_quality_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(9, 5.6), dpi=100)
        counts_ax = figure.add_subplot(221)
        duration_ax = figure.add_subplot(222)
        gyro_ax = figure.add_subplot(223)
        scatter_ax = figure.add_subplot(224)
        self.quality_figure = figure
        self.quality_axes = (counts_ax, duration_ax, gyro_ax, scatter_ax)
        self.quality_canvas = FigureCanvasTkAgg(figure, master=self.quality_plot_container)
        self.quality_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_quality_plot()

    def _build_annotation_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.5, 4.0), dpi=100)
        axis = figure.add_subplot(111)
        self.annotation_figure = figure
        self.annotation_ax = axis
        self.annotation_acc_ax = None
        self.annotation_canvas = FigureCanvasTkAgg(figure, master=self.annotation_plot_container)
        self.annotation_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_annotation_plot()

    def _annotation_video_supported(self) -> bool:
        return bool(importlib_util.find_spec("cv2")) and bool(importlib_util.find_spec("PIL"))

    def _build_sessions_side_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Add Session", padding=12)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Session ID").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_id_var, width=18).grid(row=0, column=1, sticky="ew")

        ttk.Label(panel, text="Relative path").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_path_var, width=22).grid(row=1, column=1, sticky="ew")

        ttk.Label(panel, text="Synchro").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_synchro_var, width=18).grid(row=2, column=1, sticky="ew")

        ttk.Button(panel, text="Add session", command=self._add_session_from_gui).grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Training Setup", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        train_task_box = ttk.Combobox(controls, textvariable=self.train_task_var, values=["type", "success"], state="readonly", width=18)
        train_task_box.grid(row=0, column=1, sticky="ew")
        train_task_box.bind("<<ComboboxSelected>>", self._on_train_task_changed)

        ttk.Label(controls, text="Architecture").grid(row=1, column=0, sticky="w", pady=4)
        self.train_architecture_box = ttk.Combobox(controls, textvariable=self.train_architecture_var, state="readonly", width=24)
        self.train_architecture_box.grid(row=1, column=1, sticky="ew")
        self._sync_train_architectures()

        ttk.Label(controls, text="Dataset").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.dataset_var, width=34).grid(row=2, column=1, sticky="ew")

        ttk.Label(controls, text="Epochs").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.epochs_var, width=10).grid(row=3, column=1, sticky="w")

        ttk.Checkbutton(
            controls,
            text="Start from pretrained model",
            variable=self.use_pretrained_var,
            command=self._sync_pretrained_controls,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 4))

        self.pretrained_model_box = ttk.Combobox(controls, textvariable=self.pretrained_model_var, state="readonly", width=34)
        self.pretrained_model_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.pretrained_model_box.bind("<<ComboboxSelected>>", self._on_pretrained_model_changed)

        buttons = ttk.Frame(controls)
        buttons.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        ttk.Button(buttons, text="Run training", command=self._run_train).grid(row=0, column=0, sticky="w")
        ttk.Button(buttons, text="Refresh dataset stats", command=self._refresh_training_dataset_stats).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(
            controls,
            textvariable=self.pretrained_models_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(
            controls,
            textvariable=self.train_dataset_stats_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(
            controls,
            textvariable=self.train_quality_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=9, column=0, columnspan=2, sticky="w")

        results = ttk.Frame(parent)
        results.grid(row=0, column=1, rowspan=2, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)

        train_plot_frame = ttk.LabelFrame(results, text="Training Curves", padding=8)
        train_plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        train_plot_frame.columnconfigure(0, weight=1)
        train_plot_frame.rowconfigure(0, weight=1)
        self.train_plot_container = ttk.Frame(train_plot_frame)
        self.train_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_train_plot_canvas()

        log_frame = ttk.LabelFrame(results, text="Training Log", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.train_log = scrolledtext.ScrolledText(log_frame, height=14, wrap=tk.WORD)
        self.train_log.grid(row=0, column=0, sticky="nsew")
        self._refresh_pretrained_models()
        self._sync_pretrained_controls()
        self._refresh_training_dataset_stats()

    def _build_quality_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Quality Control", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(4, weight=1)

        ttk.Label(controls, text="Dataset").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.quality_dataset_var, width=34).grid(row=1, column=0, sticky="ew", pady=(4, 8))
        buttons = ttk.Frame(controls)
        buttons.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(buttons, text="Run quality scan", command=self._run_quality_scan).grid(row=0, column=0, sticky="w")

        ttk.Label(
            controls,
            textvariable=self.quality_summary_var,
            justify=tk.LEFT,
            wraplength=320,
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))

        suspicious_frame = ttk.LabelFrame(controls, text="Suspicious jumps to review", padding=8)
        suspicious_frame.grid(row=4, column=0, sticky="nsew")
        suspicious_frame.columnconfigure(0, weight=1)
        suspicious_frame.rowconfigure(0, weight=1)
        self.quality_suspicious_listbox = tk.Listbox(suspicious_frame, listvariable=self.quality_suspicious_var, exportselection=False, height=18, width=42)
        self.quality_suspicious_listbox.grid(row=0, column=0, sticky="nsew")
        self.quality_suspicious_listbox.bind("<<ListboxSelect>>", self._on_quality_suspicious_selected)

        results = ttk.Frame(parent)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        results.rowconfigure(1, weight=0)

        plot_frame = ttk.LabelFrame(results, text="Distributions and outliers", padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.quality_plot_container = ttk.Frame(plot_frame)
        self.quality_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_quality_plot_canvas()

        details_frame = ttk.LabelFrame(results, text="Selected jump details", padding=8)
        details_frame.grid(row=1, column=0, sticky="ew")
        details_frame.columnconfigure(0, weight=1)
        ttk.Label(
            details_frame,
            textvariable=self.quality_details_var,
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")

    def _build_notes_tab(self, parent: ttk.Frame) -> None:
        notes = scrolledtext.ScrolledText(parent, height=20, wrap=tk.WORD)
        notes.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        notes.insert(
            tk.END,
            "Algorithm audit notes\n\n"
            "- Jump segmentation starts from derivatives of Gyr_X only.\n"
            "- Rotation magnitude is exported as an absolute value.\n"
            "- Signed rotation is now kept for diagnostics as `signed_rotations`.\n"
            "- `rotation_direction` is labelled as positive/negative/unknown.\n"
            "- If the training set contains mostly one spin direction, the model may still learn that bias from raw sequences.\n"
            "- The Inspect IMU tab lets you tune the threshold, smoothing sigma and combination gap visually.\n",
        )
        notes.configure(state="disabled")

    def _populate_sessions(self) -> None:
        self.sessions_text.delete("1.0", tk.END)
        for session in operations.list_sessions():
            metadata = operations.session_metadata(session)
            self.sessions_text.insert(
                tk.END,
                f"{session}: path={metadata['path']} synchro={metadata['sample_time_fine_synchro']}\n",
            )

    def _refresh_session_selectors(self) -> None:
        values = operations.list_sessions()
        self.process_session_box.configure(values=values)
        self.inspect_session_box.configure(values=values)
        if values:
            if self.session_var.get() not in values:
                self.session_var.set(values[0])
            if self.inspect_session_var.get() not in values:
                self.inspect_session_var.set(values[0])

    def _refresh_all_sessions(self) -> None:
        self._populate_sessions()
        self._refresh_session_selectors()
        self._refresh_session_file_lists()

    def _add_session_from_gui(self) -> None:
        session_id = self.new_session_id_var.get().strip()
        path = self.new_session_path_var.get().strip()
        synchro_value = self.new_session_synchro_var.get().strip()
        if not session_id or not path or not synchro_value:
            messagebox.showwarning("Synergie Tools", "Fill session id, path and synchro before adding a session.")
            return

        try:
            metadata = operations.add_session(session_id, path, int(synchro_value))
        except ValueError as exc:
            messagebox.showerror("Synergie Tools", str(exc))
            return

        self.new_session_id_var.set("")
        self.new_session_path_var.set("")
        self.new_session_synchro_var.set("")
        self.session_var.set(session_id)
        self.inspect_session_var.set(session_id)
        self._refresh_all_sessions()
        self.status_var.set(f"Session added: {session_id} -> {metadata['path']}")

    def _refresh_session_file_lists(self) -> None:
        self._populate_process_session_files()
        self._populate_inspect_session_files()

    def _populate_process_session_files(self) -> None:
        session_name = self.session_var.get()
        directory = operations.session_directory(session_name)
        files = [str(path) for path in operations.list_directory_files(directory)]
        self.session_files_var.set(files)
        self.session_folder_summary_var.set(self._format_folder_summary(session_name, directory, files))
        self.process_selected_file_info_var.set("No file selected.")

    def _refresh_new_data_directories(self) -> None:
        directories = [item["relative_path"] for item in operations.list_new_data_directories()]
        self.new_data_directory_box.configure(values=directories)
        if directories and self.new_data_directory_var.get() not in directories:
            self.new_data_directory_var.set(directories[0])
        self._populate_new_data_files()

    def _populate_new_data_files(self) -> None:
        directory = self.new_data_directory_var.get().strip()
        sessions = operations.list_new_imu_sessions(directory=directory) if directory else operations.list_new_imu_sessions()
        self._new_data_files_cache = sessions
        self.new_data_files_var.set([self._format_new_data_file_label(item) for item in sessions])
        self.new_data_summary_var.set(f"Sessions found: {len(sessions)}")
        if not sessions:
            self.new_data_output_var.set("")

    def _format_new_data_file_label(self, metadata: dict) -> str:
        if "files" in metadata:
            sensors = ", ".join(file_metadata["sensor_id"] for file_metadata in metadata["files"])
            return (
                f"{metadata['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{len(metadata['files'])} sensors [{sensors}]"
            )
        return (
            f"sensor {metadata['sensor_id']} | {metadata['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')} | "
            f"{metadata['name']}"
        )

    def _populate_inspect_session_files(self) -> None:
        session_name = self.inspect_session_var.get()
        directory = operations.session_directory(session_name)
        files = [str(path) for path in operations.list_directory_files(directory)]
        self.inspect_session_files_var.set(files)
        self.inspect_folder_summary_var.set(self._format_folder_summary(session_name, directory, files))
        self.inspect_selected_file_info_var.set("No file selected.")

    def _format_folder_summary(self, session_name: str, directory, files: list[str]) -> str:
        csv_count = sum(1 for file_path in files if str(file_path).lower().endswith(".csv"))
        return (
            f"Session {session_name} folder: {directory}\n"
            f"Files found: {len(files)} total, {csv_count} CSV"
        )

    def _describe_selected_file(self, path: str) -> str:
        description = operations.describe_file(path)
        return (
            f"Selected file: {description['name']}\n"
            f"Type: {description['suffix'] or 'no extension'} | Size: {description['size_bytes']} bytes"
        )

    def _sync_slider_labels(self) -> None:
        self.threshold_label_var.set(f"{self.threshold_var.get():.2f}")
        self.sigma_label_var.set(f"{self.sigma_var.get():.0f}")
        self.gap_label_var.set(f"{self.gap_var.get()}")

    def _sync_train_architectures(self) -> None:
        task = self.train_task_var.get()
        options = self.TRAIN_ARCHITECTURES.get(task, [])
        self.train_architecture_box.configure(values=options)
        if self.train_architecture_var.get() not in options and options:
            self.train_architecture_var.set(options[0])

    def _format_training_dataset_stats(self, stats: dict) -> str:
        class_counts = ", ".join(f"{label}: {count}" for label, count in stats["class_counts"].items())
        mirror_text = "yes" if stats["augment_mirror"] else "no"
        return (
            f"Training set stats ({stats['task']}): "
            f"{stats['base_samples']} labelled jumps, {stats['unique_skaters']} skaters, "
            f"{stats['effective_samples']} effective samples with mirror augmentation ({mirror_text}).\n"
            f"Class distribution: {class_counts}"
        )

    def _format_training_quality_summary(self, summary: dict) -> str:
        def metric(value) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        saved_model = summary.get("saved_model") or {}
        saved_model_text = saved_model.get("id", "not archived yet")
        return (
            f"Model quality: test_acc={metric(summary.get('test_accuracy'))}, "
            f"best_val_acc={metric(summary.get('best_val_accuracy'))}, "
            f"final_val_acc={metric(summary.get('final_val_accuracy'))}, "
            f"epochs={summary.get('epochs_ran', 0)}, "
            f"test_samples={summary.get('test_samples', 0)}\n"
            f"Saved model: {saved_model_text}"
        )

    def _refresh_pretrained_models(self) -> None:
        task = self.train_task_var.get()
        models = operations.list_pretrained_training_models(task=task)
        compatible_models = [model for model in models if model.get("compatible") and model.get("path_exists")]
        labels = [operations.format_pretrained_model_label(model) for model in compatible_models]
        self.pretrained_model_box.configure(values=labels)
        if labels:
            current = self.pretrained_model_var.get()
            if current not in labels:
                self.pretrained_model_var.set(labels[0])
        else:
            self.pretrained_model_var.set("")

        if compatible_models:
            summary_lines = [
                "Compatible pretrained models:",
                *[f"- {operations.format_pretrained_model_label(model)}" for model in compatible_models],
            ]
        else:
            summary_lines = ["No compatible pretrained model found for this task."]

        incompatible_models = [model for model in models if not (model.get("compatible") and model.get("path_exists"))]
        if incompatible_models:
            summary_lines.extend(
                [f"Detected but unavailable: {model['label']} ({model.get('notes', 'not compatible')})" for model in incompatible_models]
            )
        self.pretrained_models_summary_var.set("\n".join(summary_lines))

    def _selected_pretrained_model_id(self) -> str | None:
        selected_label = self.pretrained_model_var.get()
        if not selected_label:
            return None
        for model in operations.list_pretrained_training_models(task=self.train_task_var.get(), compatible_only=True):
            if operations.format_pretrained_model_label(model) == selected_label:
                return model["id"]
        return None

    def _sync_pretrained_controls(self) -> None:
        self.pretrained_model_box.configure(state="readonly" if self.use_pretrained_var.get() else "disabled")
        self.train_architecture_box.configure(state="disabled" if self.use_pretrained_var.get() else "readonly")
        if self.use_pretrained_var.get():
            self._sync_pretrained_architecture()

    def _sync_pretrained_architecture(self) -> None:
        pretrained_model_id = self._selected_pretrained_model_id()
        if not pretrained_model_id:
            return
        for model in operations.list_pretrained_training_models(task=self.train_task_var.get(), compatible_only=True):
            if model["id"] == pretrained_model_id:
                self.train_architecture_var.set(model["architecture"])
                return

    def _draw_placeholder_training_plot(self) -> None:
        if self.train_axes is None:
            return
        loss_ax, accuracy_ax = self.train_axes
        loss_ax.clear()
        accuracy_ax.clear()
        loss_ax.set_title("Loss")
        accuracy_ax.set_title("Accuracy")
        loss_ax.set_xlabel("Epoch")
        accuracy_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        accuracy_ax.set_ylabel("Accuracy")
        loss_ax.text(0.5, 0.5, "Run a training to see the curves", ha="center", va="center", transform=loss_ax.transAxes)
        accuracy_ax.text(0.5, 0.5, "Train and validation accuracy", ha="center", va="center", transform=accuracy_ax.transAxes)
        self.train_figure.tight_layout()
        self.train_canvas.draw_idle()

    def _draw_placeholder_quality_plot(self) -> None:
        if self.quality_axes is None:
            return
        for axis, title in zip(
            self.quality_axes,
            [
                "Counts by jump type",
                "Duration distribution",
                "Max |Gyr_X| distribution",
                "Duration vs max |Gyr_X|",
            ],
        ):
            axis.clear()
            axis.set_title(title)
            axis.text(0.5, 0.5, "Run the quality scan", ha="center", va="center", transform=axis.transAxes)
        self.quality_figure.tight_layout()
        self.quality_canvas.draw_idle()

    def _draw_quality_analysis(self, analysis: dict) -> None:
        if self.quality_axes is None:
            return
        counts_ax, duration_ax, gyro_ax, scatter_ax = self.quality_axes
        for axis in self.quality_axes:
            axis.clear()

        type_summary = analysis.get("type_summary", [])
        records = analysis.get("records", [])
        suspicious_records = analysis.get("suspicious_records", [])
        if not records:
            self._draw_placeholder_quality_plot()
            return

        labels = [item["label"] for item in type_summary]
        counts = [item["count"] for item in type_summary]
        suspicious_counts = [item["suspicious_count"] for item in type_summary]
        x_positions = list(range(len(labels)))
        counts_ax.bar(x_positions, counts, color="steelblue", alpha=0.8, label="All labelled")
        counts_ax.bar(x_positions, suspicious_counts, color="tomato", alpha=0.9, label="Suspicious")
        counts_ax.set_xticks(x_positions)
        counts_ax.set_xticklabels(labels, rotation=30, ha="right")
        counts_ax.set_title("Counts by jump type")
        counts_ax.legend(fontsize=8)

        grouped_duration = [[record["duration_ms"] for record in records if record["type_label"] == label] for label in labels]
        duration_ax.boxplot(grouped_duration, tick_labels=labels, patch_artist=True)
        duration_ax.set_title("Duration distribution")
        duration_ax.tick_params(axis="x", rotation=30)
        duration_ax.set_ylabel("ms")

        grouped_gyro = [[record["max_abs_gyr_x"] for record in records if record["type_label"] == label] for label in labels]
        gyro_ax.boxplot(grouped_gyro, tick_labels=labels, patch_artist=True)
        gyro_ax.set_title("Max |Gyr_X| distribution")
        gyro_ax.tick_params(axis="x", rotation=30)
        gyro_ax.set_ylabel("deg/s")

        normal_records = [record for record in records if not record["suspicious"]]
        if normal_records:
            scatter_ax.scatter(
                [record["duration_ms"] for record in normal_records],
                [record["max_abs_gyr_x"] for record in normal_records],
                color="steelblue",
                alpha=0.55,
                label="Normal",
            )
        if suspicious_records:
            scatter_ax.scatter(
                [record["duration_ms"] for record in suspicious_records],
                [record["max_abs_gyr_x"] for record in suspicious_records],
                color="tomato",
                alpha=0.9,
                label="Suspicious",
            )
        scatter_ax.set_title("Duration vs max |Gyr_X|")
        scatter_ax.set_xlabel("Duration (ms)")
        scatter_ax.set_ylabel("Max |Gyr_X|")
        if normal_records or suspicious_records:
            scatter_ax.legend(fontsize=8)

        self.quality_figure.tight_layout()
        self.quality_canvas.draw_idle()

    def _draw_training_history(self, summary: dict) -> None:
        if self.train_axes is None:
            return
        history = summary.get("history", {})
        accuracy = history.get("accuracy", [])
        val_accuracy = history.get("val_accuracy", [])
        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])
        if not any((accuracy, val_accuracy, loss, val_loss)):
            self._draw_placeholder_training_plot()
            return

        loss_ax, accuracy_ax = self.train_axes
        loss_ax.clear()
        accuracy_ax.clear()
        loss_epochs = list(range(1, max(len(loss), len(val_loss)) + 1))
        accuracy_epochs = list(range(1, max(len(accuracy), len(val_accuracy)) + 1))

        if loss:
            loss_ax.plot(loss_epochs[: len(loss)], loss, label="train loss", color="firebrick", linewidth=1.8)
        if val_loss:
            loss_ax.plot(loss_epochs[: len(val_loss)], val_loss, label="val loss", color="tomato", linestyle="--", linewidth=1.6)
        if accuracy:
            accuracy_ax.plot(accuracy_epochs[: len(accuracy)], accuracy, label="train acc", color="navy", linewidth=1.8)
        if val_accuracy:
            accuracy_ax.plot(
                accuracy_epochs[: len(val_accuracy)],
                val_accuracy,
                label="val acc",
                color="royalblue",
                linestyle="--",
                linewidth=1.6,
            )

        loss_ax.set_title("Loss")
        accuracy_ax.set_title("Accuracy")
        loss_ax.set_xlabel("Epoch")
        accuracy_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        accuracy_ax.set_ylabel("Accuracy")
        loss_ax.legend(loc="best", fontsize=8)
        accuracy_ax.legend(loc="best", fontsize=8)
        self.train_figure.tight_layout()
        self.train_canvas.draw_idle()

    def _refresh_training_dataset_stats(self) -> None:
        try:
            stats = operations.describe_training_dataset(self.train_task_var.get(), self.dataset_var.get())
        except Exception as exc:
            self.train_dataset_stats_var.set(f"Unable to load dataset stats: {exc}")
            return
        self.train_dataset_stats_var.set(self._format_training_dataset_stats(stats))

    def _format_quality_summary(self, analysis: dict) -> str:
        return (
            f"Dataset: {analysis['dataset_path']}\n"
            f"Labelled jumps analyzed: {analysis['total_labelled_jumps']} | "
            f"Suspicious: {analysis['suspicious_count']} | "
            f"Skipped rows: {analysis['skipped_rows']}"
        )

    def _refresh_quality_suspicious_list(self) -> None:
        if not self.quality_analysis:
            self.quality_suspicious_var.set([])
            return
        records = self.quality_analysis.get("suspicious_records", [])
        labels = [
            f"{index + 1:03d} | {record['type_label']} | skater {record['skater']} | {', '.join(record['reasons'])}"
            for index, record in enumerate(records)
        ]
        self.quality_suspicious_var.set(labels)
        if labels:
            self.quality_suspicious_listbox.selection_clear(0, tk.END)
            self.quality_suspicious_listbox.selection_set(0)
            self._on_quality_suspicious_selected()
        else:
            self.quality_details_var.set("No suspicious jump selected.")

    def _selected_quality_suspicious_record(self) -> dict | None:
        if not self.quality_analysis:
            return None
        selection = self.quality_suspicious_listbox.curselection()
        if not selection:
            return None
        records = self.quality_analysis.get("suspicious_records", [])
        index = selection[0]
        if index >= len(records):
            return None
        return records[index]

    def _on_quality_suspicious_selected(self, _event=None) -> None:
        record = self._selected_quality_suspicious_record()
        if record is None:
            self.quality_details_var.set("No suspicious jump selected.")
            return
        self.quality_details_var.set(
            f"Type: {record['type_label']} | Skater: {record['skater']} | Success: {record['success']}\n"
            f"Path: {record['path']}\n"
            f"Reasons: {', '.join(record['reasons'])}\n"
            f"Duration: {record['duration_ms']:.1f} ms | Rotations: {record['rotations']:.2f} | "
            f"Max |Gyr_X|: {record['max_abs_gyr_x']:.1f} | Max |Acc_X|: {record['max_abs_acc_x']:.2f}"
        )

    def _pick_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path_var.set(path)
            self._set_default_output_path(path)

    def _pick_output(self) -> None:
        input_path = self.csv_path_var.get().strip()
        initialdir = None
        initialfile = None
        if input_path:
            suggested_path = self._suggest_output_path(input_path)
            initialdir = str(suggested_path.parent)
            initialfile = suggested_path.name
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=initialdir,
            initialfile=initialfile,
        )
        if path:
            self.output_path_var.set(path)

    def _pick_inspect_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.inspect_csv_path_var.set(path)

    def _on_process_session_changed(self, _event=None) -> None:
        self._populate_process_session_files()

    def _on_inspect_session_changed(self, _event=None) -> None:
        self._populate_inspect_session_files()

    def _on_new_data_directory_changed(self, _event=None) -> None:
        self._populate_new_data_files()

    def _on_train_task_changed(self, _event=None) -> None:
        self._sync_train_architectures()
        self._refresh_pretrained_models()
        self._sync_pretrained_controls()
        self._refresh_training_dataset_stats()

    def _on_pretrained_model_changed(self, _event=None) -> None:
        self._sync_pretrained_architecture()

    def _on_process_file_selected(self, _event=None) -> None:
        selection = self.process_session_files.curselection()
        if selection:
            selected_path = self.process_session_files.get(selection[0])
            self.csv_path_var.set(selected_path)
            self._set_default_output_path(selected_path)
            self.process_selected_file_info_var.set(self._describe_selected_file(selected_path))

    def _on_inspect_file_selected(self, _event=None) -> None:
        selection = self.inspect_session_files.curselection()
        if selection:
            selected_path = self.inspect_session_files.get(selection[0])
            self.inspect_csv_path_var.set(selected_path)
            self.inspect_selected_file_info_var.set(self._describe_selected_file(selected_path))

    def _on_new_data_file_selected(self, _event=None) -> None:
        selection = self.new_data_files.curselection()
        if not selection:
            return
        session = self._new_data_files_cache[selection[0]]
        representative_file = session["files"][0]
        self.new_data_selected_file_var.set(str(representative_file["path"]))
        self.new_data_output_var.set(str(operations.suggest_for_annotation_output_path(representative_file["path"])))
        sensors = ", ".join(file_metadata["sensor_id"] for file_metadata in session["files"])
        self.new_data_summary_var.set(
            f"Session: {session['session_key']}\n"
            f"Sensors: {sensors} | {session['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _refresh_annotation_files(self) -> None:
        files = operations.list_pending_annotation_files()
        self._annotation_files_cache = files
        self.annotation_files_var.set([path.name for path in files])
        if not files:
            self.annotation_summary_var.set("No pending annotation file found.")
            self.annotation_jump_listbox.delete(0, tk.END)
            self.annotation_dataframe = None
            self.annotation_file_path = None
            self.annotation_metadata = {}
            self.annotation_video_path_var.set("")
            self.annotation_video_info_var.set("No video loaded.")
            self.annotation_sensor_sync_var.set("No sync offset saved for current sensor.")
            self._stop_annotation_playback()
            self._release_annotation_video()
            self._draw_placeholder_annotation_video()
            self._draw_placeholder_annotation_plot()

    def _on_annotation_file_selected(self, _event=None) -> None:
        selection = self.annotation_files_listbox.curselection()
        if not selection:
            return
        file_path = self._annotation_files_cache[selection[0]]
        self._load_annotation_file(file_path)

    def _load_annotation_file(self, file_path: Path) -> None:
        import pandas as pd

        self._stop_annotation_playback()
        self.annotation_file_path = file_path
        self.annotation_metadata = operations.load_annotation_metadata(file_path)
        self.annotation_dataframe = pd.read_csv(file_path)
        self._refresh_annotation_jump_list()
        sensor_count = 0 if self.annotation_dataframe.empty else self.annotation_dataframe["sensor_id"].nunique()
        self.annotation_summary_var.set(f"{file_path.name}\nEntries: {len(self.annotation_dataframe)} | Sensors: {sensor_count}")
        video_path = self.annotation_metadata.get("video_path", "")
        self.annotation_video_path_var.set(video_path)
        if video_path and Path(video_path).exists():
            self._load_annotation_video(video_path, persist=False)
        else:
            self._stop_annotation_playback()
            self._release_annotation_video()
            self._draw_placeholder_annotation_video()
            self.annotation_video_info_var.set("No video loaded.")
        if len(self.annotation_dataframe) > 0:
            self.annotation_jump_listbox.selection_clear(0, tk.END)
            self.annotation_jump_listbox.selection_set(0)
            self._on_annotation_jump_selected()

    def _refresh_annotation_jump_list(self) -> None:
        self.annotation_jump_listbox.delete(0, tk.END)
        if self.annotation_dataframe is None:
            return
        for index, row in self.annotation_dataframe.iterrows():
            sensor_id = str(row.get("sensor_id", ""))
            offset_ms = operations.get_annotation_sensor_sync_offset(self.annotation_metadata, sensor_id)
            video_label = self._format_video_ms(operations.compute_annotation_jump_video_time_ms(row, offset_ms))
            label = (
                f"{index + 1:03d} | {video_label} | "
                f"{row.get('athlete_id', row.get('skater', 'unknown'))} | "
                f"{row.get('detection_status', 'detected_jump')}"
            )
            self.annotation_jump_listbox.insert(tk.END, label)

    def _selected_annotation_index(self) -> int | None:
        selection = self.annotation_jump_listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def _on_annotation_jump_selected(self, _event=None) -> None:
        index = self._selected_annotation_index()
        if index is None or self.annotation_dataframe is None:
            return
        row = self.annotation_dataframe.iloc[index]
        review_status = operations.annotation_review_status_from_row(row)
        type_value = int(float(row.get("type", 8)))
        type_key = next((key for key, _label, value in operations.ANNOTATION_JUMP_TYPE_OPTIONS if value == type_value), "toe_loop")
        self.annotation_type_var.set(type_key)
        self.annotation_turn_var.set(operations.annotation_turn_value_for_ui(type_key, row.get("turns", "")))
        self.annotation_success_var.set(str(int(float(row.get("success", 2)))))
        self.annotation_review_status_var.set(review_status)
        self.annotation_athlete_var.set(str(row.get("athlete_id", row.get("skater", ""))))
        self._sync_annotation_turn_options()
        self._sync_annotation_review_controls()
        self._refresh_annotation_video_context()
        self._draw_annotation_segment(row)

    def _sync_annotation_turn_options(self) -> None:
        options = operations.annotation_turn_options(self.annotation_type_var.get())
        if self.annotation_turn_var.get() not in options:
            self.annotation_turn_var.set(options[0] if options else "")
        self._sync_annotation_review_controls()

    def _sync_annotation_review_controls(self) -> None:
        excluded = self.annotation_review_status_var.get() in {"not_seen_on_video", "not_a_jump"}
        hint = ""
        if self.annotation_review_status_var.get() == "not_seen_on_video":
            hint = "This jump will be excluded from training because it is marked as unseen on video."
        elif self.annotation_review_status_var.get() == "not_a_jump":
            hint = "This jump will be excluded from training because it is marked as not a jump."
        self.annotation_exclusion_hint_var.set(hint)

        desired_state = "disabled" if excluded else "normal"
        for button in self.annotation_type_buttons:
            button.configure(state=desired_state)
        for button in self.annotation_turn_buttons:
            button.configure(state=desired_state)
        for button in self.annotation_success_buttons:
            button.configure(state=desired_state)

    def _annotate_tab_active(self) -> bool:
        try:
            return self.notebook.tab(self.notebook.select(), "text") == "Annotate"
        except Exception:
            return False

    def _on_global_keypress(self, event) -> None:
        if not self._annotate_tab_active():
            return
        if self.annotation_dataframe is None or self._selected_annotation_index() is None:
            return

        widget = event.widget
        if isinstance(widget, (tk.Entry, tk.Text, scrolledtext.ScrolledText)):
            return

        key = (event.keysym or event.char or "").lower()
        if not key:
            return

        if key in self.ANNOTATION_SHORTCUTS:
            selected_type = self.ANNOTATION_SHORTCUTS[key]
            self.annotation_type_var.set(selected_type)
            self._sync_annotation_turn_options()
            self.status_var.set(f"Annotation jump type selected: {selected_type}")
            return

        if key == "u":
            self.annotation_review_status_var.set("not_seen_on_video")
            self.status_var.set("Annotation review status selected: unseen on video")
            return

        if key == "x":
            self.annotation_review_status_var.set("not_a_jump")
            self.status_var.set("Annotation review status selected: not a jump")
            return

        if key in {"1", "2", "3", "4"}:
            options = operations.annotation_turn_options(self.annotation_type_var.get())
            if key in options:
                self.annotation_turn_var.set(key)
                self.status_var.set(f"Annotation turns selected: {key}")
                return

        if key in {"0", "1"}:
            self.annotation_success_var.set("0" if key == "0" else "1")
            self.status_var.set("Annotation success selected")

    def _draw_placeholder_annotation_plot(self) -> None:
        if self.annotation_ax is None:
            return
        self.annotation_ax.clear()
        if self.annotation_acc_ax is not None:
            self.annotation_acc_ax.remove()
            self.annotation_acc_ax = None
        self.annotation_ax.set_title("Annotation signals")
        self.annotation_ax.set_xlabel("ms")
        self.annotation_ax.set_ylabel("Gyroscope")
        self.annotation_ax.text(0.5, 0.5, "Select an annotation candidate", ha="center", va="center", transform=self.annotation_ax.transAxes)
        self.annotation_figure.tight_layout()
        self.annotation_canvas.draw_idle()

    def _draw_placeholder_annotation_video(self, message: str = "Load a session video to review jumps.") -> None:
        self._annotation_video_photo = None
        self.annotation_video_label.configure(image="", text=message)

    def _format_video_ms(self, milliseconds: float) -> str:
        total_ms = max(0, int(round(float(milliseconds))))
        minutes, remaining_ms = divmod(total_ms, 60000)
        seconds, millis = divmod(remaining_ms, 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

    def _selected_annotation_row(self):
        index = self._selected_annotation_index()
        if index is None or self.annotation_dataframe is None:
            return None
        return self.annotation_dataframe.iloc[index]

    def _current_annotation_sensor_id(self) -> str | None:
        row = self._selected_annotation_row()
        if row is None:
            return None
        sensor_id = row.get("sensor_id", "")
        if sensor_id != sensor_id:
            return None
        return str(sensor_id)

    def _refresh_annotation_video_context(self) -> None:
        row = self._selected_annotation_row()
        if row is None:
            self.annotation_sensor_sync_var.set("No sync offset saved for current sensor.")
            return
        sensor_id = self._current_annotation_sensor_id()
        if sensor_id is None:
            self.annotation_sensor_sync_var.set("No sensor ID available for this entry.")
            return
        offset_ms = operations.get_annotation_sensor_sync_offset(self.annotation_metadata, sensor_id)
        jump_video_ms = operations.compute_annotation_jump_video_time_ms(row, offset_ms)
        self.annotation_sensor_sync_var.set(
            f"Sensor {sensor_id} sync offset: {offset_ms:+.0f} ms | "
            f"jump at {self._format_video_ms(jump_video_ms)} in video"
        )

    def _browse_annotation_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select session video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.annotation_video_path_var.set(file_path)
        self._load_annotation_video(file_path, persist=True)

    def _load_annotation_video_from_entry(self) -> None:
        video_path = self.annotation_video_path_var.get().strip()
        if not video_path:
            messagebox.showwarning("Synergie Tools", "Select a video file first.")
            return
        self._load_annotation_video(video_path, persist=True)

    def _release_annotation_video(self) -> None:
        if self.annotation_video_capture is not None:
            self.annotation_video_capture.release()
            self.annotation_video_capture = None
        self.annotation_video_fps = 0.0
        self.annotation_video_frame_count = 0
        self.annotation_video_duration_ms = 0.0
        self.annotation_video_current_ms = 0.0

    def _load_annotation_video(self, video_path: str | Path, persist: bool = False) -> None:
        if not self._annotation_video_supported():
            messagebox.showwarning(
                "Synergie Tools",
                "Video review needs 'opencv-python-headless' and 'Pillow' in the synergie-data environment.",
            )
            return

        import cv2

        path = Path(video_path)
        if not path.exists():
            messagebox.showerror("Synergie Tools", f"Video file not found:\n{path}")
            return

        self._stop_annotation_playback()
        self._release_annotation_video()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            messagebox.showerror("Synergie Tools", f"Unable to open video:\n{path}")
            return

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_ms = 0.0
        if fps > 0 and frame_count > 0:
            duration_ms = frame_count / fps * 1000.0

        self.annotation_video_capture = capture
        self.annotation_video_fps = fps
        self.annotation_video_frame_count = frame_count
        self.annotation_video_duration_ms = duration_ms
        self.annotation_video_path_var.set(str(path))
        self.annotation_video_slider.configure(to=max(duration_ms, 1.0))
        self.annotation_video_info_var.set(f"{path.name} | fps={fps:.2f}")
        if persist and self.annotation_file_path is not None:
            self.annotation_metadata = operations.set_annotation_video_path(self.annotation_file_path, path)
        self._display_annotation_video_frame(0.0)
        self._refresh_annotation_video_context()

    def _display_annotation_video_frame(self, milliseconds: float) -> None:
        if self.annotation_video_capture is None:
            self._draw_placeholder_annotation_video()
            return

        import cv2
        from PIL import Image, ImageTk

        target_ms = max(0.0, min(float(milliseconds), self.annotation_video_duration_ms or float(milliseconds)))
        if self.annotation_video_fps > 0:
            frame_index = int(round((target_ms / 1000.0) * self.annotation_video_fps))
            max_frame_index = max(self.annotation_video_frame_count - 1, 0)
            frame_index = min(frame_index, max_frame_index)
            self.annotation_video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            self.annotation_video_capture.set(cv2.CAP_PROP_POS_MSEC, target_ms)

        ok, frame = self.annotation_video_capture.read()
        if not ok:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        image.thumbnail((420, 260))
        photo = ImageTk.PhotoImage(image)
        self._annotation_video_photo = photo
        self.annotation_video_label.configure(image=photo, text="")
        self.annotation_video_current_ms = target_ms
        self.annotation_video_slider_var.set(target_ms)
        self.annotation_video_time_var.set(self._format_video_ms(target_ms))
        duration_text = self._format_video_ms(self.annotation_video_duration_ms) if self.annotation_video_duration_ms else "unknown"
        self.annotation_video_info_var.set(
            f"{Path(self.annotation_video_path_var.get()).name} | "
            f"{self.annotation_video_frame_count} frames | "
            f"{duration_text}"
        )

    def _on_annotation_video_slider_released(self, _event=None) -> None:
        if self.annotation_video_capture is None:
            return
        self._stop_annotation_playback()
        self._display_annotation_video_frame(self.annotation_video_slider_var.get())

    def _seek_annotation_video_relative(self, delta_ms: float) -> None:
        if self.annotation_video_capture is None:
            return
        self._stop_annotation_playback()
        self._display_annotation_video_frame(self.annotation_video_current_ms + delta_ms)

    def _current_jump_video_time_ms(self) -> float | None:
        row = self._selected_annotation_row()
        sensor_id = self._current_annotation_sensor_id()
        if row is None or sensor_id is None:
            return None
        offset_ms = operations.get_annotation_sensor_sync_offset(self.annotation_metadata, sensor_id)
        return operations.compute_annotation_jump_video_time_ms(row, offset_ms)

    def _seek_annotation_video_to_current_jump(self) -> None:
        if self.annotation_video_capture is None:
            messagebox.showwarning("Synergie Tools", "Load the session video first.")
            return
        target_ms = self._current_jump_video_time_ms()
        if target_ms is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation entry first.")
            return
        self._stop_annotation_playback()
        self._display_annotation_video_frame(target_ms)

    def _play_annotation_to_current_jump(self) -> None:
        if self.annotation_video_capture is None:
            messagebox.showwarning("Synergie Tools", "Load the session video first.")
            return
        target_ms = self._current_jump_video_time_ms()
        if target_ms is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation entry first.")
            return

        self._stop_annotation_playback()
        start_ms = max(target_ms - 5000.0, 0.0)
        base_frame_ms = 1000.0 / self.annotation_video_fps if self.annotation_video_fps > 0 else 40.0
        step_ms = max(base_frame_ms * 5.0, 100.0)
        delay_ms = max(int(round(base_frame_ms)), 20)
        self._display_annotation_video_frame(start_ms)

        def advance() -> None:
            if self.annotation_video_capture is None:
                self._annotation_playback_after_id = None
                return
            next_ms = self.annotation_video_current_ms + step_ms
            if next_ms >= target_ms:
                self._display_annotation_video_frame(target_ms)
                self._annotation_playback_after_id = None
                return
            self._display_annotation_video_frame(next_ms)
            self._annotation_playback_after_id = self.root.after(delay_ms, advance)

        self._annotation_playback_after_id = self.root.after(delay_ms, advance)

    def _stop_annotation_playback(self) -> None:
        if self._annotation_playback_after_id is not None:
            self.root.after_cancel(self._annotation_playback_after_id)
            self._annotation_playback_after_id = None

    def _sync_current_sensor_to_video(self) -> None:
        if self.annotation_file_path is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation file first.")
            return
        if self.annotation_video_capture is None:
            messagebox.showwarning("Synergie Tools", "Load the session video first.")
            return
        row = self._selected_annotation_row()
        sensor_id = self._current_annotation_sensor_id()
        if row is None or sensor_id is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation entry first.")
            return

        base_ms = operations.compute_annotation_jump_video_time_ms(row, 0.0)
        offset_ms = self.annotation_video_current_ms - base_ms
        self.annotation_metadata = operations.set_annotation_sensor_sync_offset(self.annotation_file_path, sensor_id, offset_ms)
        self._refresh_annotation_jump_list()
        index = self._selected_annotation_index()
        if index is not None:
            self.annotation_jump_listbox.selection_clear(0, tk.END)
            self.annotation_jump_listbox.selection_set(index)
        self._refresh_annotation_video_context()
        self.status_var.set(f"Saved video sync for sensor {sensor_id}")

    def _clear_current_sensor_sync(self) -> None:
        if self.annotation_file_path is None:
            return
        sensor_id = self._current_annotation_sensor_id()
        if sensor_id is None:
            return
        offsets = dict(self.annotation_metadata.get("sensor_sync_offsets_ms", {}))
        if sensor_id in offsets:
            offsets.pop(sensor_id)
            self.annotation_metadata["sensor_sync_offsets_ms"] = offsets
            operations.save_annotation_metadata(self.annotation_file_path, self.annotation_metadata)
        self._refresh_annotation_jump_list()
        index = self._selected_annotation_index()
        if index is not None:
            self.annotation_jump_listbox.selection_clear(0, tk.END)
            self.annotation_jump_listbox.selection_set(index)
        self._refresh_annotation_video_context()
        self.status_var.set(f"Cleared video sync for sensor {sensor_id}")

    def _draw_annotation_segment(self, row) -> None:
        import pandas as pd

        if self.annotation_ax is None:
            return
        path = Path(str(row["path"]))
        if not path.exists():
            self._draw_placeholder_annotation_plot()
            return
        dataframe = pd.read_csv(path)
        sensor_id = str(row.get("sensor_id", ""))
        offset_ms = operations.get_annotation_sensor_sync_offset(self.annotation_metadata, sensor_id)
        video_time_label = self._format_video_ms(operations.compute_annotation_jump_video_time_ms(row, offset_ms))
        self.annotation_ax.clear()
        if self.annotation_acc_ax is not None:
            self.annotation_acc_ax.remove()
            self.annotation_acc_ax = None
        self.annotation_acc_ax = self.annotation_ax.twinx()
        gyro_line = self.annotation_ax.plot(dataframe["ms"], dataframe["Gyr_X"], label="Gyr_X", linewidth=1.2, color="#1f77b4")[0]
        acc_line = self.annotation_acc_ax.plot(dataframe["ms"], dataframe["Acc_X"], label="Acc_X", linewidth=0.9, alpha=0.8, color="#ff7f0e")[0]
        self.annotation_ax.set_title(
            f"{video_time_label} | "
            f"{row.get('athlete_id', row.get('skater', 'unknown'))} | "
            f"sensor {sensor_id} | {row.get('source_file', '')}"
        )
        self.annotation_ax.set_xlabel("ms")
        self.annotation_ax.set_ylabel("Gyroscope")
        self.annotation_acc_ax.set_ylabel("Acceleration")
        self.annotation_ax.legend([gyro_line, acc_line], ["Gyr_X", "Acc_X"], loc="upper right", fontsize=8)
        self.annotation_figure.tight_layout()
        self.annotation_canvas.draw_idle()

    def _save_current_annotation(self) -> None:
        index = self._selected_annotation_index()
        if index is None or self.annotation_dataframe is None or self.annotation_file_path is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation entry first.")
            return

        type_value = next(
            value for key, _label, value in operations.ANNOTATION_JUMP_TYPE_OPTIONS
            if key == self.annotation_type_var.get()
        )
        backend_status = operations.annotation_review_status_to_backend(self.annotation_review_status_var.get())
        is_excluded_by_status = self.annotation_review_status_var.get() in {"not_seen_on_video", "not_a_jump"}
        stored_type = 8 if is_excluded_by_status else type_value
        stored_turns = "" if is_excluded_by_status else operations.annotation_turn_value_for_storage(
            self.annotation_type_var.get(),
            self.annotation_turn_var.get(),
        )
        stored_success = 2 if is_excluded_by_status else int(self.annotation_success_var.get())
        self.annotation_dataframe.at[index, "type"] = stored_type
        self.annotation_dataframe.at[index, "turns"] = stored_turns
        self.annotation_dataframe.at[index, "success"] = stored_success
        self.annotation_dataframe.at[index, "video_status"] = backend_status["video_status"]
        self.annotation_dataframe.at[index, "detection_status"] = backend_status["detection_status"]
        self.annotation_dataframe.at[index, "athlete_id"] = self.annotation_athlete_var.get()
        self.annotation_dataframe.to_csv(self.annotation_file_path, index=False)
        self._refresh_annotation_jump_list()
        self.annotation_jump_listbox.selection_clear(0, tk.END)
        self.annotation_jump_listbox.selection_set(index)
        self.status_var.set("Annotation saved")

    def _on_inspect_file_double_clicked(self, _event=None) -> None:
        self._on_inspect_file_selected()
        self._run_inspection()

    def _suggest_output_path(self, input_path: str) -> str:
        return str(operations.next_jumplist_output_path(Path(input_path).parent))

    def _set_default_output_path(self, input_path: str) -> None:
        self.output_path_var.set(self._suggest_output_path(input_path))

    def _log(self, widget: scrolledtext.ScrolledText, message: str) -> None:
        widget.insert(tk.END, message + "\n")
        widget.see(tk.END)

    def _run_in_thread(self, target, on_error_message: str) -> None:
        def runner() -> None:
            try:
                target()
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Synergie Tools", f"{on_error_message}\n\n{exc}"))
                self.root.after(0, lambda: self.status_var.set("Error"))

        threading.Thread(target=runner, daemon=True).start()

    def _run_process_file(self) -> None:
        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("Synergie Tools", "Select an input CSV first.")
            return

        session = operations.session_metadata(self.session_var.get())
        output_path = self.output_path_var.get().strip() or None
        self.status_var.set("Processing file...")
        self.process_log.delete("1.0", tk.END)

        def action() -> None:
            destination = operations.process_csv_file(
                csv_path,
                synchro=session["sample_time_fine_synchro"],
                output_path=output_path,
            )
            self.root.after(0, lambda: self._log(self.process_log, f"Created: {destination}"))
            self.root.after(0, lambda: self.status_var.set("Processing completed"))

        self._run_in_thread(action, "Unable to process file.")

    def _run_train(self) -> None:
        self.status_var.set("Training started...")
        self.train_log.delete("1.0", tk.END)
        self.train_quality_summary_var.set("Training in progress...")
        self._draw_placeholder_training_plot()

        def action() -> None:
            epochs = int(self.epochs_var.get())
            task = self.train_task_var.get()
            pretrained_model_id = self._selected_pretrained_model_id() if self.use_pretrained_var.get() else None
            architecture = None if pretrained_model_id else self.train_architecture_var.get()
            pretrained_text = f", pretrained={pretrained_model_id}" if pretrained_model_id else ""
            self.root.after(
                0,
                lambda: self._log(
                    self.train_log,
                    f"Running training: task={task}, architecture={self.train_architecture_var.get()}, epochs={epochs}{pretrained_text}",
                ),
            )
            summary = operations.train_model(task, self.dataset_var.get(), epochs, architecture, pretrained_model_id=pretrained_model_id)
            formatted_summary = self._format_training_quality_summary(summary)
            self.root.after(0, lambda: self._draw_training_history(summary))
            self.root.after(0, lambda: self.train_quality_summary_var.set(formatted_summary))
            self.root.after(0, lambda: self._log(self.train_log, formatted_summary))
            self.root.after(0, lambda: self._log(self.train_log, f"Confusion matrix: {summary.get('confusion_matrix', [])}"))
            self.root.after(0, lambda: self._log(self.train_log, f"Saved model path: {summary.get('saved_model', {}).get('path', 'n/a')}"))
            self.root.after(0, lambda: self._log(self.train_log, f"Latest model alias: {summary.get('latest_model_path', 'n/a')}"))
            self.root.after(0, self._refresh_pretrained_models)
            self.root.after(0, lambda: self._log(self.train_log, "Training finished."))
            self.root.after(0, lambda: self.status_var.set("Training completed"))

        self._run_in_thread(action, "Unable to run training.")

    def _run_process_new_data_file(self) -> None:
        raw_file = self.new_data_selected_file_var.get().strip()
        if not raw_file:
            messagebox.showwarning("Synergie Tools", "Select an IMU file from New Data first.")
            return

        self.status_var.set("Processing new IMU file for annotation...")
        self.new_data_log.delete("1.0", tk.END)

        def action() -> None:
            result = operations.process_new_imu_file_for_annotation(
                raw_file,
                output_path=self.new_data_output_var.get().strip() or None,
            )
            self.root.after(0, lambda: self._log(self.new_data_log, f"Created annotation CSV: {result['annotation_csv']}"))
            self.root.after(0, lambda: self._log(self.new_data_log, f"Created jump segments in: {result['segment_directory']}"))
            self.root.after(0, lambda: self._log(self.new_data_log, f"Jumps ready for annotation: {result['jump_count']} across {result['sensor_count']} sensors"))
            self.root.after(0, self._refresh_annotation_files)
            self.root.after(0, lambda: self.status_var.set("New IMU file processed for annotation"))

        self._run_in_thread(action, "Unable to process the new IMU file for annotation.")

    def _run_inspection(self) -> None:
        csv_path = self.inspect_csv_path_var.get().strip()
        if not csv_path:
            messagebox.showwarning("Synergie Tools", "Select an input CSV for inspection first.")
            return

        self.status_var.set("Loading IMU file and detecting jumps...")

        def action() -> None:
            import pandas as pd
            from core.data_treatment.data_generation.trainingSession import trainingSession

            dataframe = pd.read_csv(csv_path)
            session_metadata = operations.session_metadata(self.inspect_session_var.get())
            session = trainingSession(
                dataframe,
                sampleTimefineSynchro=session_metadata["sample_time_fine_synchro"],
                detection_threshold=self.threshold_var.get(),
                smoothing_sigma=self.sigma_var.get(),
                combination_gap_frames=self.gap_var.get(),
            )
            self.root.after(0, lambda: self._apply_inspection_results(dataframe, session))

        self._run_in_thread(action, "Unable to inspect IMU data.")

    def _apply_inspection_results(self, dataframe, session) -> None:
        self.inspect_dataframe = dataframe
        self.inspect_session = session
        self.detected_jumps = session.jumps
        self._refresh_jump_list()
        self._redraw_plots()
        self.status_var.set(f"Inspection ready: {len(self.detected_jumps)} jumps detected")

    def _refresh_jump_list(self) -> None:
        self.jump_listbox.delete(0, tk.END)
        for index, jump in enumerate(self.detected_jumps, start=1):
            label = (
                f"Jump {index} | start={jump.startTimestamp:.0f} ms | "
                f"len={jump.length:.2f} s | rot={jump.signed_rotation:.2f}"
            )
            self.jump_listbox.insert(tk.END, label)

    def _draw_placeholder_plots(self) -> None:
        overview_ax, zoom_ax = self.axes
        self._reset_secondary_axes()
        overview_ax.clear()
        zoom_ax.clear()
        overview_ax.set_title("Overview of IMU jump localisation signals")
        zoom_ax.set_title("Zoom on selected jump")
        overview_ax.set_xlabel("ms")
        zoom_ax.set_xlabel("ms")
        overview_ax.set_ylabel("Gyroscope")
        zoom_ax.set_ylabel("Gyroscope")
        overview_ax.text(0.5, 0.5, "Load a CSV in Inspect IMU", ha="center", va="center", transform=overview_ax.transAxes)
        zoom_ax.text(0.5, 0.5, "Select a detected jump to zoom", ha="center", va="center", transform=zoom_ax.transAxes)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _jump_window_bounds_ms(self, session_df, jump) -> dict[str, tuple[float, float]]:
        type_start_idx = max(0, jump.start - SUCCESS_WINDOW_START)
        type_end_idx = min(len(session_df) - 1, jump.start + (TYPE_WINDOW_FRAMES - SUCCESS_WINDOW_START) - 1)
        success_start_idx = max(0, jump.start)
        success_end_idx = min(len(session_df) - 1, jump.start + (len(jump.df_success) - 1))
        return {
            "type": (float(session_df.iloc[type_start_idx]["ms"]), float(session_df.iloc[type_end_idx]["ms"])),
            "success": (float(session_df.iloc[success_start_idx]["ms"]), float(session_df.iloc[success_end_idx]["ms"])),
            "detected": (float(session_df.iloc[jump.start]["ms"]), float(session_df.iloc[jump.end]["ms"])),
        }

    def _jump_center_ms(self, session_df, jump) -> float:
        bounds = self._jump_window_bounds_ms(session_df, jump)
        return (bounds["detected"][0] + bounds["detected"][1]) / 2.0

    def _jump_has_gyro_saturation(self, session_df, jump) -> bool:
        start_idx = max(0, jump.start - SUCCESS_WINDOW_START)
        end_idx = min(len(session_df), jump.start + len(jump.df))
        window = session_df.iloc[start_idx:end_idx]["Gyr_X_unfiltered"].abs()
        return bool((window >= GYRO_SATURATION_WARNING_THRESHOLD).any())

    def _draw_jump_windows(self, axis, session_df, jump, alpha_scale: float = 1.0) -> None:
        bounds = self._jump_window_bounds_ms(session_df, jump)
        axis.axvspan(bounds["type"][0], bounds["type"][1], color="royalblue", alpha=0.10 * alpha_scale)
        axis.axvspan(bounds["success"][0], bounds["success"][1], color="seagreen", alpha=0.10 * alpha_scale)
        if self._jump_has_gyro_saturation(session_df, jump):
            axis.axvspan(bounds["detected"][0], bounds["detected"][1], color="crimson", alpha=0.12 * alpha_scale)
            axis.axvline(bounds["detected"][0], color="crimson", linestyle=":", linewidth=1.2)
            axis.axvline(bounds["detected"][1], color="crimson", linestyle=":", linewidth=1.2)

    def _redraw_plots(self) -> None:
        if self.inspect_session is None:
            self._draw_placeholder_plots()
            return

        selected_index = self._selected_jump_index()
        self._draw_overview_plot()
        self._draw_zoom_plot(selected_index)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_overview_plot(self) -> None:
        overview_ax, _ = self.axes
        session_df = self.inspect_session.df
        selected_index = self._selected_jump_index()
        self._reset_secondary_axes()
        overview_ax.clear()
        overview_right_ax = overview_ax.twinx()
        overview_ax.set_title("Signals used to localise jumps")
        overview_ax.plot(session_df["ms"], session_df["Gyr_X_unfiltered"], label="Gyr_X raw", linewidth=0.8, alpha=0.4)
        overview_ax.plot(session_df["ms"], session_df["Gyr_X_smoothed"], label="Gyr_X smoothed", linewidth=1.3)
        overview_right_ax.plot(
            session_df["ms"],
            session_df["X_gyr_second_derivative"],
            label="2nd derivative",
            linewidth=1.0,
            color="crimson",
        )
        overview_right_ax.axhline(
            self.inspect_session.detection_threshold,
            color="crimson",
            linestyle="--",
            label="Detection threshold",
        )

        selected_center_ms = None
        selected_center_y = None
        for index, jump in enumerate(self.detected_jumps):
            self._draw_jump_windows(overview_ax, session_df, jump)
            center_ms = self._jump_center_ms(session_df, jump)
            center_y = float(session_df.iloc[jump.start + ((jump.end - jump.start) // 2)]["Gyr_X_smoothed"])
            overview_ax.plot(
                center_ms,
                center_y,
                marker="o",
                markersize=6,
                color="black",
                markerfacecolor="white",
                markeredgewidth=1.2,
                linestyle="None",
            )
            if selected_index == index:
                selected_center_ms = center_ms
                selected_center_y = center_y

        if selected_center_ms is not None and selected_center_y is not None:
            overview_ax.plot(
                selected_center_ms,
                selected_center_y,
                marker="*",
                markersize=14,
                color="gold",
                markeredgecolor="black",
                markeredgewidth=1.0,
                linestyle="None",
            )

        overview_ax.set_xlabel("ms")
        overview_ax.set_ylabel("Gyroscope")
        overview_right_ax.set_ylabel("2nd derivative")
        overview_ax.plot([], [], color="royalblue", linewidth=6, alpha=0.35, label="Type window")
        overview_ax.plot([], [], color="seagreen", linewidth=6, alpha=0.35, label="Success window")
        overview_ax.plot([], [], color="crimson", linewidth=2, linestyle=":", label="Gyro saturation")
        overview_ax.plot([], [], marker="o", markersize=6, color="black", markerfacecolor="white", linestyle="None", label="Detected jump center")
        if selected_center_ms is not None:
            overview_ax.plot([], [], marker="*", markersize=12, color="gold", markeredgecolor="black", linestyle="None", label="Selected jump")
        overview_lines, overview_labels = overview_ax.get_legend_handles_labels()
        overview_right_lines, overview_right_labels = overview_right_ax.get_legend_handles_labels()
        overview_ax.legend(overview_lines + overview_right_lines, overview_labels + overview_right_labels, loc="upper right", fontsize=8)

    def _draw_zoom_plot(self, jump_index: int | None) -> None:
        _, zoom_ax = self.axes
        zoom_ax.clear()
        zoom_right_ax = zoom_ax.twinx()
        zoom_ax.set_title("Zoom on selected jump")

        if jump_index is None or jump_index >= len(self.detected_jumps):
            zoom_ax.text(0.5, 0.5, "Select a jump in the list", ha="center", va="center", transform=zoom_ax.transAxes)
            zoom_ax.set_xlabel("ms")
            zoom_ax.set_ylabel("Gyroscope")
            zoom_right_ax.set_ylabel("2nd derivative")
            return

        jump = self.detected_jumps[jump_index]
        session_df = self.inspect_session.df
        start_idx = max(0, jump.start - 140)
        end_idx = min(len(session_df) - 1, jump.end + 140)
        view_df = session_df.iloc[start_idx : end_idx + 1]

        zoom_ax.plot(view_df["ms"], view_df["Gyr_X_unfiltered"], label="Gyr_X raw", linewidth=0.8, alpha=0.4)
        zoom_ax.plot(view_df["ms"], view_df["Gyr_X_smoothed"], label="Gyr_X smoothed", linewidth=1.3)
        zoom_right_ax.plot(
            view_df["ms"],
            view_df["X_gyr_second_derivative"],
            label="2nd derivative",
            linewidth=1.0,
            color="crimson",
        )
        zoom_right_ax.axhline(
            self.inspect_session.detection_threshold,
            color="crimson",
            linestyle="--",
            label="Detection threshold",
        )

        self._draw_jump_windows(zoom_ax, session_df, jump, alpha_scale=1.5)
        bounds = self._jump_window_bounds_ms(session_df, jump)
        zoom_ax.axvline(bounds["detected"][0], color="black", linestyle=":")
        zoom_ax.axvline(bounds["detected"][1], color="black", linestyle=":")
        zoom_ax.set_xlabel("ms")
        zoom_ax.set_ylabel("Gyroscope")
        zoom_right_ax.set_ylabel("2nd derivative")
        saturation_text = " | Gyro saturated" if self._jump_has_gyro_saturation(session_df, jump) else ""
        zoom_ax.set_title(f"Zoom on selected jump #{jump_index + 1}{saturation_text}")
        zoom_ax.plot([], [], color="royalblue", linewidth=6, alpha=0.35, label="Type window")
        zoom_ax.plot([], [], color="seagreen", linewidth=6, alpha=0.35, label="Success window")
        zoom_ax.plot([], [], color="crimson", linewidth=2, linestyle=":", label="Gyro saturation")
        zoom_lines, zoom_labels = zoom_ax.get_legend_handles_labels()
        zoom_right_lines, zoom_right_labels = zoom_right_ax.get_legend_handles_labels()
        zoom_ax.legend(zoom_lines + zoom_right_lines, zoom_labels + zoom_right_labels, loc="upper right", fontsize=8)

    def _reset_secondary_axes(self) -> None:
        primary_axes = set(self.axes)
        for axis in list(self.figure.axes):
            if axis not in primary_axes:
                self.figure.delaxes(axis)

    def _selected_jump_index(self) -> int | None:
        selection = self.jump_listbox.curselection()
        if not selection:
            return None
        return selection[0]

    def _on_jump_selected(self, _event) -> None:
        selected_index = self._selected_jump_index()
        if selected_index is not None and selected_index < len(self.detected_jumps):
            jump = self.detected_jumps[selected_index]
            self.status_var.set(
                f"Selected jump {selected_index + 1}/{len(self.detected_jumps)} | "
                f"start={jump.startTimestamp:.0f} ms | len={jump.length:.2f} s"
            )
        self._redraw_plots()


def launch() -> None:
    root = tk.Tk()
    SynergieToolsApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
