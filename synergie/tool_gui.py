from __future__ import annotations

import threading
import tkinter as tk
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
        self.figure = None
        self.axes = None
        self.canvas = None
        self.train_figure = None
        self.train_axes = None
        self.train_canvas = None

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

        sessions_tab = ttk.Frame(notebook, padding=12)
        process_tab = ttk.Frame(notebook, padding=12)
        inspect_tab = ttk.Frame(notebook, padding=12)
        train_tab = ttk.Frame(notebook, padding=12)
        notes_tab = ttk.Frame(notebook, padding=12)
        notebook.add(sessions_tab, text="Sessions")
        notebook.add(process_tab, text="Process CSV")
        notebook.add(inspect_tab, text="Inspect IMU")
        notebook.add(train_tab, text="Train")
        notebook.add(notes_tab, text="Algo Notes")

        sessions_tab.columnconfigure(0, weight=1)
        sessions_tab.columnconfigure(1, weight=0)
        sessions_tab.rowconfigure(0, weight=1)

        self.sessions_text = scrolledtext.ScrolledText(sessions_tab, height=18, wrap=tk.WORD)
        self.sessions_text.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self._build_sessions_side_panel(sessions_tab)
        ttk.Button(sessions_tab, text="Refresh sessions", command=self._refresh_all_sessions).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._build_process_tab(process_tab)
        self._build_inspect_tab(inspect_tab)
        self._build_train_tab(train_tab)
        self._build_notes_tab(notes_tab)

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

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
        for index in range(2):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)
        parent.rowconfigure(8, weight=1)
        parent.rowconfigure(9, weight=1)

        ttk.Label(parent, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        train_task_box = ttk.Combobox(parent, textvariable=self.train_task_var, values=["type", "success"], state="readonly")
        train_task_box.grid(row=0, column=1, sticky="w")
        train_task_box.bind("<<ComboboxSelected>>", self._on_train_task_changed)

        ttk.Label(parent, text="Architecture").grid(row=1, column=0, sticky="w", pady=4)
        self.train_architecture_box = ttk.Combobox(parent, textvariable=self.train_architecture_var, state="readonly")
        self.train_architecture_box.grid(row=1, column=1, sticky="w")
        self._sync_train_architectures()

        ttk.Label(parent, text="Dataset").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.dataset_var).grid(row=2, column=1, sticky="ew")

        ttk.Label(parent, text="Epochs").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.epochs_var).grid(row=3, column=1, sticky="w")

        ttk.Checkbutton(
            parent,
            text="Start from pretrained model",
            variable=self.use_pretrained_var,
            command=self._sync_pretrained_controls,
        ).grid(row=4, column=0, sticky="w", pady=(12, 4))

        self.pretrained_model_box = ttk.Combobox(parent, textvariable=self.pretrained_model_var, state="readonly")
        self.pretrained_model_box.grid(row=4, column=1, sticky="ew", pady=(12, 4))
        self.pretrained_model_box.bind("<<ComboboxSelected>>", self._on_pretrained_model_changed)

        ttk.Label(parent, textvariable=self.pretrained_models_summary_var, justify=tk.LEFT).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Button(parent, text="Refresh dataset stats", command=self._refresh_training_dataset_stats).grid(row=6, column=0, sticky="w", pady=(4, 4))
        ttk.Label(parent, textvariable=self.train_dataset_stats_var, justify=tk.LEFT).grid(row=7, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Button(parent, text="Run training", command=self._run_train).grid(row=8, column=0, sticky="w", pady=(8, 4))
        ttk.Label(parent, textvariable=self.train_quality_summary_var, justify=tk.LEFT).grid(row=9, column=0, columnspan=2, sticky="w", pady=(0, 8))

        train_plot_frame = ttk.LabelFrame(parent, text="Training Curves", padding=8)
        train_plot_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", pady=(0, 8))
        train_plot_frame.columnconfigure(0, weight=1)
        train_plot_frame.rowconfigure(0, weight=1)
        self.train_plot_container = ttk.Frame(train_plot_frame)
        self.train_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_train_plot_canvas()

        self.train_log = scrolledtext.ScrolledText(parent, height=18, wrap=tk.WORD)
        self.train_log.grid(row=11, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(10, weight=1)
        parent.rowconfigure(11, weight=1)
        self._refresh_pretrained_models()
        self._sync_pretrained_controls()
        self._refresh_training_dataset_stats()

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

        return (
            f"Model quality: test_acc={metric(summary.get('test_accuracy'))}, "
            f"best_val_acc={metric(summary.get('best_val_accuracy'))}, "
            f"final_val_acc={metric(summary.get('final_val_accuracy'))}, "
            f"epochs={summary.get('epochs_ran', 0)}, "
            f"test_samples={summary.get('test_samples', 0)}"
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

    def _pick_csv(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            self.csv_path_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
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
            self.process_selected_file_info_var.set(self._describe_selected_file(selected_path))

    def _on_inspect_file_selected(self, _event=None) -> None:
        selection = self.inspect_session_files.curselection()
        if selection:
            selected_path = self.inspect_session_files.get(selection[0])
            self.inspect_csv_path_var.set(selected_path)
            self.inspect_selected_file_info_var.set(self._describe_selected_file(selected_path))

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
            self.root.after(0, lambda: self._log(self.train_log, "Training finished."))
            self.root.after(0, lambda: self.status_var.set("Training completed"))

        self._run_in_thread(action, "Unable to run training.")

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

        for jump in self.detected_jumps:
            self._draw_jump_windows(overview_ax, session_df, jump)

        overview_ax.set_xlabel("ms")
        overview_ax.set_ylabel("Gyroscope")
        overview_right_ax.set_ylabel("2nd derivative")
        overview_ax.plot([], [], color="royalblue", linewidth=6, alpha=0.35, label="Type window")
        overview_ax.plot([], [], color="seagreen", linewidth=6, alpha=0.35, label="Success window")
        overview_ax.plot([], [], color="crimson", linewidth=2, linestyle=":", label="Gyro saturation")
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
        zoom_ax.set_title(f"Zoom on selected jump{saturation_text}")
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
        self._redraw_plots()


def launch() -> None:
    root = tk.Tk()
    SynergieToolsApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
