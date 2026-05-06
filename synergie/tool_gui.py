from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import TYPE_CHECKING

import constants
from synergie import operations
from synergie.config import DEFAULT_COMBINATION_GAP_FRAMES, DEFAULT_SMOOTHING_SIGMA, DEFAULT_DETECTION_THRESHOLD

if TYPE_CHECKING:
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession


class SynergieToolsApp:
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
        self.train_task_var = tk.StringVar(value="type")
        self.dataset_var = tk.StringVar(value="data/annotated/total")
        self.epochs_var = tk.StringVar(value="10")

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

        self.sessions_text = scrolledtext.ScrolledText(sessions_tab, height=18, wrap=tk.WORD)
        self.sessions_text.grid(row=0, column=0, sticky="nsew")
        sessions_tab.columnconfigure(0, weight=1)
        sessions_tab.rowconfigure(0, weight=1)
        ttk.Button(sessions_tab, text="Refresh sessions", command=self._populate_sessions).grid(row=1, column=0, sticky="w", pady=(8, 0))

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
        session_box = ttk.Combobox(parent, textvariable=self.session_var, values=sorted(constants.sessions), state="readonly")
        session_box.grid(row=2, column=1, sticky="w", padx=8)
        session_box.bind("<<ComboboxSelected>>", self._on_process_session_changed)

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
        inspect_session_box = ttk.Combobox(
            controls,
            textvariable=self.inspect_session_var,
            values=sorted(constants.sessions),
            state="readonly",
            width=16,
        )
        inspect_session_box.grid(row=1, column=1, sticky="w", padx=8)
        inspect_session_box.bind("<<ComboboxSelected>>", self._on_inspect_session_changed)

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

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        for index in range(2):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)

        ttk.Label(parent, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.train_task_var, values=["type", "success"], state="readonly").grid(row=0, column=1, sticky="w")

        ttk.Label(parent, text="Dataset").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.dataset_var).grid(row=1, column=1, sticky="ew")

        ttk.Label(parent, text="Epochs").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.epochs_var).grid(row=2, column=1, sticky="w")

        ttk.Button(parent, text="Run training", command=self._run_train).grid(row=3, column=0, sticky="w", pady=(12, 12))

        self.train_log = scrolledtext.ScrolledText(parent, height=18, wrap=tk.WORD)
        self.train_log.grid(row=4, column=0, columnspan=2, sticky="nsew")
        parent.rowconfigure(4, weight=1)

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

        def action() -> None:
            epochs = int(self.epochs_var.get())
            operations.train_model(self.train_task_var.get(), self.dataset_var.get(), epochs)
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
        overview_ax.clear()
        zoom_ax.clear()
        overview_ax.set_title("Overview of IMU jump localisation signals")
        zoom_ax.set_title("Zoom on selected jump")
        overview_ax.set_xlabel("ms")
        zoom_ax.set_xlabel("ms")
        overview_ax.set_ylabel("Signal")
        zoom_ax.set_ylabel("Signal")
        overview_ax.text(0.5, 0.5, "Load a CSV in Inspect IMU", ha="center", va="center", transform=overview_ax.transAxes)
        zoom_ax.text(0.5, 0.5, "Select a detected jump to zoom", ha="center", va="center", transform=zoom_ax.transAxes)
        self.figure.tight_layout()
        self.canvas.draw_idle()

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
        overview_ax.clear()
        overview_ax.set_title("Signals used to localise jumps")
        overview_ax.plot(session_df["ms"], session_df["Gyr_X_unfiltered"], label="Gyr_X raw", linewidth=0.8, alpha=0.4)
        overview_ax.plot(session_df["ms"], session_df["Gyr_X_smoothed"], label="Gyr_X smoothed", linewidth=1.3)
        overview_ax.plot(session_df["ms"], session_df["X_gyr_second_derivative"], label="2nd derivative", linewidth=1.0)
        overview_ax.axhline(self.inspect_session.detection_threshold, color="crimson", linestyle="--", label="Detection threshold")

        for jump in self.detected_jumps:
            start_ms = session_df.iloc[jump.start]["ms"]
            end_ms = session_df.iloc[jump.end]["ms"]
            overview_ax.axvspan(start_ms, end_ms, color="orange", alpha=0.18)

        overview_ax.set_xlabel("ms")
        overview_ax.set_ylabel("Signal")
        overview_ax.legend(loc="upper right", fontsize=8)

    def _draw_zoom_plot(self, jump_index: int | None) -> None:
        _, zoom_ax = self.axes
        zoom_ax.clear()
        zoom_ax.set_title("Zoom on selected jump")

        if jump_index is None or jump_index >= len(self.detected_jumps):
            zoom_ax.text(0.5, 0.5, "Select a jump in the list", ha="center", va="center", transform=zoom_ax.transAxes)
            zoom_ax.set_xlabel("ms")
            zoom_ax.set_ylabel("Signal")
            return

        jump = self.detected_jumps[jump_index]
        session_df = self.inspect_session.df
        start_idx = max(0, jump.start - 140)
        end_idx = min(len(session_df) - 1, jump.end + 140)
        view_df = session_df.iloc[start_idx : end_idx + 1]

        zoom_ax.plot(view_df["ms"], view_df["Gyr_X_unfiltered"], label="Gyr_X raw", linewidth=0.8, alpha=0.4)
        zoom_ax.plot(view_df["ms"], view_df["Gyr_X_smoothed"], label="Gyr_X smoothed", linewidth=1.3)
        zoom_ax.plot(view_df["ms"], view_df["X_gyr_second_derivative"], label="2nd derivative", linewidth=1.0)
        zoom_ax.axhline(self.inspect_session.detection_threshold, color="crimson", linestyle="--", label="Detection threshold")

        start_ms = session_df.iloc[jump.start]["ms"]
        end_ms = session_df.iloc[jump.end]["ms"]
        zoom_ax.axvspan(start_ms, end_ms, color="orange", alpha=0.22)
        zoom_ax.axvline(start_ms, color="black", linestyle=":")
        zoom_ax.axvline(end_ms, color="black", linestyle=":")
        zoom_ax.set_xlabel("ms")
        zoom_ax.set_ylabel("Signal")
        zoom_ax.legend(loc="upper right", fontsize=8)

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
