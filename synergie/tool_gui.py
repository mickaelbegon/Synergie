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
    SEGMENT_FRAMES_BEFORE_TAKEOFF,
    SUCCESS_WINDOW_START,
    TYPE_WINDOW_START,
    TYPE_WINDOW_FRAMES,
)

if TYPE_CHECKING:
    import pandas as pd
    from core.data_treatment.data_generation.trainingSession import trainingSession


class Tooltip:
    """Small hover tooltip for Tk widgets."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self.window is not None or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.window,
            text=self.text,
            justify=tk.LEFT,
            background="#fff7d6",
            relief=tk.SOLID,
            borderwidth=1,
            padx=7,
            pady=4,
            wraplength=320,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None


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
        self.new_data_session_suggestion_var = tk.StringVar(value="No session selected.")
        self.new_data_automation_var = tk.StringVar(value="Select a session to see the automatic actions that will be applied.")
        self.annotation_files_var = tk.StringVar(value=[])
        self.annotation_summary_var = tk.StringVar(value="No annotation file selected.")
        self.annotation_progress_var = tk.StringVar(value="Pending annotations: 0")
        self.annotation_global_progress_var = tk.StringVar(value="All files pending annotations: 0")
        self.annotation_type_var = tk.StringVar(value="toe_loop")
        self.annotation_turn_var = tk.StringVar(value="")
        self.annotation_success_var = tk.StringVar(value="2")
        self.annotation_review_status_var = tk.StringVar(value="normal")
        self.annotation_athlete_var = tk.StringVar(value="")
        self.annotation_exclusion_hint_var = tk.StringVar(value="")
        self.annotation_combination_var = tk.BooleanVar(value=False)
        self.process_type_model_var = tk.StringVar()
        self.process_success_model_var = tk.StringVar()
        self.annotation_video_path_var = tk.StringVar()
        self.annotation_video_directory_var = tk.StringVar()
        self.annotation_video_matches_var = tk.StringVar(value=[])
        self.annotation_video_info_var = tk.StringVar(value="No video loaded.")
        self.annotation_sensor_sync_var = tk.StringVar(value="No sync offset saved for current sensor.")
        self.annotation_video_time_var = tk.StringVar(value="00:00.000")
        self.annotation_video_slider_var = tk.DoubleVar(value=0.0)
        self.annotation_play_button_var = tk.StringVar(value="▶")
        self.new_session_id_var = tk.StringVar()
        self.new_session_path_var = tk.StringVar()
        self.new_session_synchro_var = tk.StringVar()
        self.train_task_var = tk.StringVar(value="type")
        self.train_architecture_var = tk.StringVar(value=self.TRAIN_ARCHITECTURES["type"][0])
        self.dataset_var = tk.StringVar(value="data/annotated/total")
        self.epochs_var = tk.StringVar(value="10")
        self.train_batch_size_var = tk.StringVar(value="")
        self.train_learning_rate_var = tk.StringVar(value="")
        self.train_dropout_var = tk.StringVar(value="")
        self.train_filters_var = tk.StringVar(value="")
        self.train_modules_var = tk.StringVar(value="")
        self.train_parameter_profile_var = tk.StringVar(value="Default parameters")
        self.train_parameter_profile_summary_var = tk.StringVar(value="Using architecture defaults.")
        self.train_use_scalar_features_var = tk.BooleanVar(value=True)
        self.train_dataset_stats_var = tk.StringVar(value="Dataset stats not loaded yet.")
        self.train_quality_summary_var = tk.StringVar(value="No training run yet.")
        self.use_pretrained_var = tk.BooleanVar(value=False)
        self.pretrained_model_var = tk.StringVar()
        self.pretrained_models_summary_var = tk.StringVar(value="No pretrained model scan yet.")
        self.quality_dataset_var = tk.StringVar(value="data/annotated/total")
        self.quality_summary_var = tk.StringVar(value="Run the quality control analysis to inspect suspicious jumps.")
        self.quality_details_var = tk.StringVar(value="No suspicious jump selected.")
        self.quality_suspicious_var = tk.StringVar(value=[])
        self.dataset_review_dataset_var = tk.StringVar(value="data/annotated/total")
        self.dataset_review_summary_var = tk.StringVar(value="Load the training dataset to review already-finalized jumps.")
        self.dataset_review_feedback_var = tk.StringVar(value="")
        self.dataset_review_records_var = tk.StringVar(value=[])
        self.dataset_review_details_var = tk.StringVar(value="No training jump selected.")
        self.rotation_audit_root_var = tk.StringVar(value="data/annotated/total")
        self.rotation_audit_summary_var = tk.StringVar(value="Run the audit after labelled turn annotations are available.")
        self.rotation_audit_records_var = tk.StringVar(value=[])
        self.rotation_audit_details_var = tk.StringVar(value="No suspicious turn estimate selected.")
        self.model_audit_summary_var = tk.StringVar(value="Run the audit to verify model compatibility in the current environment.")
        self.signal_task_var = tk.StringVar(value="type")
        self.signal_dataset_var = tk.StringVar(value="data/annotated/total")
        self.signal_model_choice_var = tk.StringVar()
        self.signal_model_path_var = tk.StringVar(value=operations.latest_model_path_for_task("type"))
        self.signal_repeats_var = tk.StringVar(value="3")
        self.signal_windows_var = tk.StringVar(value="6")
        self.signal_summary_var = tk.StringVar(value="Run the analysis to estimate which signals matter.")
        self.tuner_task_var = tk.StringVar(value="type")
        self.tuner_architecture_var = tk.StringVar(value=self.TRAIN_ARCHITECTURES["type"][0])
        self.tuner_dataset_var = tk.StringVar(value="data/annotated/total")
        self.tuner_trials_var = tk.StringVar(value="6")
        self.tuner_epochs_var = tk.StringVar(value="8")
        self.tuner_summary_var = tk.StringVar(value="Run a bounded validation search to compare candidates.")
        self.tuner_progress_var = tk.DoubleVar(value=0.0)
        self.tuner_progress_text_var = tk.StringVar(value="No search running.")
        self.window_task_var = tk.StringVar(value="type")
        self.window_architecture_var = tk.StringVar(value=self.TRAIN_ARCHITECTURES["type"][0])
        self.window_dataset_var = tk.StringVar(value="data/annotated/total")
        self.window_candidates_var = tk.StringVar(value="240,200,160,120")
        self.window_epochs_var = tk.StringVar(value="8")
        self.window_summary_var = tk.StringVar(value="Compare shorter windows before changing the training window.")
        self.window_progress_var = tk.DoubleVar(value=0.0)
        self.window_progress_text_var = tk.StringVar(value="No benchmark running.")
        self.window_offset_frames_var = tk.StringVar(value="200")
        self.window_offsets_var = tk.StringVar(value="-60,-40,-20,0")
        self.detection_review_summary_var = tk.StringVar(value="Run the review scan to inspect false positives and false negatives.")
        self.detection_review_records_var = tk.StringVar(value=[])
        self.detection_tuning_summary_var = tk.StringVar(value="No threshold sweep run yet.")

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
        self.process_batch_progress_var = tk.StringVar(value="No batch running.")
        self.inspect_selected_file_info_var = tk.StringVar(value="No file selected.")
        self.detection_parameter_mode_var = tk.StringVar(value="Default parameters")
        self.detection_parameter_summary_var = tk.StringVar(value="Using default detection parameters.")

        self.inspect_dataframe = None
        self.inspect_session = None
        self.detected_jumps: list = []
        self._new_data_files_cache: list[dict] = []
        self._annotation_files_cache: list[Path] = []
        self._annotation_video_match_cache: list[dict] = []
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
        self.train_confusion_figure = None
        self.train_confusion_ax = None
        self.train_confusion_canvas = None
        self._train_confusion_colorbar = None
        self.quality_figure = None
        self.quality_axes = None
        self.quality_canvas = None
        self.dataset_review_signal_figure = None
        self.dataset_review_signal_ax = None
        self.dataset_review_signal_acc_ax = None
        self.dataset_review_angle_ax = None
        self.dataset_review_signal_canvas = None
        self.rotation_audit_figure = None
        self.rotation_audit_ax = None
        self.rotation_audit_error_ax = None
        self.rotation_audit_rule_ax = None
        self.rotation_audit_canvas = None
        self.rotation_audit_signal_figure = None
        self.rotation_audit_signal_ax = None
        self.rotation_audit_signal_acc_ax = None
        self.rotation_audit_angle_ax = None
        self.rotation_audit_signal_canvas = None
        self.signal_figure = None
        self.signal_axes = None
        self.signal_canvas = None
        self.tuner_figure = None
        self.tuner_ax = None
        self.tuner_canvas = None
        self.annotation_dataframe = None
        self.annotation_file_path: Path | None = None
        self.quality_analysis: dict | None = None
        self.dataset_review_records: list[dict] = []
        self.rotation_audit_analysis: dict | None = None
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
        self.annotation_video_popup = None
        self.annotation_video_popup_info_label = None
        self.annotation_video_matches_listbox = None
        self._tooltips: list[Tooltip] = []

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
            text="Pipeline: New data -> Process -> Annotate -> Quality review -> Train, puis outils d'inspection et d'analyse des modeles.",
        ).grid(row=1, column=0, sticky="w")

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.notebook = notebook

        workflow_tab = ttk.Frame(notebook, padding=12)
        sessions_tab = ttk.Frame(notebook, padding=12)
        process_tab = ttk.Frame(notebook, padding=12)
        new_data_tab = ttk.Frame(notebook, padding=12)
        annotate_tab = ttk.Frame(notebook, padding=12)
        inspect_tab = ttk.Frame(notebook, padding=12)
        train_tab = ttk.Frame(notebook, padding=12)
        model_audit_tab = ttk.Frame(notebook, padding=12)
        signal_tab = ttk.Frame(notebook, padding=12)
        tuner_tab = ttk.Frame(notebook, padding=12)
        window_tab = ttk.Frame(notebook, padding=12)
        detection_tuning_tab = ttk.Frame(notebook, padding=12)
        quality_tab = ttk.Frame(notebook, padding=12)
        dataset_review_tab = ttk.Frame(notebook, padding=12)
        rotation_audit_tab = ttk.Frame(notebook, padding=12)
        inventory_tab = ttk.Frame(notebook, padding=12)
        notes_tab = ttk.Frame(notebook, padding=12)
        notebook.add(workflow_tab, text="Start - Workflow")
        notebook.add(inventory_tab, text="Data - Status")
        notebook.add(sessions_tab, text="Data - Sessions")
        notebook.add(new_data_tab, text="Data - New")
        notebook.add(process_tab, text="Data - Process")
        notebook.add(annotate_tab, text="Data - Annotate")
        notebook.add(quality_tab, text="Review - Quality")
        notebook.add(dataset_review_tab, text="Review - Dataset")
        notebook.add(rotation_audit_tab, text="Review - Turns")
        notebook.add(inspect_tab, text="Review - Inspect IMU")
        notebook.add(detection_tuning_tab, text="Review - Detection")
        notebook.add(train_tab, text="Models - Train")
        notebook.add(signal_tab, text="Models - Importance")
        notebook.add(tuner_tab, text="Models - Tune")
        notebook.add(window_tab, text="Models - Windows")
        notebook.add(model_audit_tab, text="Models - Audit")
        notebook.add(notes_tab, text="Notes")

        sessions_tab.columnconfigure(0, weight=1)
        sessions_tab.columnconfigure(1, weight=0)
        sessions_tab.rowconfigure(1, weight=1)

        ttk.Label(
            sessions_tab,
            text=(
                "Le flux normal cree les sessions depuis Data - New a partir de la date/heure des fichiers IMU. "
                "Utiliser l'ajout manuel seulement pour corriger une ancienne session ou un cas exceptionnel."
            ),
            justify=tk.LEFT,
            wraplength=720,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        self.sessions_text = scrolledtext.ScrolledText(sessions_tab, height=18, wrap=tk.WORD)
        self.sessions_text.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        self._build_sessions_side_panel(sessions_tab)
        ttk.Button(sessions_tab, text="Refresh sessions", command=self._refresh_all_sessions).grid(row=2, column=0, sticky="w", pady=(8, 0))

        self._build_workflow_tab(workflow_tab)
        self._build_inventory_tab(inventory_tab)
        self._build_process_tab(process_tab)
        self._build_new_data_tab(new_data_tab)
        self._build_annotate_tab(annotate_tab)
        self._build_inspect_tab(inspect_tab)
        self._build_train_tab(train_tab)
        self._build_model_audit_tab(model_audit_tab)
        self._build_signal_importance_tab(signal_tab)
        self._build_hyperparameter_tab(tuner_tab)
        self._build_window_benchmark_tab(window_tab)
        self._build_detection_tuning_tab(detection_tuning_tab)
        self._build_quality_tab(quality_tab)
        self._build_dataset_review_tab(dataset_review_tab)
        self._build_rotation_audit_tab(rotation_audit_tab)
        self._build_notes_tab(notes_tab)

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.root.bind_all("<KeyPress>", self._on_global_keypress)

    def _add_tooltip(self, widget, text: str):
        self._tooltips.append(Tooltip(widget, text))
        return widget

    def _build_workflow_tab(self, parent: ttk.Frame) -> None:
        """Build the first-tab guide for the end-to-end data workflow."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="Workflow conseille pour ajouter de nouvelles donnees et re-entrainer les modeles",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        workflow_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=24)
        workflow_text.grid(row=1, column=0, sticky="nsew")
        workflow_text.insert(
            tk.END,
            "\n".join(
                [
                    "1. Data - New",
                    "   Choisir une seance brute depuis data/new. Le logiciel propose automatiquement l'ID de session",
                    "   et le dossier data/raw cible a partir de la date/heure des fichiers IMU.",
                    "",
                    "2. Data - Sessions",
                    "   Normalement cree automatiquement depuis Data - New.",
                    "   Utiliser l'ajout manuel seulement pour corriger une ancienne seance ou un cas exceptionnel.",
                    "",
                    "3. Data - New",
                    "   Verifier les capteurs detectes, puis utiliser Process for annotation.",
                    "   Cette etape cree les segments IMU et un fichier *_for_annotation.csv dans data/pending.",
                    "",
                    "4. Data - Process",
                    "   Optionnel pour le flux d'annotation complet.",
                    "   Sert au retraitement manuel ou en batch de CSV deja classes par session, avec les modeles initiaux choisis.",
                    "",
                    "5. Data - Annotate",
                    "   Charger le fichier for_annotation, synchroniser la video, verifier/corriger les labels, traiter les faux positifs",
                    "   et les sauts manques, puis utiliser Finalize annotated file quand tout est termine.",
                    "",
                    "6. Review - Quality et Review - Detection",
                    "   Verifier les outliers, les sequences suspectes, les faux positifs et les faux negatifs avant d'entrainer.",
                    "",
                    "7. Models - Train",
                    "   Re-entrainer type, success et turns quand le dataset a change.",
                    "",
                    "8. Models - Audit / Importance / Tune",
                    "   Verifier les performances, comprendre les signaux utiles et ajuster les hyperparametres avant de promouvoir un modele.",
                    "",
                    "Points de vigilance",
                    "- Les sessions doivent idealement etre creees depuis Data - New; l'ajout manuel est un mode avance.",
                    "- Finalize annotated file est l'etape qui ajoute reellement les nouveaux labels au dataset d'entrainement.",
                    "- Les fichiers jumplist sont des sorties derivees; ils ne doivent pas etre retraités comme des CSV IMU bruts.",
                    "- Le controle apres entrainement est recommande avant d'utiliser un nouveau modele pour pre-remplir d'autres annotations.",
                ]
            ),
        )
        workflow_text.configure(state=tk.DISABLED)

    def _build_process_tab(self, parent: ttk.Frame) -> None:
        for index in range(3):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)

        ttk.Label(parent, text="Session").grid(row=0, column=0, sticky="w", pady=4)
        self.process_session_box = ttk.Combobox(parent, textvariable=self.session_var, values=operations.list_sessions(), state="readonly")
        self.process_session_box.grid(row=0, column=1, sticky="w", padx=8)
        self._add_tooltip(self.process_session_box, "Session utilisee pour recuperer l'offset de synchronisation configure.")
        self.process_session_box.bind("<<ComboboxSelected>>", self._on_process_session_changed)

        ttk.Label(parent, text="Session CSV files").grid(row=1, column=0, sticky="nw", pady=4)
        self.process_session_files = tk.Listbox(
            parent,
            listvariable=self.session_files_var,
            height=6,
            exportselection=False,
            selectmode=tk.EXTENDED,
        )
        self.process_session_files.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=8)
        self._add_tooltip(self.process_session_files, "Selectionner un ou plusieurs CSV IMU bruts. Les sorties jumplist sont nommees automatiquement.")
        self.process_session_files.bind("<<ListboxSelect>>", self._on_process_file_selected)

        ttk.Label(parent, textvariable=self.session_folder_summary_var, justify=tk.LEFT).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(parent, textvariable=self.process_selected_file_info_var, justify=tk.LEFT).grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 8))

        batch_process_button = ttk.Button(parent, text="Batch process selected", command=self._run_batch_process_files)
        batch_process_button.grid(row=4, column=0, sticky="w", pady=(12, 12))
        self._add_tooltip(batch_process_button, "Traite la selection, meme s'il n'y a qu'un seul CSV, et genere automatiquement chaque sortie.")
        global_batch_button = ttk.Button(parent, text="Batch process all", command=self._run_global_batch_process_files)
        global_batch_button.grid(row=4, column=1, sticky="w", pady=(12, 12), padx=8)
        self._add_tooltip(global_batch_button, "Traite tous les fichiers CSV bruts disponibles dans toutes les sessions configurees.")

        batch_progress_label = ttk.Label(parent, textvariable=self.process_batch_progress_var, justify=tk.LEFT)
        batch_progress_label.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 12))
        self._add_tooltip(batch_progress_label, "Affiche la progression du batch courant, fichier par fichier.")

        prediction_frame = ttk.LabelFrame(parent, text="Initial model predictions", padding=8)
        prediction_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 12))
        prediction_frame.columnconfigure(1, weight=1)
        ttk.Label(prediction_frame, text="Type model").grid(row=0, column=0, sticky="w")
        self.process_type_model_box = ttk.Combobox(prediction_frame, textvariable=self.process_type_model_var, state="readonly")
        self.process_type_model_box.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._add_tooltip(self.process_type_model_box, "Modele utilise pour proposer le type initial pendant le traitement.")
        ttk.Label(prediction_frame, text="Success model").grid(row=1, column=0, sticky="w")
        self.process_success_model_box = ttk.Combobox(prediction_frame, textvariable=self.process_success_model_var, state="readonly")
        self.process_success_model_box.grid(row=1, column=1, sticky="ew", padx=(8, 0))
        self._add_tooltip(self.process_success_model_box, "Modele utilise pour proposer succes ou chute pendant le traitement.")
        self._refresh_process_prediction_models()

        self.process_log = scrolledtext.ScrolledText(parent, height=18, wrap=tk.WORD)
        self.process_log.grid(row=7, column=0, columnspan=3, sticky="nsew")
        parent.rowconfigure(7, weight=1)

    def _build_inventory_tab(self, parent: ttk.Frame) -> None:
        """Build a cross-folder view of the current data pipeline state."""
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        ttk.Label(
            parent,
            text="Etat derive automatiquement depuis data/new, data/pending, data/annotated et le jumplist d'entrainement.",
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(parent, text="Refresh data status", command=self._refresh_data_inventory).grid(row=1, column=0, sticky="w", pady=(0, 8))
        columns = ("session", "new", "pending", "segments", "trainable", "total", "trainable_total", "workflow", "predictions")
        self.data_inventory_tree = ttk.Treeview(parent, columns=columns, show="headings")
        headings = {
            "session": "Session / folder",
            "new": "New IMU",
            "pending": "Pending CSV",
            "segments": "Segments stored",
            "trainable": "Trainable labels",
            "total": "Rows in total",
            "trainable_total": "Trainable in total",
            "workflow": "Workflow",
            "predictions": "Predictions",
        }
        widths = {"session": 180, "new": 80, "pending": 90, "segments": 110, "trainable": 110, "total": 100, "trainable_total": 110, "workflow": 140, "predictions": 120}
        for column in columns:
            self.data_inventory_tree.heading(column, text=headings[column])
            self.data_inventory_tree.column(column, width=widths[column], anchor="center" if column != "session" else "w")
        self.data_inventory_tree.tag_configure("total", font=("Segoe UI", 9, "bold"))
        self.data_inventory_tree.grid(row=2, column=0, sticky="nsew")
        ttk.Button(parent, text="Import legacy annotated session...", command=self._import_legacy_annotated_session).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self._refresh_data_inventory()

    def _build_inspect_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(parent, text="Detection Controls", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=(0, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Input CSV").grid(row=0, column=0, sticky="w", pady=4)
        inspect_entry = ttk.Entry(controls, textvariable=self.inspect_csv_path_var, width=44)
        inspect_entry.grid(row=0, column=1, sticky="ew", padx=8)
        self._add_tooltip(inspect_entry, "CSV IMU a inspecter visuellement.")
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
        self._add_tooltip(self.inspect_session_box, "Session utilisee pour appliquer sa synchronisation.")
        self.inspect_session_box.bind("<<ComboboxSelected>>", self._on_inspect_session_changed)

        ttk.Label(controls, text="Session CSV files").grid(row=2, column=0, sticky="nw", pady=(12, 0))
        self.inspect_session_files = tk.Listbox(controls, listvariable=self.inspect_session_files_var, height=5, exportselection=False)
        self.inspect_session_files.grid(row=2, column=1, columnspan=2, sticky="ew", padx=8)
        self._add_tooltip(self.inspect_session_files, "Double-cliquer sur un CSV pour le charger et lancer la detection.")
        self.inspect_session_files.bind("<<ListboxSelect>>", self._on_inspect_file_selected)
        self.inspect_session_files.bind("<Double-1>", self._on_inspect_file_double_clicked)

        ttk.Label(controls, textvariable=self.inspect_folder_summary_var, justify=tk.LEFT).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Label(controls, textvariable=self.inspect_selected_file_info_var, justify=tk.LEFT).grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 8))

        ttk.Label(controls, text="Parameter set").grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.detection_parameter_mode_box = ttk.Combobox(
            controls,
            textvariable=self.detection_parameter_mode_var,
            values=["Default parameters"],
            state="readonly",
        )
        self.detection_parameter_mode_box.grid(row=5, column=1, sticky="ew", padx=8)
        self.detection_parameter_mode_box.bind("<<ComboboxSelected>>", self._on_detection_parameter_mode_changed)
        ttk.Label(controls, textvariable=self.detection_parameter_summary_var, justify=tk.LEFT).grid(row=5, column=2, sticky="w")

        ttk.Label(controls, text="2nd derivative threshold").grid(row=6, column=0, sticky="w", pady=(12, 0))
        threshold_scale = tk.Scale(
            controls,
            from_=-2.0,
            to=0.5,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        )
        threshold_scale.grid(row=6, column=1, sticky="ew", padx=8)
        self._add_tooltip(threshold_scale, "Seuil applique a la derivee seconde. Plus proche de zero = detection plus permissive.")
        ttk.Label(controls, textvariable=self.threshold_label_var).grid(row=6, column=2, sticky="w")

        ttk.Label(controls, text="Smoothing sigma").grid(row=7, column=0, sticky="w", pady=(12, 0))
        sigma_scale = tk.Scale(
            controls,
            from_=1,
            to=60,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.sigma_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        )
        sigma_scale.grid(row=7, column=1, sticky="ew", padx=8)
        self._add_tooltip(sigma_scale, "Lissage du gyroscope avant detection. Plus grand = signal plus lisse mais moins reactif.")
        ttk.Label(controls, textvariable=self.sigma_label_var).grid(row=7, column=2, sticky="w")

        ttk.Label(controls, text="Combination gap (frames)").grid(row=8, column=0, sticky="w", pady=(12, 0))
        gap_scale = tk.Scale(
            controls,
            from_=60,
            to=360,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.gap_var,
            command=lambda _value: self._sync_slider_labels(),
            length=260,
        )
        gap_scale.grid(row=8, column=1, sticky="ew", padx=8)
        self._add_tooltip(gap_scale, "Deux sauts proches de moins que cet ecart sont marques comme combinaison.")
        ttk.Label(controls, textvariable=self.gap_label_var).grid(row=8, column=2, sticky="w")

        actions = ttk.Frame(controls)
        actions.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        inspect_button = ttk.Button(actions, text="Load and detect", command=self._run_inspection)
        inspect_button.grid(row=0, column=0, sticky="w")
        self._add_tooltip(inspect_button, "Charge le CSV et recalcule les sauts avec les seuils visibles.")
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
        self._refresh_detection_parameter_modes()

    def _build_new_data_tab(self, parent: ttk.Frame) -> None:
        for index in range(3):
            parent.columnconfigure(index, weight=1 if index == 1 else 0)
        parent.rowconfigure(6, weight=1)

        ttk.Label(parent, text="Folder").grid(row=0, column=0, sticky="w", pady=4)
        self.new_data_directory_box = ttk.Combobox(parent, textvariable=self.new_data_directory_var, state="readonly")
        self.new_data_directory_box.grid(row=0, column=1, sticky="ew", padx=8)
        self.new_data_directory_box.bind("<<ComboboxSelected>>", self._on_new_data_directory_changed)

        ttk.Button(parent, text="Refresh folders", command=self._refresh_new_data_directories).grid(row=0, column=2, sticky="e")

        ttk.Label(parent, text="IMU files").grid(row=1, column=0, sticky="nw", pady=4)
        self.new_data_files = tk.Listbox(parent, listvariable=self.new_data_files_var, height=8, exportselection=False)
        self.new_data_files.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=8)
        self.new_data_files.bind("<<ListboxSelect>>", self._on_new_data_file_selected)

        ttk.Label(parent, text="Automatic annotation CSV").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Label(parent, textvariable=self.new_data_output_var, justify=tk.LEFT).grid(row=2, column=1, columnspan=2, sticky="w", padx=8)

        ttk.Label(parent, text="Automatic session").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Label(parent, textvariable=self.new_data_session_suggestion_var, justify=tk.LEFT).grid(row=3, column=1, sticky="w", padx=8)
        auto_session_button = ttk.Button(parent, text="Add session automatically", command=self._add_selected_new_data_session)
        auto_session_button.grid(row=3, column=2, sticky="e")
        self._add_tooltip(auto_session_button, "Cree la session avec un ID et un chemin derives de la date/heure du fichier IMU.")

        ttk.Button(parent, text="Process for annotation", command=self._run_process_new_data_file).grid(row=4, column=0, sticky="w", pady=(12, 8))
        ttk.Label(parent, textvariable=self.new_data_summary_var, justify=tk.LEFT).grid(row=4, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Label(
            parent,
            textvariable=self.new_data_automation_var,
            justify=tk.LEFT,
            foreground="gray",
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.new_data_log = scrolledtext.ScrolledText(parent, height=14, wrap=tk.WORD)
        self.new_data_log.grid(row=6, column=0, columnspan=3, sticky="nsew")
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
        self._add_tooltip(self.annotation_files_listbox, "Fichiers a annoter. Le compteur montre les sauts encore non traites.")
        self.annotation_files_listbox.bind("<<ListboxSelect>>", self._on_annotation_file_selected)

        jumps_panel = ttk.LabelFrame(left_panel, text="Session Timeline", padding=8)
        jumps_panel.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        jumps_panel.columnconfigure(0, weight=1)
        jumps_panel.rowconfigure(3, weight=1)
        ttk.Label(jumps_panel, textvariable=self.annotation_summary_var, justify=tk.LEFT).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(jumps_panel, textvariable=self.annotation_progress_var, justify=tk.LEFT, foreground="firebrick").grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Label(jumps_panel, textvariable=self.annotation_global_progress_var, justify=tk.LEFT).grid(row=2, column=0, sticky="w", pady=(0, 8))
        self.annotation_jump_listbox = tk.Listbox(jumps_panel, exportselection=False, height=16)
        self.annotation_jump_listbox.grid(row=3, column=0, sticky="nsew")
        self._add_tooltip(self.annotation_jump_listbox, "Tous les candidats de la seance, tries par temps video synchronise.")
        self.annotation_jump_listbox.bind("<<ListboxSelect>>", self._on_annotation_jump_selected)

        right_panel = ttk.Panedwindow(parent, orient=tk.HORIZONTAL)
        right_panel.grid(row=0, column=1, sticky="nsew")

        video_frame = ttk.LabelFrame(right_panel, text="Video Review", padding=8)
        video_frame.columnconfigure(1, weight=1)
        video_frame.rowconfigure(2, weight=1)

        ttk.Label(video_frame, text="Session video").grid(row=0, column=0, sticky="w")
        ttk.Label(video_frame, textvariable=self.annotation_video_path_var, justify=tk.LEFT, wraplength=300).grid(row=0, column=1, sticky="w", padx=8)
        choose_video_button = ttk.Button(video_frame, text="Choose video...", command=self._open_annotation_video_popup)
        choose_video_button.grid(row=0, column=2, sticky="e")
        self._add_tooltip(choose_video_button, "Cherche les videos proches de l'heure de la seance et permet de choisir la bonne.")

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
        self._add_tooltip(self.annotation_video_label, "Video de la seance utilisee pour verifier les labels.")

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
        self._add_tooltip(self.annotation_video_slider, "Position courante dans la video.")
        self.annotation_video_slider.bind("<ButtonRelease-1>", self._on_annotation_video_slider_released)
        ttk.Label(video_timeline, textvariable=self.annotation_video_time_var, width=12).grid(row=0, column=1, sticky="e", padx=(8, 0))

        video_controls = ttk.Frame(video_frame)
        video_controls.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        ttk.Button(video_controls, text="⏪", width=3, command=lambda: self._seek_annotation_video_relative(-1000)).grid(row=0, column=0, sticky="w")
        ttk.Button(video_controls, text="⏩", width=3, command=lambda: self._seek_annotation_video_relative(1000)).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Button(video_controls, textvariable=self.annotation_play_button_var, width=3, command=self._play_annotation_video).grid(row=0, column=2, sticky="w", padx=(12, 0))
        ttk.Button(video_controls, text="⌖", width=3, command=self._seek_annotation_video_to_current_jump).grid(row=0, column=3, sticky="w", padx=(6, 0))
        ttk.Button(video_controls, text="⏩J", width=4, command=self._play_annotation_to_current_jump).grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Button(video_controls, text="⏹", width=3, command=self._stop_annotation_playback).grid(row=0, column=5, sticky="w", padx=(6, 0))

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

        ttk.Label(
            controls,
            text="Shortcuts: t/f/z/s/l/a = jump type | 1-4 = turns | 0/1 = fall/success | u = unseen | x = weird signal",
            justify=tk.LEFT,
            wraplength=360,
            foreground="#666666",
        ).grid(row=11, column=0, sticky="w", pady=(0, 8))

        combination_check = ttk.Checkbutton(controls, text="Combination jump", variable=self.annotation_combination_var)
        combination_check.grid(row=12, column=0, sticky="w", pady=(0, 8))
        self._add_tooltip(combination_check, "Deux sauts du meme capteur espaces de moins de 1,5 s sont pre-marques comme combinaison.")
        annotation_actions = ttk.Frame(controls)
        annotation_actions.grid(row=13, column=0, sticky="w", pady=(8, 0))
        save_annotation_button = ttk.Button(annotation_actions, text="Save current annotation", command=self._save_current_annotation)
        save_annotation_button.grid(row=0, column=0, sticky="w")
        self._add_tooltip(save_annotation_button, "Enregistre le label du saut actuellement selectionne.")
        finalize_button = ttk.Button(annotation_actions, text="Finalize annotated file", command=self._finalize_current_annotation_file)
        finalize_button.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0),
        )
        self._add_tooltip(finalize_button, "Archive le jumplist actuel, deplace les segments et ajoute les labels termines au jeu d'entrainement.")
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

    def _build_train_confusion_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(4.2, 3.8), dpi=100)
        axis = figure.add_subplot(111)
        self.train_confusion_figure = figure
        self.train_confusion_ax = axis
        self.train_confusion_canvas = FigureCanvasTkAgg(figure, master=self.train_confusion_container)
        self.train_confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_confusion_matrix()

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

    def _build_dataset_review_signal_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10.0, 4.4), dpi=100)
        axis = figure.add_subplot(121)
        angle_ax = figure.add_subplot(122, sharex=axis)
        self.dataset_review_signal_figure = figure
        self.dataset_review_signal_ax = axis
        self.dataset_review_signal_acc_ax = None
        self.dataset_review_angle_ax = angle_ax
        self.dataset_review_signal_canvas = FigureCanvasTkAgg(figure, master=self.dataset_review_signal_container)
        self.dataset_review_signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_dataset_review_signal()

    def _build_rotation_audit_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10.0, 3.8), dpi=100)
        matrix_ax = figure.add_subplot(131)
        error_ax = figure.add_subplot(132)
        rule_ax = figure.add_subplot(133)
        self.rotation_audit_figure = figure
        self.rotation_audit_ax = matrix_ax
        self.rotation_audit_error_ax = error_ax
        self.rotation_audit_rule_ax = rule_ax
        self.rotation_audit_canvas = FigureCanvasTkAgg(figure, master=self.rotation_audit_plot_container)
        self.rotation_audit_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_rotation_audit()

    def _build_rotation_audit_signal_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(10.0, 4.4), dpi=100)
        axis = figure.add_subplot(121)
        angle_ax = figure.add_subplot(122, sharex=axis)
        self.rotation_audit_signal_figure = figure
        self.rotation_audit_signal_ax = axis
        self.rotation_audit_signal_acc_ax = None
        self.rotation_audit_angle_ax = angle_ax
        self.rotation_audit_signal_canvas = FigureCanvasTkAgg(figure, master=self.rotation_audit_signal_container)
        self.rotation_audit_signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_rotation_audit_signal()

    def _build_signal_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.8, 6.6), dpi=100)
        channel_ax = figure.add_subplot(311)
        scalar_ax = figure.add_subplot(312)
        time_ax = figure.add_subplot(313)
        self.signal_figure = figure
        self.signal_axes = (channel_ax, scalar_ax, time_ax)
        self.signal_canvas = FigureCanvasTkAgg(figure, master=self.signal_plot_container)
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_signal_plot()

    def _build_tuner_plot_canvas(self) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(8.8, 3.2), dpi=100)
        axis = figure.add_subplot(111)
        self.tuner_figure = figure
        self.tuner_ax = axis
        self.tuner_canvas = FigureCanvasTkAgg(figure, master=self.tuner_plot_container)
        self.tuner_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._draw_placeholder_tuner_plot()

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
        panel = ttk.LabelFrame(parent, text="Advanced manual add", padding=12)
        panel.grid(row=1, column=1, sticky="ns")
        panel.columnconfigure(1, weight=1)

        ttk.Label(
            panel,
            text="Reserve aux exceptions.\nLe flux normal passe par Data - New.",
            justify=tk.LEFT,
            wraplength=220,
            foreground="gray",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(panel, text="Session ID").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_id_var, width=18).grid(row=1, column=1, sticky="ew")

        ttk.Label(panel, text="Relative path").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_path_var, width=22).grid(row=2, column=1, sticky="ew")

        ttk.Label(panel, text="Synchro").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(panel, textvariable=self.new_session_synchro_var, width=18).grid(row=3, column=1, sticky="ew")

        ttk.Button(panel, text="Add session", command=self._add_session_from_gui).grid(row=4, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _build_train_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls_host = ttk.Frame(parent)
        controls_host.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls_host.columnconfigure(0, weight=1)
        controls_host.rowconfigure(0, weight=1)
        controls_canvas = tk.Canvas(controls_host, highlightthickness=0, width=420)
        controls_canvas.grid(row=0, column=0, sticky="nsew")
        controls_scrollbar = ttk.Scrollbar(controls_host, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

        controls = ttk.LabelFrame(controls_canvas, text="Training Setup", padding=12)
        controls.columnconfigure(1, weight=1)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(controls_window, width=event.width),
        )

        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        train_task_box = ttk.Combobox(controls, textvariable=self.train_task_var, values=["type", "success"], state="readonly", width=18)
        train_task_box.grid(row=0, column=1, sticky="ew")
        self._add_tooltip(train_task_box, "Choisit la cible a apprendre: type de saut ou succes/chute.")
        train_task_box.bind("<<ComboboxSelected>>", self._on_train_task_changed)

        ttk.Label(controls, text="Architecture").grid(row=1, column=0, sticky="w", pady=4)
        self.train_architecture_box = ttk.Combobox(controls, textvariable=self.train_architecture_var, state="readonly", width=24)
        self.train_architecture_box.grid(row=1, column=1, sticky="ew")
        self._add_tooltip(self.train_architecture_box, "Architecture du reseau pour un nouvel entrainement.")
        self.train_architecture_box.bind("<<ComboboxSelected>>", self._on_train_architecture_changed)
        self._sync_train_architectures()

        ttk.Label(controls, text="Dataset").grid(row=2, column=0, sticky="w", pady=4)
        dataset_entry = ttk.Entry(controls, textvariable=self.dataset_var, width=34)
        dataset_entry.grid(row=2, column=1, sticky="ew")
        self._add_tooltip(dataset_entry, "Dossier contenant jumplist.csv et skaterData.csv pour l'entrainement.")

        ttk.Label(controls, text="Epochs").grid(row=3, column=0, sticky="w", pady=4)
        epochs_entry = ttk.Entry(controls, textvariable=self.epochs_var, width=10)
        epochs_entry.grid(row=3, column=1, sticky="w")
        self._add_tooltip(epochs_entry, "Nombre maximal de passes sur le jeu d'entrainement.")

        ttk.Label(controls, text="Parameter profile").grid(row=4, column=0, sticky="w", pady=4)
        self.train_parameter_profile_box = ttk.Combobox(
            controls,
            textvariable=self.train_parameter_profile_var,
            values=["Default parameters", "Optimized parameters"],
            state="readonly",
            width=24,
        )
        self.train_parameter_profile_box.grid(row=4, column=1, sticky="ew")
        self.train_parameter_profile_box.bind("<<ComboboxSelected>>", self._on_train_parameter_profile_changed)
        self._add_tooltip(
            self.train_parameter_profile_box,
            "Choisit les valeurs par defaut de l'architecture ou le meilleur profil sauvegarde depuis Models - Tune.",
        )
        ttk.Label(controls, textvariable=self.train_parameter_profile_summary_var, justify=tk.LEFT, wraplength=330).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 4),
        )
        scalar_check = ttk.Checkbutton(
            controls,
            text="Use weight and height",
            variable=self.train_use_scalar_features_var,
        )
        scalar_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 4))
        self._add_tooltip(
            scalar_check,
            "Si decoche, l'entrainement neutralise les entrees masse/taille avec des zeros et ne requiert plus skaterData.csv pour ces lignes.",
        )

        ttk.Label(controls, text="Batch size").grid(row=7, column=0, sticky="w", pady=4)
        batch_entry = ttk.Entry(controls, textvariable=self.train_batch_size_var, width=10)
        batch_entry.grid(row=7, column=1, sticky="w")
        self._add_tooltip(batch_entry, "Nombre d'exemples traites avant chaque mise a jour des poids. Vide = valeur Keras par defaut.")
        ttk.Label(controls, text="Learning rate").grid(row=8, column=0, sticky="w", pady=4)
        learning_rate_entry = ttk.Entry(controls, textvariable=self.train_learning_rate_var, width=10)
        learning_rate_entry.grid(row=8, column=1, sticky="w")
        self._add_tooltip(learning_rate_entry, "Taille des mises a jour de l'optimiseur. Vide = valeur de l'architecture.")
        ttk.Label(controls, text="Dropout").grid(row=9, column=0, sticky="w", pady=4)
        dropout_entry = ttk.Entry(controls, textvariable=self.train_dropout_var, width=10)
        dropout_entry.grid(row=9, column=1, sticky="w")
        self._add_tooltip(dropout_entry, "Part de neurones coupes pendant l'entrainement pour limiter le surapprentissage.")
        ttk.Label(controls, text="Filters / units").grid(row=10, column=0, sticky="w", pady=4)
        filters_entry = ttk.Entry(controls, textvariable=self.train_filters_var, width=10)
        filters_entry.grid(row=10, column=1, sticky="w")
        self._add_tooltip(filters_entry, "Largeur principale du modele: filtres convolutionnels ou unites LSTM selon l'architecture.")
        ttk.Label(controls, text="Modules / blocks").grid(row=11, column=0, sticky="w", pady=4)
        modules_entry = ttk.Entry(controls, textvariable=self.train_modules_var, width=10)
        modules_entry.grid(row=11, column=1, sticky="w")
        self._add_tooltip(modules_entry, "Profondeur principale: modules Inception ou blocs Transformer selon l'architecture.")

        pretrained_check = ttk.Checkbutton(
            controls,
            text="Start from pretrained model",
            variable=self.use_pretrained_var,
            command=self._sync_pretrained_controls,
        )
        pretrained_check.grid(row=12, column=0, columnspan=2, sticky="w", pady=(12, 4))
        self._add_tooltip(pretrained_check, "Reprend un modele existant compatible au lieu de repartir de zero.")

        self.pretrained_model_box = ttk.Combobox(controls, textvariable=self.pretrained_model_var, state="readonly", width=34)
        self.pretrained_model_box.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._add_tooltip(self.pretrained_model_box, "Modeles compatibles classes du plus recent au moins recent.")
        self.pretrained_model_box.bind("<<ComboboxSelected>>", self._on_pretrained_model_changed)

        buttons = ttk.Frame(controls)
        buttons.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(4, 8))
        train_button = ttk.Button(buttons, text="Run training", command=self._run_train)
        train_button.grid(row=0, column=0, sticky="w")
        self._add_tooltip(train_button, "Lance l'entrainement avec les options visibles.")
        refresh_stats_button = ttk.Button(buttons, text="Refresh dataset stats", command=self._refresh_training_dataset_stats)
        refresh_stats_button.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self._add_tooltip(refresh_stats_button, "Recharge les comptes de classes, doublons et statut de reentrainement.")
        delete_model_button = ttk.Button(buttons, text="Delete selected model", command=self._delete_selected_pretrained_model)
        delete_model_button.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self._add_tooltip(delete_model_button, "Supprime du registre et du disque le modele archive selectionne, apres confirmation.")

        ttk.Label(
            controls,
            textvariable=self.pretrained_models_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=15, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(
            controls,
            textvariable=self.train_dataset_stats_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=16, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(
            controls,
            textvariable=self.train_quality_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=17, column=0, columnspan=2, sticky="w")

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

        bottom_results = ttk.Frame(results)
        bottom_results.grid(row=1, column=0, sticky="nsew")
        bottom_results.columnconfigure(0, weight=1)
        bottom_results.columnconfigure(1, weight=1)
        bottom_results.rowconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(bottom_results, text="Training Log", padding=8)
        log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.train_log = scrolledtext.ScrolledText(log_frame, height=14, wrap=tk.WORD)
        self.train_log.grid(row=0, column=0, sticky="nsew")

        confusion_frame = ttk.LabelFrame(bottom_results, text="Confusion Matrix", padding=8)
        confusion_frame.grid(row=0, column=1, sticky="nsew")
        confusion_frame.columnconfigure(0, weight=1)
        confusion_frame.rowconfigure(0, weight=1)
        self.train_confusion_container = ttk.Frame(confusion_frame)
        self.train_confusion_container.grid(row=0, column=0, sticky="nsew")
        self._build_train_confusion_canvas()
        self._refresh_pretrained_models()
        self._sync_pretrained_controls()
        self._refresh_training_dataset_stats()

    def _build_model_audit_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        header = ttk.Frame(parent)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(header, text="Run model audit", command=self._run_model_audit).grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.model_audit_summary_var, justify=tk.LEFT, wraplength=860).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(12, 0),
        )
        self.model_audit_text = scrolledtext.ScrolledText(parent, height=24, wrap=tk.WORD)
        self.model_audit_text.grid(row=1, column=0, sticky="nsew")

    def _build_signal_importance_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Signal Importance Setup", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        signal_task_box = ttk.Combobox(controls, textvariable=self.signal_task_var, values=["type", "success"], state="readonly")
        signal_task_box.grid(row=0, column=1, sticky="ew")
        signal_task_box.bind("<<ComboboxSelected>>", self._on_signal_task_changed)
        ttk.Label(controls, text="Dataset").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.signal_dataset_var, width=34).grid(row=1, column=1, sticky="ew")
        ttk.Label(controls, text="Saved model").grid(row=2, column=0, sticky="w", pady=4)
        self.signal_model_box = ttk.Combobox(controls, textvariable=self.signal_model_choice_var, state="readonly", width=34)
        self.signal_model_box.grid(row=2, column=1, sticky="ew")
        self.signal_model_box.bind("<<ComboboxSelected>>", self._on_signal_model_changed)
        self._add_tooltip(self.signal_model_box, "Modeles compatibles classes du plus recent au moins recent.")
        ttk.Label(controls, text="Model path").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.signal_model_path_var, width=34).grid(row=3, column=1, sticky="ew")
        ttk.Label(controls, text="Repeats").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.signal_repeats_var, width=10).grid(row=4, column=1, sticky="w")
        ttk.Label(controls, text="Time windows").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.signal_windows_var, width=10).grid(row=5, column=1, sticky="w")
        ttk.Button(controls, text="Run signal importance", command=self._run_signal_importance).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(12, 8),
        )
        ttk.Label(
            controls,
            textvariable=self.signal_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=7, column=0, columnspan=2, sticky="w")
        self._refresh_signal_models()

        plot_frame = ttk.LabelFrame(parent, text="Permutation Importance", padding=8)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.signal_plot_container = ttk.Frame(plot_frame)
        self.signal_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_signal_plot_canvas()

    def _build_hyperparameter_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Exploratory Search Setup", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        tuner_task_box = ttk.Combobox(controls, textvariable=self.tuner_task_var, values=["type", "success"], state="readonly")
        tuner_task_box.grid(row=0, column=1, sticky="ew")
        tuner_task_box.bind("<<ComboboxSelected>>", self._on_tuner_task_changed)
        ttk.Label(controls, text="Architecture").grid(row=1, column=0, sticky="w", pady=4)
        self.tuner_architecture_box = ttk.Combobox(controls, textvariable=self.tuner_architecture_var, state="readonly")
        self.tuner_architecture_box.grid(row=1, column=1, sticky="ew")
        self._sync_tuner_architectures()
        ttk.Label(controls, text="Dataset").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.tuner_dataset_var, width=34).grid(row=2, column=1, sticky="ew")
        ttk.Label(controls, text="Trials").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.tuner_trials_var, width=10).grid(row=3, column=1, sticky="w")
        ttk.Label(controls, text="Epochs / trial").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.tuner_epochs_var, width=10).grid(row=4, column=1, sticky="w")
        tuner_scalar_check = ttk.Checkbutton(
            controls,
            text="Use weight and height",
            variable=self.train_use_scalar_features_var,
        )
        tuner_scalar_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._add_tooltip(
            tuner_scalar_check,
            "Utilise ou neutralise les entrees masse/taille pour comparer les essais avec la meme configuration que l'entrainement.",
        )
        ttk.Button(controls, text="Run search", command=self._run_hyperparameter_search).grid(
            row=6,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(12, 8),
        )
        ttk.Label(
            controls,
            textvariable=self.tuner_summary_var,
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=7, column=0, columnspan=2, sticky="w")
        self.tuner_progress = ttk.Progressbar(controls, variable=self.tuner_progress_var, maximum=100)
        self.tuner_progress.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(12, 4))
        ttk.Label(controls, textvariable=self.tuner_progress_text_var, justify=tk.LEFT, wraplength=340).grid(
            row=9,
            column=0,
            columnspan=2,
            sticky="w",
        )

        results = ttk.Frame(parent)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        results.rowconfigure(1, weight=1)
        plot_frame = ttk.LabelFrame(results, text="Top Trials", padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.tuner_plot_container = ttk.Frame(plot_frame)
        self.tuner_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_tuner_plot_canvas()
        log_frame = ttk.LabelFrame(results, text="Search Log", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.tuner_log = scrolledtext.ScrolledText(log_frame, height=12, wrap=tk.WORD)
        self.tuner_log.grid(row=0, column=0, sticky="nsew")

    def _build_window_benchmark_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        controls = ttk.LabelFrame(parent, text="Window Benchmark Setup", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Task").grid(row=0, column=0, sticky="w", pady=4)
        task_box = ttk.Combobox(controls, textvariable=self.window_task_var, values=["type", "success"], state="readonly")
        task_box.grid(row=0, column=1, sticky="ew")
        task_box.bind("<<ComboboxSelected>>", self._on_window_task_changed)
        ttk.Label(controls, text="Architecture").grid(row=1, column=0, sticky="w", pady=4)
        self.window_architecture_box = ttk.Combobox(controls, textvariable=self.window_architecture_var, state="readonly")
        self.window_architecture_box.grid(row=1, column=1, sticky="ew")
        self._sync_window_architectures()
        ttk.Label(controls, text="Dataset").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.window_dataset_var, width=32).grid(row=2, column=1, sticky="ew")
        ttk.Label(controls, text="Frames").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.window_candidates_var, width=18).grid(row=3, column=1, sticky="w")
        ttk.Label(controls, text="Epochs / window").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.window_epochs_var, width=10).grid(row=4, column=1, sticky="w")
        ttk.Button(controls, text="Run window benchmark", command=self._run_window_benchmark).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(12, 8)
        )
        ttk.Label(controls, text="Offset window frames").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.window_offset_frames_var, width=10).grid(row=6, column=1, sticky="w")
        ttk.Label(controls, text="Offsets in segment").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.window_offsets_var, width=18).grid(row=7, column=1, sticky="w")
        ttk.Button(controls, text="Run offset benchmark", command=self._run_window_offset_benchmark).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(8, 8)
        )
        ttk.Label(controls, textvariable=self.window_summary_var, justify=tk.LEFT, wraplength=340).grid(
            row=9, column=0, columnspan=2, sticky="w"
        )
        ttk.Progressbar(controls, variable=self.window_progress_var, maximum=100).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=(12, 4)
        )
        ttk.Label(controls, textvariable=self.window_progress_text_var, justify=tk.LEFT, wraplength=340).grid(
            row=11, column=0, columnspan=2, sticky="w"
        )
        results = ttk.LabelFrame(parent, text="Window Results", padding=8)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        self.window_log = scrolledtext.ScrolledText(results, height=22, wrap=tk.WORD)
        self.window_log.grid(row=0, column=0, sticky="nsew")

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

    def _build_dataset_review_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Training Dataset Review", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(5, weight=1)

        ttk.Label(controls, text="Dataset").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.dataset_review_dataset_var, width=34).grid(row=1, column=0, sticky="ew", pady=(4, 8))
        buttons = ttk.Frame(controls)
        buttons.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(buttons, text="Load trainable jumps", command=self._run_dataset_review_load).grid(row=0, column=0, sticky="w")
        ttk.Button(buttons, text="Exclude selected (x)", command=self._exclude_selected_training_jump).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(
            controls,
            textvariable=self.dataset_review_summary_var,
            justify=tk.LEFT,
            wraplength=320,
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))
        tk.Label(
            controls,
            textvariable=self.dataset_review_feedback_var,
            justify=tk.LEFT,
            wraplength=320,
            fg="#9f2a22",
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

        list_frame = ttk.LabelFrame(controls, text="Trainable jumps", padding=8)
        list_frame.grid(row=5, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.dataset_review_listbox = tk.Listbox(
            list_frame,
            listvariable=self.dataset_review_records_var,
            exportselection=False,
            height=18,
            width=42,
        )
        self.dataset_review_listbox.grid(row=0, column=0, sticky="nsew")
        self.dataset_review_listbox.bind("<<ListboxSelect>>", self._on_dataset_review_selected)

        results = ttk.Frame(parent)
        results.grid(row=0, column=1, sticky="nsew")
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=0)
        results.rowconfigure(1, weight=1)

        details_frame = ttk.LabelFrame(results, text="Selected jump details", padding=8)
        details_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        details_frame.columnconfigure(0, weight=1)
        ttk.Label(
            details_frame,
            textvariable=self.dataset_review_details_var,
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")

        signal_frame = ttk.LabelFrame(results, text="Selected jump signals", padding=8)
        signal_frame.grid(row=1, column=0, sticky="nsew")
        signal_frame.columnconfigure(0, weight=1)
        signal_frame.rowconfigure(0, weight=1)
        self.dataset_review_signal_container = ttk.Frame(signal_frame)
        self.dataset_review_signal_container.grid(row=0, column=0, sticky="nsew")
        self._build_dataset_review_signal_canvas()

    def _build_rotation_audit_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)
        controls = ttk.LabelFrame(parent, text="Turn Estimation Audit", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(4, weight=1)
        ttk.Label(controls, text="Annotation source").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.rotation_audit_root_var, width=34).grid(row=1, column=0, sticky="ew", pady=(4, 8))
        ttk.Button(controls, text="Run turn audit", command=self._run_rotation_audit).grid(row=2, column=0, sticky="w")
        ttk.Label(
            controls,
            textvariable=self.rotation_audit_summary_var,
            justify=tk.LEFT,
            wraplength=320,
        ).grid(row=3, column=0, sticky="w", pady=(8, 8))
        list_frame = ttk.LabelFrame(controls, text="Suspicious estimates", padding=8)
        list_frame.grid(row=4, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        self.rotation_audit_listbox = tk.Listbox(
            list_frame,
            listvariable=self.rotation_audit_records_var,
            exportselection=False,
            height=18,
            width=42,
        )
        self.rotation_audit_listbox.grid(row=0, column=0, sticky="nsew")
        self.rotation_audit_listbox.bind("<<ListboxSelect>>", self._on_rotation_audit_selected)
        results = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        results.grid(row=0, column=1, sticky="nsew")
        selected_panel = ttk.Frame(results)
        selected_panel.columnconfigure(0, weight=1)
        selected_panel.rowconfigure(1, weight=1)
        details_frame = ttk.LabelFrame(selected_panel, text="Selected estimate details", padding=8)
        details_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            details_frame,
            textvariable=self.rotation_audit_details_var,
            justify=tk.LEFT,
            wraplength=760,
        ).grid(row=0, column=0, sticky="w")
        signal_frame = ttk.LabelFrame(selected_panel, text="Selected jump signals", padding=8)
        signal_frame.grid(row=1, column=0, sticky="nsew")
        signal_frame.columnconfigure(0, weight=1)
        signal_frame.rowconfigure(0, weight=1)
        self.rotation_audit_signal_container = ttk.Frame(signal_frame)
        self.rotation_audit_signal_container.grid(row=0, column=0, sticky="nsew")
        self._build_rotation_audit_signal_canvas()

        diagnostics_panel = ttk.Frame(results)
        diagnostics_panel.columnconfigure(0, weight=1)
        diagnostics_panel.rowconfigure(0, weight=1)
        plot_frame = ttk.LabelFrame(diagnostics_panel, text="Turn estimation diagnostics", padding=8)
        plot_frame.grid(row=0, column=0, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        self.rotation_audit_plot_container = ttk.Frame(plot_frame)
        self.rotation_audit_plot_container.grid(row=0, column=0, sticky="nsew")
        self._build_rotation_audit_canvas()
        results.add(selected_panel, weight=3)
        results.add(diagnostics_panel, weight=2)

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

    def _refresh_data_inventory(self) -> None:
        if not hasattr(self, "data_inventory_tree"):
            return
        for item in self.data_inventory_tree.get_children():
            self.data_inventory_tree.delete(item)
        rows = operations.build_data_inventory()
        totals = {
            "new_files": sum(row["new_files"] for row in rows),
            "pending_files": sum(row["pending_files"] for row in rows),
            "stored_segments": sum(row["stored_segments"] for row in rows),
            "trainable_labels": sum(row["trainable_labels"] for row in rows),
            "total_rows": sum(row["total_rows"] for row in rows),
            "trainable_total_rows": sum(row["trainable_total_rows"] for row in rows),
        }
        self.data_inventory_tree.insert(
            "",
            tk.END,
            values=(
                "TOTAL",
                totals["new_files"],
                totals["pending_files"],
                totals["stored_segments"],
                totals["trainable_labels"],
                totals["total_rows"],
                totals["trainable_total_rows"],
                "-",
                "-",
            ),
            tags=("total",),
        )
        for row in rows:
            self.data_inventory_tree.insert(
                "",
                tk.END,
                values=(
                    row["session"],
                    row["new_files"],
                    row["pending_files"],
                    row["stored_segments"],
                    row["trainable_labels"],
                    row["total_rows"],
                    row["trainable_total_rows"],
                    row["workflow_status"] or "-",
                    row["prediction_status"] or "-",
                ),
            )

    def _import_legacy_annotated_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Select legacy jumplist",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        if not messagebox.askyesno(
            "Synergie Tools",
            "Import trainable rows from this legacy jumplist into the total training dataset?",
        ):
            return
        try:
            result = operations.import_legacy_jumplist(path, dataset_path=self.dataset_var.get())
        except Exception as exc:
            messagebox.showerror("Synergie Tools", f"Unable to import legacy jumplist.\n\n{exc}")
            return
        self._refresh_data_inventory()
        self._refresh_training_dataset_stats()
        messagebox.showinfo(
            "Synergie Tools",
            f"Legacy import complete.\n\nRows added: {result['rows_added']}\n"
            f"Previous total jumplist archived at: {result['archive_path']}",
        )

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
            self.new_data_session_suggestion_var.set("No session selected.")
            self.new_data_automation_var.set("Select a session to see the automatic actions that will be applied.")

    def _format_new_data_file_label(self, metadata: dict) -> str:
        if "files" in metadata:
            sensors = ", ".join(file_metadata["sensor_id"] for file_metadata in metadata["files"])
            workflow = metadata.get("workflow") or {}
            status_suffix = f" | {workflow['status']}" if workflow.get("status") else ""
            return (
                f"{metadata['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')} | "
                f"{len(metadata['files'])} sensors [{sensors}]{status_suffix}"
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

    def _refresh_detection_parameter_modes(self) -> None:
        optimized = operations.load_optimized_detection_parameters()
        values = ["Default parameters"]
        if optimized is not None:
            values.append("Optimized parameters")
        self.detection_parameter_mode_box.configure(values=values)
        if self.detection_parameter_mode_var.get() not in values:
            self.detection_parameter_mode_var.set(values[0])
        self._on_detection_parameter_mode_changed()

    def _on_detection_parameter_mode_changed(self, _event=None) -> None:
        if self.detection_parameter_mode_var.get() == "Optimized parameters":
            optimized = operations.load_optimized_detection_parameters()
            if optimized is not None:
                self.threshold_var.set(optimized["threshold"])
                self.sigma_var.set(optimized["smoothing_sigma"])
                self.detection_parameter_summary_var.set(
                    f"balanced error {optimized['balanced_error']:.3f}"
                )
            else:
                self.detection_parameter_mode_var.set("Default parameters")
        if self.detection_parameter_mode_var.get() == "Default parameters":
            self.threshold_var.set(DEFAULT_DETECTION_THRESHOLD)
            self.sigma_var.set(DEFAULT_SMOOTHING_SIGMA)
            self.detection_parameter_summary_var.set("using built-in defaults")
        self._sync_slider_labels()

    def _sync_train_architectures(self) -> None:
        task = self.train_task_var.get()
        options = self.TRAIN_ARCHITECTURES.get(task, [])
        self.train_architecture_box.configure(values=options)
        if self.train_architecture_var.get() not in options and options:
            self.train_architecture_var.set(options[0])
        self._sync_train_parameter_profile()

    def _sync_train_parameter_profile(self) -> None:
        optimized = operations.load_optimized_model_parameters(
            self.train_task_var.get(),
            self.train_architecture_var.get(),
        )
        if self.train_parameter_profile_var.get() == "Optimized parameters" and optimized is None:
            self.train_parameter_profile_var.set("Default parameters")
        self._apply_train_parameter_profile()

    def _apply_train_parameter_profile(self) -> None:
        self._clear_training_parameter_fields()
        if self.train_parameter_profile_var.get() != "Optimized parameters":
            self.train_parameter_profile_summary_var.set("Using architecture defaults.")
            return
        optimized = operations.load_optimized_model_parameters(
            self.train_task_var.get(),
            self.train_architecture_var.get(),
        )
        if optimized is None:
            self.train_parameter_profile_var.set("Default parameters")
            self.train_parameter_profile_summary_var.set("No optimized profile saved for this task and architecture.")
            return
        parameters = optimized["parameters"]
        self.train_batch_size_var.set(_string_or_empty(parameters.get("batch_size")))
        self.train_learning_rate_var.set(_string_or_empty(parameters.get("learning_rate")))
        self.train_dropout_var.set(_string_or_empty(parameters.get("dropout")))
        self.train_filters_var.set(_string_or_empty(parameters.get("filters", parameters.get("first_units"))))
        self.train_modules_var.set(_string_or_empty(parameters.get("modules", parameters.get("num_transformer_blocks"))))
        self.train_parameter_profile_summary_var.set(
            f"Optimized profile saved from Models - Tune: val_accuracy={optimized['best_val_accuracy']:.3f}"
        )

    def _clear_training_parameter_fields(self) -> None:
        self.train_batch_size_var.set("")
        self.train_learning_rate_var.set("")
        self.train_dropout_var.set("")
        self.train_filters_var.set("")
        self.train_modules_var.set("")

    def _format_training_dataset_stats(self, stats: dict) -> str:
        class_counts = ", ".join(f"{label}: {count}" for label, count in stats["class_counts"].items())
        class_weight = ", ".join(
            f"{label}: {weight:.3f}" for label, weight in stats.get("recommended_class_weight", {}).items()
        ) or "n/a"
        mirror_text = "yes" if stats["augment_mirror"] else "no"
        stratified_text = "yes" if stats.get("stratified_split_possible") else "fallback to random split"
        duplicate_text = (
            f"\nWARNING: {stats['duplicate_rows']} duplicate rows across "
            f"{len(stats['duplicate_paths'])} repeated paths."
            if stats.get("has_duplicates")
            else "\nDuplicate path check: ok"
        )
        dataset_state = operations.load_training_dataset_state(self.dataset_var.get())
        dataset_change_text = ""
        if dataset_state.get("retraining_recommended"):
            dataset_change_text = (
                f"\nDataset changed at {dataset_state.get('changed_at', 'unknown')}: "
                f"{dataset_state.get('rows_added', 0)} new rows added. Retraining recommended."
            )
        return (
            f"Training set stats ({stats['task']}): "
            f"{stats['base_samples']} labelled jumps, {stats['unique_skaters']} skaters, "
            f"{stats['effective_samples']} effective samples with mirror augmentation ({mirror_text}).\n"
            f"Class distribution: {class_counts}\n"
            f"Recommended class weights: {class_weight}\n"
            f"Stratified train/val split possible: {stratified_text}"
            f"{duplicate_text}"
            f"{dataset_change_text}"
        )

    def _format_training_quality_summary(self, summary: dict) -> str:
        def metric(value) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        saved_model = summary.get("saved_model") or {}
        saved_model_text = saved_model.get("id", "not archived yet")
        class_weight = ", ".join(
            f"{label}: {weight:.3f}" for label, weight in (summary.get("class_weight") or {}).items()
        ) or "n/a"
        stratified_text = "yes" if summary.get("stratified_split_used") else "no"
        return (
            f"Model quality: test_acc={metric(summary.get('test_accuracy'))}, "
            f"best_val_acc={metric(summary.get('best_val_accuracy'))}, "
            f"final_val_acc={metric(summary.get('final_val_accuracy'))}, "
            f"epochs={summary.get('epochs_ran', 0)}, "
            f"test_samples={summary.get('test_samples', 0)}\n"
            f"Training config: stratified_split={stratified_text}, class_weight={class_weight}\n"
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
            self.use_pretrained_var.set(False)
            self._draw_placeholder_confusion_matrix()

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

    def _refresh_process_prediction_models(self) -> None:
        type_models = operations.list_pretrained_models_by_performance("type")
        success_models = operations.list_pretrained_models_by_performance("success")
        type_labels = [operations.format_pretrained_model_label(item) for item in type_models]
        success_labels = [operations.format_pretrained_model_label(item) for item in success_models]
        self.process_type_model_box.configure(values=type_labels)
        self.process_success_model_box.configure(values=success_labels)
        if type_labels and self.process_type_model_var.get() not in type_labels:
            self.process_type_model_var.set(type_labels[0])
        if success_labels and self.process_success_model_var.get() not in success_labels:
            self.process_success_model_var.set(success_labels[0])

    def _process_prediction_model_path(self, task: str, selected_label: str) -> str | None:
        for item in operations.list_pretrained_models_by_performance(task):
            if operations.format_pretrained_model_label(item) == selected_label:
                return item["path"]
        return None

    def _selected_pretrained_model_id(self) -> str | None:
        selected_label = self.pretrained_model_var.get()
        if not selected_label:
            return None
        for model in operations.list_pretrained_training_models(task=self.train_task_var.get(), compatible_only=True):
            if operations.format_pretrained_model_label(model) == selected_label:
                return model["id"]
        return None

    def _delete_selected_pretrained_model(self) -> None:
        model_id = self._selected_pretrained_model_id()
        if not model_id:
            messagebox.showinfo("Synergie Tools", "Select a compatible archived model first.")
            return
        if not messagebox.askyesno(
            "Synergie Tools",
            f"Delete archived model '{model_id}' from the registry and disk?\n\nThis cannot be undone.",
        ):
            return
        try:
            removed = operations.delete_pretrained_training_model(model_id)
        except Exception as exc:
            messagebox.showerror("Synergie Tools", f"Unable to delete model.\n\n{exc}")
            return
        self._refresh_pretrained_models()
        self._sync_pretrained_controls()
        messagebox.showinfo("Synergie Tools", f"Deleted model:\n{removed['label']}")

    def _training_overrides_from_gui(self) -> tuple[dict, int | None]:
        optimized = None
        if self.train_parameter_profile_var.get() == "Optimized parameters":
            optimized = operations.load_optimized_model_parameters(
                self.train_task_var.get(),
                self.train_architecture_var.get(),
            )
        optimized_parameters = dict(optimized["parameters"]) if optimized else {}
        batch_size_value = optimized_parameters.pop("batch_size", None)
        overrides: dict = optimized_parameters
        learning_rate = self.train_learning_rate_var.get().strip()
        dropout = self.train_dropout_var.get().strip()
        filters = self.train_filters_var.get().strip()
        modules = self.train_modules_var.get().strip()
        batch_size = self.train_batch_size_var.get().strip()
        architecture = self.train_architecture_var.get()
        task = self.train_task_var.get()
        if learning_rate:
            overrides["learning_rate"] = float(learning_rate)
        if dropout:
            overrides["dropout"] = float(dropout)
        if filters:
            if task == "success" and architecture == "lstm":
                overrides["first_units"] = int(filters)
            else:
                overrides["filters"] = int(filters)
        if modules:
            if task == "type" and architecture == "transformer":
                overrides["num_transformer_blocks"] = int(modules)
            elif task == "type" and architecture == "inceptiontime":
                overrides["modules"] = int(modules)
        return overrides, (int(batch_size) if batch_size else (int(batch_size_value) if batch_size_value else None))

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

    def _draw_placeholder_confusion_matrix(self) -> None:
        if self.train_confusion_ax is None:
            return
        if self._train_confusion_colorbar is not None:
            self._train_confusion_colorbar.remove()
            self._train_confusion_colorbar = None
        self.train_confusion_ax.clear()
        self.train_confusion_ax.set_title("Confusion Matrix")
        self.train_confusion_ax.text(
            0.5,
            0.5,
            "Run a training to see the confusion matrix",
            ha="center",
            va="center",
            transform=self.train_confusion_ax.transAxes,
        )
        self.train_confusion_ax.set_xticks([])
        self.train_confusion_ax.set_yticks([])
        self.train_confusion_figure.tight_layout()
        self.train_confusion_canvas.draw_idle()

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

    def _draw_placeholder_signal_plot(self) -> None:
        if self.signal_axes is None:
            return
        channel_ax, scalar_ax, time_ax = self.signal_axes
        channel_ax.clear()
        scalar_ax.clear()
        time_ax.clear()
        channel_ax.set_title("Signal importance")
        scalar_ax.set_title("Scalar importance")
        time_ax.set_title("Temporal importance")
        channel_ax.text(0.5, 0.5, "Run signal importance", ha="center", va="center", transform=channel_ax.transAxes)
        scalar_ax.text(0.5, 0.5, "Permutation drop for weight / height", ha="center", va="center", transform=scalar_ax.transAxes)
        time_ax.text(0.5, 0.5, "Permutation drop by time window", ha="center", va="center", transform=time_ax.transAxes)
        self.signal_figure.tight_layout()
        self.signal_canvas.draw_idle()

    def _draw_signal_importance(self, analysis: dict) -> None:
        if self.signal_axes is None:
            return
        channel_ax, scalar_ax, time_ax = self.signal_axes
        channel_ax.clear()
        scalar_ax.clear()
        time_ax.clear()
        channels = analysis.get("channel_importance", [])
        scalars = analysis.get("scalar_importance", [])
        temporal = analysis.get("temporal_importance", [])
        channel_labels = [item["label"] for item in channels]
        channel_values = [item["mean_drop"] for item in channels]
        channel_ax.bar(channel_labels, channel_values, color="teal", alpha=0.85)
        channel_ax.set_title("Signal importance (balanced accuracy drop)")
        channel_ax.set_ylabel("Drop")
        channel_ax.tick_params(axis="x", rotation=30)

        scalar_labels = [item["label"] for item in scalars]
        scalar_values = [item["mean_drop"] for item in scalars]
        scalar_bars = scalar_ax.bar(scalar_labels, scalar_values, color="slateblue", alpha=0.85)
        scalar_ax.set_title("Scalar importance (balanced accuracy drop)")
        scalar_ax.set_ylabel("Drop")
        scalar_ax.axhline(0.0, color="0.35", linewidth=0.8)
        if scalar_values:
            max_abs_scalar = max(abs(value) for value in scalar_values)
            if max_abs_scalar > 0:
                scalar_ax.set_ylim(0.0, max_abs_scalar * 1.35)
            else:
                scalar_ax.set_ylim(-0.05, 0.05)
                scalar_ax.text(
                    0.5,
                    0.5,
                    "No measurable drop",
                    ha="center",
                    va="center",
                    transform=scalar_ax.transAxes,
                )
            for bar, value in zip(scalar_bars, scalar_values):
                scalar_ax.annotate(
                    f"{value:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        temporal_labels = [f"{item['start_frame']}-{item['end_frame']}" for item in temporal]
        temporal_values = [item["mean_drop"] for item in temporal]
        time_ax.bar(temporal_labels, temporal_values, color="darkorange", alpha=0.85)
        time_ax.set_title("Temporal importance by frame window")
        time_ax.set_xlabel("Frames")
        time_ax.set_ylabel("Drop")
        self.signal_figure.tight_layout()
        self.signal_canvas.draw_idle()

    def _draw_placeholder_tuner_plot(self) -> None:
        if self.tuner_ax is None:
            return
        self.tuner_ax.clear()
        self.tuner_ax.set_title("Top trials")
        self.tuner_ax.text(0.5, 0.5, "Run a search to compare candidates", ha="center", va="center", transform=self.tuner_ax.transAxes)
        self.tuner_ax.set_xticks([])
        self.tuner_ax.set_yticks([])
        self.tuner_figure.tight_layout()
        self.tuner_canvas.draw_idle()

    def _draw_hyperparameter_results(self, summary: dict) -> None:
        if self.tuner_ax is None:
            return
        results = summary.get("results", [])[:10]
        if not results:
            self._draw_placeholder_tuner_plot()
            return
        self.tuner_ax.clear()
        labels = [f"T{item['trial']}" for item in results]
        values = [item["best_val_accuracy"] for item in results]
        self.tuner_ax.bar(labels, values, color="slateblue", alpha=0.85)
        self.tuner_ax.set_title("Best validation accuracy by trial")
        self.tuner_ax.set_ylabel("Val accuracy")
        self.tuner_ax.set_ylim(0, max(1.0, max(values) * 1.05))
        self.tuner_figure.tight_layout()
        self.tuner_canvas.draw_idle()

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

    def _draw_placeholder_rotation_audit(self) -> None:
        if self.rotation_audit_ax is None:
            return
        self.rotation_audit_ax.clear()
        self.rotation_audit_error_ax.clear()
        self.rotation_audit_rule_ax.clear()
        self.rotation_audit_ax.set_title("Confusion matrix")
        self.rotation_audit_ax.text(
            0.5,
            0.5,
            "Run turn audit after labelled turns exist",
            ha="center",
            va="center",
            transform=self.rotation_audit_ax.transAxes,
        )
        self.rotation_audit_ax.set_xticks([])
        self.rotation_audit_ax.set_yticks([])
        self.rotation_audit_error_ax.set_title("Error by type")
        self.rotation_audit_error_ax.text(
            0.5,
            0.5,
            "No labelled turns yet",
            ha="center",
            va="center",
            transform=self.rotation_audit_error_ax.transAxes,
        )
        self.rotation_audit_error_ax.set_xticks([])
        self.rotation_audit_error_ax.set_yticks([])
        self.rotation_audit_rule_ax.set_title("Strategies")
        self.rotation_audit_rule_ax.text(
            0.5,
            0.5,
            "No rule comparison yet",
            ha="center",
            va="center",
            transform=self.rotation_audit_rule_ax.transAxes,
        )
        self.rotation_audit_rule_ax.set_xticks([])
        self.rotation_audit_rule_ax.set_yticks([])
        self.rotation_audit_figure.tight_layout()
        self.rotation_audit_canvas.draw_idle()

    def _draw_placeholder_dataset_review_signal(self) -> None:
        if self.dataset_review_signal_ax is None:
            return
        if self.dataset_review_signal_acc_ax is not None:
            self.dataset_review_signal_acc_ax.remove()
            self.dataset_review_signal_acc_ax = None
        self.dataset_review_signal_ax.clear()
        self.dataset_review_angle_ax.clear()
        self.dataset_review_signal_ax.set_title("Selected training jump signals")
        self.dataset_review_signal_ax.text(
            0.5,
            0.5,
            "Load the training dataset and select a jump",
            ha="center",
            va="center",
            transform=self.dataset_review_signal_ax.transAxes,
        )
        self.dataset_review_signal_ax.set_xticks([])
        self.dataset_review_signal_ax.set_yticks([])
        self.dataset_review_angle_ax.text(
            0.5,
            0.5,
            "Cumulative rotation will appear here",
            ha="center",
            va="center",
            transform=self.dataset_review_angle_ax.transAxes,
        )
        self.dataset_review_angle_ax.set_xticks([])
        self.dataset_review_angle_ax.set_yticks([])
        self.dataset_review_signal_figure.tight_layout()
        self.dataset_review_signal_canvas.draw_idle()

    def _draw_dataset_review_signal(self, signal: dict | None) -> None:
        if self.dataset_review_signal_ax is None or signal is None:
            self._draw_placeholder_dataset_review_signal()
            return
        frame = signal["frame"]
        if self.dataset_review_signal_acc_ax is not None:
            self.dataset_review_signal_acc_ax.remove()
        self.dataset_review_signal_ax.clear()
        self.dataset_review_angle_ax.clear()
        self.dataset_review_signal_acc_ax = self.dataset_review_signal_ax.twinx()
        x = frame["ms"] if "ms" in frame else list(range(len(frame)))
        gyro_line = self.dataset_review_signal_ax.plot(x, frame["Gyr_X"], color="#1f77b4", label="Gyr_X", linewidth=1.1)[0]
        handles = [gyro_line]
        labels = ["Gyr_X"]
        if "Acc_X" in frame:
            acc_line = self.dataset_review_signal_acc_ax.plot(
                x,
                frame["Acc_X"],
                color="#ff7f0e",
                label="Acc_X",
                linewidth=1.0,
                alpha=0.85,
            )[0]
            handles.append(acc_line)
            labels.append("Acc_X")
        takeoff_x = x.iloc[signal["takeoff_index"]] if hasattr(x, "iloc") else x[signal["takeoff_index"]]
        landing_x = x.iloc[signal["landing_index"]] if hasattr(x, "iloc") else x[signal["landing_index"]]
        self.dataset_review_signal_ax.axvline(takeoff_x, color="black", linestyle="--", linewidth=1.0, label="Takeoff")
        self.dataset_review_signal_ax.axvline(landing_x, color="black", linestyle=":", linewidth=1.0, label="Landing")
        handles.extend(self.dataset_review_signal_ax.lines[-2:])
        labels.extend(["Takeoff", "Landing"])
        self.dataset_review_signal_ax.set_title("Vertical gyro and acceleration")
        self.dataset_review_signal_ax.set_xlabel("ms" if "ms" in frame else "frame")
        self.dataset_review_signal_ax.set_ylabel("Gyroscope")
        self.dataset_review_signal_acc_ax.set_ylabel("Acceleration")
        self.dataset_review_signal_ax.legend(handles, labels, loc="upper right", fontsize=8)
        angle = self._cumulative_rotation_turns(frame)
        self.dataset_review_angle_ax.plot(x, angle, color="#2f7d32", linewidth=1.2, label="Integrated Gyr_X")
        self.dataset_review_angle_ax.axvline(takeoff_x, color="black", linestyle="--", linewidth=1.0)
        self.dataset_review_angle_ax.axvline(landing_x, color="black", linestyle=":", linewidth=1.0)
        self.dataset_review_angle_ax.set_title("Cumulative vertical rotation")
        self.dataset_review_angle_ax.set_xlabel("ms" if "ms" in frame else "frame")
        self.dataset_review_angle_ax.set_ylabel("turns")
        self.dataset_review_angle_ax.legend(loc="upper left", fontsize=8)
        self.dataset_review_signal_figure.tight_layout()
        self.dataset_review_signal_canvas.draw_idle()

    def _draw_rotation_audit(self, analysis: dict) -> None:
        if self.rotation_audit_ax is None:
            return
        matrix = analysis.get("confusion_matrix", [])
        labels = analysis.get("labels", [])
        if not matrix:
            self._draw_placeholder_rotation_audit()
            return
        self.rotation_audit_ax.clear()
        self.rotation_audit_error_ax.clear()
        self.rotation_audit_rule_ax.clear()
        image = self.rotation_audit_ax.imshow(matrix, cmap="Blues")
        self.rotation_audit_ax.set_title("Annotated vs estimated")
        self.rotation_audit_ax.set_xlabel("Estimated turns")
        self.rotation_audit_ax.set_ylabel("Annotated turns")
        self.rotation_audit_ax.set_xticks(range(len(labels)), labels)
        self.rotation_audit_ax.set_yticks(range(len(labels)), labels)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if value:
                    self.rotation_audit_ax.text(column_index, row_index, str(value), ha="center", va="center", fontsize=8)
        if getattr(self, "_rotation_audit_colorbar", None) is not None:
            self._rotation_audit_colorbar.remove()
        self._rotation_audit_colorbar = self.rotation_audit_figure.colorbar(image, ax=self.rotation_audit_ax, fraction=0.046, pad=0.04)
        type_summary = analysis.get("type_summary", [])
        type_labels = [item["label"] for item in type_summary]
        grouped_errors = [
            [record["signed_error"] for record in analysis["records"] if record["type_label"] == label]
            for label in type_labels
        ]
        self.rotation_audit_error_ax.boxplot(grouped_errors, tick_labels=type_labels, patch_artist=True)
        self.rotation_audit_error_ax.axhline(0.0, color="black", linewidth=1.0)
        self.rotation_audit_error_ax.set_title("Measured - annotated")
        self.rotation_audit_error_ax.set_ylabel("turns")
        self.rotation_audit_error_ax.tick_params(axis="x", rotation=30)
        rule_summary = analysis.get("strategy_summary", [])
        rule_labels = [self._rotation_strategy_label(item["label"]) for item in rule_summary]
        rule_scores = [item["exact_accuracy"] for item in rule_summary]
        bars = self.rotation_audit_rule_ax.barh(rule_labels, rule_scores, color="#5b8ff9")
        if len(bars) >= 2:
            bars[1].set_color("#2f7d32")
        self.rotation_audit_rule_ax.set_xlim(0.0, 1.0)
        self.rotation_audit_rule_ax.set_xlabel("exact accuracy")
        self.rotation_audit_rule_ax.set_title("Strategies")
        for bar, item in zip(bars, rule_summary):
            self.rotation_audit_rule_ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{item['exact_accuracy']:.3f}",
                ha="left",
                va="center",
                fontsize=8,
            )
        self.rotation_audit_figure.tight_layout()
        self.rotation_audit_canvas.draw_idle()

    def _draw_placeholder_rotation_audit_signal(self) -> None:
        if self.rotation_audit_signal_ax is None:
            return
        if self.rotation_audit_signal_acc_ax is not None:
            self.rotation_audit_signal_acc_ax.remove()
            self.rotation_audit_signal_acc_ax = None
        self.rotation_audit_signal_ax.clear()
        self.rotation_audit_angle_ax.clear()
        self.rotation_audit_signal_ax.set_title("Selected jump signals")
        self.rotation_audit_signal_ax.text(
            0.5,
            0.5,
            "Select a suspicious turn estimate",
            ha="center",
            va="center",
            transform=self.rotation_audit_signal_ax.transAxes,
        )
        self.rotation_audit_signal_ax.set_xticks([])
        self.rotation_audit_signal_ax.set_yticks([])
        self.rotation_audit_angle_ax.text(
            0.5,
            0.5,
            "Cumulative rotation will appear here",
            ha="center",
            va="center",
            transform=self.rotation_audit_angle_ax.transAxes,
        )
        self.rotation_audit_angle_ax.set_xticks([])
        self.rotation_audit_angle_ax.set_yticks([])
        self.rotation_audit_signal_figure.tight_layout()
        self.rotation_audit_signal_canvas.draw_idle()

    def _draw_rotation_audit_signal(self, signal: dict | None) -> None:
        if self.rotation_audit_signal_ax is None or signal is None:
            self._draw_placeholder_rotation_audit_signal()
            return
        frame = signal["frame"]
        if self.rotation_audit_signal_acc_ax is not None:
            self.rotation_audit_signal_acc_ax.remove()
        self.rotation_audit_signal_ax.clear()
        self.rotation_audit_angle_ax.clear()
        self.rotation_audit_signal_acc_ax = self.rotation_audit_signal_ax.twinx()
        x = frame["ms"] if "ms" in frame else list(range(len(frame)))
        gyro_line = self.rotation_audit_signal_ax.plot(x, frame["Gyr_X"], color="#1f77b4", label="Gyr_X", linewidth=1.1)[0]
        handles = [gyro_line]
        labels = ["Gyr_X"]
        if "Acc_X" in frame:
            acc_line = self.rotation_audit_signal_acc_ax.plot(
                x,
                frame["Acc_X"],
                color="#ff7f0e",
                label="Acc_X",
                linewidth=1.0,
                alpha=0.85,
            )[0]
            handles.append(acc_line)
            labels.append("Acc_X")
        takeoff_x = x.iloc[signal["takeoff_index"]] if hasattr(x, "iloc") else x[signal["takeoff_index"]]
        landing_x = x.iloc[signal["landing_index"]] if hasattr(x, "iloc") else x[signal["landing_index"]]
        self.rotation_audit_signal_ax.axvline(takeoff_x, color="black", linestyle="--", linewidth=1.0, label="Takeoff")
        self.rotation_audit_signal_ax.axvline(landing_x, color="black", linestyle=":", linewidth=1.0, label="Landing")
        handles.extend(self.rotation_audit_signal_ax.lines[-2:])
        labels.extend(["Takeoff", "Landing"])
        self.rotation_audit_signal_ax.set_title("Vertical gyro and acceleration")
        self.rotation_audit_signal_ax.set_xlabel("ms" if "ms" in frame else "frame")
        self.rotation_audit_signal_ax.set_ylabel("Gyroscope")
        self.rotation_audit_signal_acc_ax.set_ylabel("Acceleration")
        self.rotation_audit_signal_ax.legend(handles, labels, loc="upper right", fontsize=8)
        angle = self._cumulative_rotation_turns(frame)
        self.rotation_audit_angle_ax.plot(x, angle, color="#2f7d32", linewidth=1.2, label="Integrated Gyr_X")
        self.rotation_audit_angle_ax.axvline(takeoff_x, color="black", linestyle="--", linewidth=1.0)
        self.rotation_audit_angle_ax.axvline(landing_x, color="black", linestyle=":", linewidth=1.0)
        self.rotation_audit_angle_ax.set_title("Cumulative vertical rotation")
        self.rotation_audit_angle_ax.set_xlabel("ms" if "ms" in frame else "frame")
        self.rotation_audit_angle_ax.set_ylabel("turns")
        self.rotation_audit_angle_ax.legend(loc="upper left", fontsize=8)
        self.rotation_audit_signal_figure.tight_layout()
        self.rotation_audit_signal_canvas.draw_idle()

    def _cumulative_rotation_turns(self, frame):
        import numpy as np

        gyro = frame["Gyr_X"].to_numpy(dtype="float64")
        timestamps = frame["SampleTimeFine"].to_numpy(dtype="float64")
        dt_seconds = np.diff(timestamps, prepend=timestamps[0]) / 1e6
        valid = np.isfinite(gyro) & np.isfinite(dt_seconds)
        increments = np.zeros(len(frame), dtype="float64")
        increments[valid] = (gyro[valid] * dt_seconds[valid]) / 360.0
        return np.cumsum(increments)

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

    def _draw_confusion_matrix(self, summary: dict) -> None:
        if self.train_confusion_ax is None:
            return

        matrix = summary.get("confusion_matrix") or []
        if not matrix:
            self._draw_placeholder_confusion_matrix()
            return

        import numpy as np

        axis = self.train_confusion_ax
        axis.clear()
        matrix_array = np.array(matrix, dtype=float)
        image = axis.imshow(matrix_array, cmap="YlOrRd")
        labels = self._confusion_matrix_labels(summary, len(matrix_array))
        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.set_yticklabels(labels)

        max_value = float(matrix_array.max()) if matrix_array.size else 0.0
        threshold = max_value / 2.0 if max_value > 0 else 0.0
        for row_index in range(matrix_array.shape[0]):
            for column_index in range(matrix_array.shape[1]):
                value = matrix_array[row_index, column_index]
                axis.text(
                    column_index,
                    row_index,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=9,
                    fontweight="bold",
                )

        if getattr(self, "_train_confusion_colorbar", None) is not None:
            self._train_confusion_colorbar.remove()
        self._train_confusion_colorbar = self.train_confusion_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        self._train_confusion_colorbar.set_label("Samples")
        self.train_confusion_figure.tight_layout()
        self.train_confusion_canvas.draw_idle()

    def _confusion_matrix_labels(self, summary: dict, matrix_size: int) -> list[str]:
        task = summary.get("task", self.train_task_var.get())
        if task == "success" and matrix_size == 2:
            return ["Fall", "Success"]
        if task == "type":
            type_labels = {
                0: "Toe",
                1: "Flip",
                2: "Lutz",
                3: "Sal",
                4: "Loop",
                5: "Axel",
            }
            return [type_labels.get(index, str(index)) for index in range(matrix_size)]
        return [str(index) for index in range(matrix_size)]

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

    def _format_model_audit_summary(self, audit: list[dict]) -> str:
        inferable = sum(1 for item in audit if item.get("can_infer"))
        trainable = sum(1 for item in audit if item.get("can_resume_training"))
        legacy = sum(1 for item in audit if item.get("format") == "legacy_saved_model")
        return f"Models checked: {len(audit)} | inference-ready: {inferable} | training-reloadable: {trainable} | legacy inference-only: {legacy}"

    def _format_signal_summary(self, analysis: dict) -> str:
        top_channels = analysis.get("channel_importance", [])[:3]
        channel_text = ", ".join(f"{item['label']} ({item['mean_drop']:.3f})" for item in top_channels) or "n/a"
        scalar_text = ", ".join(
            f"{item['label']} ({item['mean_drop']:.3f})"
            for item in analysis.get("scalar_importance", [])
        ) or "n/a"
        top_window = max(analysis.get("temporal_importance", []), key=lambda item: item["mean_drop"], default=None)
        window_text = (
            f"{top_window['start_frame']}-{top_window['end_frame']} ({top_window['mean_drop']:.3f})"
            if top_window is not None
            else "n/a"
        )
        return (
            f"Validation samples: {analysis['validation_samples']} | "
            f"accuracy={analysis['baseline_accuracy']:.3f} | "
            f"balanced_accuracy={analysis['baseline_balanced_accuracy']:.3f}\n"
            f"Top signals: {channel_text}\n"
            f"Scalars: {scalar_text}\n"
            f"Most informative window: frames {window_text}"
        )

    def _format_tuner_summary(self, summary: dict) -> str:
        best = summary.get("best_trial")
        if not best:
            return "No completed trial."
        return (
            f"Best trial: T{best['trial']} | best_val_accuracy={best['best_val_accuracy']:.3f} | "
            f"epochs={best['epochs_ran']}\n"
            f"Parameters: {best['parameters']}\n"
            f"{summary.get('note', '')}"
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

    def _apply_dataset_review_records(self, records: list[dict], dataset_path: str) -> None:
        self.dataset_review_records = records
        self.dataset_review_summary_var.set(
            f"Trainable jumps: {len(records)}\n"
            f"Shortcut: x = exclude selected jump from future training"
        )
        labels = [
            f"{record['row_index']:04d} | {Path(str(record.get('path', ''))).name} | "
            f"type {record.get('type')} | success {record.get('success')}"
            for record in records
        ]
        self.dataset_review_records_var.set(labels)
        if labels:
            self.dataset_review_listbox.selection_clear(0, tk.END)
            self.dataset_review_listbox.selection_set(0)
            self._on_dataset_review_selected()
        else:
            self.dataset_review_details_var.set(f"No trainable jumps left in {dataset_path}.")
            self._draw_placeholder_dataset_review_signal()
        self.status_var.set(f"Training dataset review ready: {len(records)} trainable jumps")

    def _run_dataset_review_load(self) -> None:
        dataset_path = self.dataset_review_dataset_var.get().strip()
        if not dataset_path:
            messagebox.showwarning("Synergie Tools", "Select a dataset folder first.")
            return
        self.dataset_review_summary_var.set("Loading trainable jumps...")
        self.dataset_review_feedback_var.set("")
        self.dataset_review_records_var.set([])
        self.dataset_review_details_var.set("No training jump selected.")
        self._draw_placeholder_dataset_review_signal()

        def action() -> None:
            records = operations.load_training_dataset_rows(dataset_path)
            self.root.after(0, lambda: self._apply_dataset_review_records(records, dataset_path))

        self._run_in_thread(action, "Unable to load training dataset rows.")

    def _selected_dataset_review_record(self) -> dict | None:
        selection = self.dataset_review_listbox.curselection()
        if not selection:
            return None
        index = selection[0]
        if index >= len(self.dataset_review_records):
            return None
        return self.dataset_review_records[index]

    def _on_dataset_review_selected(self, _event=None) -> None:
        record = self._selected_dataset_review_record()
        if record is None:
            self.dataset_review_details_var.set("No training jump selected.")
            self._draw_placeholder_dataset_review_signal()
            return
        self.dataset_review_details_var.set(
            f"Row: {record['row_index']} | Type: {record.get('type')} | Success: {record.get('success')}\n"
            f"Skater: {record.get('skater', '')} | Path: {record.get('path', '')}\n"
            f"Shortcut: press x to exclude this jump from future training."
        )
        self._draw_dataset_review_signal(operations.load_turn_audit_signal(record["path"]))

    def _exclude_selected_training_jump(self) -> None:
        record = self._selected_dataset_review_record()
        if record is None:
            messagebox.showwarning("Synergie Tools", "Select a training jump first.")
            return
        dataset_path = self.dataset_review_dataset_var.get().strip()
        selected_index = self.dataset_review_listbox.curselection()[0]
        operations.exclude_training_dataset_row(dataset_path, record["row_index"], excluded_reason="weird_signal")
        records = operations.load_training_dataset_rows(dataset_path)
        self._apply_dataset_review_records(records, dataset_path)
        self.dataset_review_feedback_var.set(
            f"Excluded row {record['row_index']} as weird signal. The jump will not be used in future training."
        )
        if records:
            next_index = min(selected_index, len(records) - 1)
            self.dataset_review_listbox.selection_clear(0, tk.END)
            self.dataset_review_listbox.selection_set(next_index)
            self._on_dataset_review_selected()
        self.root.after(3000, lambda: self.dataset_review_feedback_var.set(""))
        self.status_var.set(f"Excluded training jump row {record['row_index']} as weird signal")

    def _apply_quality_analysis(self, analysis: dict) -> None:
        self.quality_analysis = analysis
        self.quality_summary_var.set(self._format_quality_summary(analysis))
        self._refresh_quality_suspicious_list()
        self._draw_quality_analysis(analysis)
        self.status_var.set(
            f"Quality scan ready: {analysis['suspicious_count']} suspicious jumps over {analysis['total_labelled_jumps']} labelled jumps"
        )

    def _run_quality_scan(self) -> None:
        dataset_path = self.quality_dataset_var.get().strip()
        if not dataset_path:
            messagebox.showwarning("Synergie Tools", "Select a dataset folder first.")
            return

        self.status_var.set("Running quality control scan...")
        self.quality_summary_var.set("Quality scan in progress...")
        self.quality_details_var.set("No suspicious jump selected.")
        self.quality_suspicious_var.set([])
        self._draw_placeholder_quality_plot()

        def action() -> None:
            analysis = operations.analyze_jump_quality(dataset_path)
            self.root.after(0, lambda: self._apply_quality_analysis(analysis))

        self._run_in_thread(action, "Unable to run the quality scan.")

    def _refresh_rotation_audit_records(self) -> None:
        if not self.rotation_audit_analysis:
            self.rotation_audit_records_var.set([])
            return
        records = self.rotation_audit_analysis.get("suspicious_records", [])
        labels = [
            f"{index + 1:03d} | {record['type_label']} | "
            f"annotated {record['annotated_turns']:g} vs estimated {record['estimated_turns']:g}"
            for index, record in enumerate(records)
        ]
        self.rotation_audit_records_var.set(labels)
        self.root.after(0, lambda: self._color_rotation_audit_records(records))
        if labels:
            self.rotation_audit_listbox.selection_clear(0, tk.END)
            self.rotation_audit_listbox.selection_set(0)
            self._on_rotation_audit_selected()
        else:
            self.rotation_audit_details_var.set("No suspicious turn estimate selected.")
            self._draw_placeholder_rotation_audit_signal()

    def _color_rotation_audit_records(self, records: list[dict]) -> None:
        for index, record in enumerate(records):
            has_estimation_error = record.get("annotated_turns") != record.get("estimated_turns")
            self.rotation_audit_listbox.itemconfig(
                index,
                foreground="firebrick" if has_estimation_error else "black",
            )

    def _on_rotation_audit_selected(self, _event=None) -> None:
        if not self.rotation_audit_analysis:
            self.rotation_audit_details_var.set("No suspicious turn estimate selected.")
            self._draw_placeholder_rotation_audit_signal()
            return
        selection = self.rotation_audit_listbox.curselection()
        records = self.rotation_audit_analysis.get("suspicious_records", [])
        if not selection or selection[0] >= len(records):
            self.rotation_audit_details_var.set("No suspicious turn estimate selected.")
            self._draw_placeholder_rotation_audit_signal()
            return
        record = records[selection[0]]
        self.rotation_audit_details_var.set(
            f"Type: {record['type_label']} | path: {record['path']}\n"
            f"Annotated: {record['annotated_turns']:g} | measured IMU: {record['measured_rotation']:.2f} | "
            f"estimated: {record['estimated_turns']:g}\n"
            f"Signed error: {record['signed_error']:+.2f} | reasons: {', '.join(record['reasons'])}\n"
            f"Source file: {record['source_file']}"
        )
        self._draw_rotation_audit_signal(operations.load_turn_audit_signal(record["path"]))

    def _format_rotation_audit_summary(self, analysis: dict) -> str:
        if analysis["labelled_jumps"] == 0:
            return (
                f"Scanned {analysis['scanned_files']} annotation files, but no completed labelled jumps with turns were found yet.\n"
                "Complete annotations first; this audit will then compare measured IMU rotation with human turn labels."
            )
        return (
            f"Labelled jumps: {analysis['labelled_jumps']} | suspicious: {len(analysis['suspicious_records'])}\n"
            f"Exact turn accuracy: {analysis['exact_accuracy']:.3f} | mean absolute error: {analysis['mean_absolute_error']:.3f}\n"
            f"Best simple rule: {analysis['rounding_rule_summary'][0]['label']} "
            f"(acc {analysis['rounding_rule_summary'][0]['exact_accuracy']:.3f})\n"
            f"Best fixed on-ice offset: {self._format_rotation_contact_offset(analysis)}\n"
            f"Hybrid strategy: {self._format_rotation_strategy_comparison(analysis)}\n"
            f"Across skaters: {self._format_rotation_strategy_by_skater(analysis)}\n"
            f"Best per type: {self._format_rotation_rule_by_type(analysis)}"
        )

    def _format_rotation_rule_by_type(self, analysis: dict) -> str:
        return ", ".join(
            f"{item['label']}: {item['best_rule']} ({item['best_rule_accuracy']:.3f})"
            for item in analysis.get("type_summary", [])
        )

    def _format_rotation_strategy_comparison(self, analysis: dict) -> str:
        by_label = {item["label"]: item for item in analysis.get("strategy_summary", [])}
        current = by_label.get("current_round")
        hybrid = by_label.get("hybrid_non_axel_shift")
        if not current or not hybrid:
            return "n/a"
        gain = hybrid["exact_accuracy"] - current["exact_accuracy"]
        return (
            f"current {current['exact_accuracy']:.3f} -> "
            f"hybrid {hybrid['exact_accuracy']:.3f} ({gain:+.3f})"
        )

    def _rotation_strategy_label(self, label: str) -> str:
        return {
            "current_round": "Current",
            "fixed_contact_offset_0.45": "Fixed +0.45",
            "hybrid_non_axel_shift": "Hybrid",
            "best_rule_per_type_observed": "Per-type upper bound",
        }.get(label, label)

    def _format_rotation_contact_offset(self, analysis: dict) -> str:
        summary = analysis.get("contact_offset_summary", [])
        if not summary:
            return "n/a"
        best = summary[0]
        return f"+{best['offset_turns']:.2f} turn (acc {best['exact_accuracy']:.3f})"

    def _format_rotation_strategy_by_skater(self, analysis: dict) -> str:
        summaries = analysis.get("strategy_by_skater", [])
        if not summaries:
            return "n/a"
        positive = sum(1 for item in summaries if item["accuracy_gain"] > 0)
        min_gain = min(item["accuracy_gain"] for item in summaries)
        max_gain = max(item["accuracy_gain"] for item in summaries)
        return f"hybrid better for {positive}/{len(summaries)} skaters | gain {min_gain:+.3f} to {max_gain:+.3f}"

    def _apply_rotation_audit(self, analysis: dict) -> None:
        self.rotation_audit_analysis = analysis
        self.rotation_audit_summary_var.set(self._format_rotation_audit_summary(analysis))
        self._refresh_rotation_audit_records()
        self._draw_rotation_audit(analysis)
        self.status_var.set(f"Turn audit ready: {analysis['labelled_jumps']} labelled jumps")

    def _run_rotation_audit(self) -> None:
        root = self.rotation_audit_root_var.get().strip()
        if not root:
            messagebox.showwarning("Synergie Tools", "Select an annotation folder first.")
            return
        self.status_var.set("Running turn estimation audit...")
        self.rotation_audit_summary_var.set("Turn audit in progress...")
        self.rotation_audit_records_var.set([])
        self.rotation_audit_details_var.set("No suspicious turn estimate selected.")
        self._draw_placeholder_rotation_audit()
        self._draw_placeholder_rotation_audit_signal()

        def action() -> None:
            analysis = operations.audit_turn_estimation(root)
            self.root.after(0, lambda: self._apply_rotation_audit(analysis))

        self._run_in_thread(action, "Unable to run the turn estimation audit.")

    def _run_model_audit(self) -> None:
        self.status_var.set("Auditing saved models...")
        self.model_audit_summary_var.set("Model audit in progress...")
        self.model_audit_text.delete("1.0", tk.END)

        def action() -> None:
            audit = operations.audit_saved_models()
            summary = self._format_model_audit_summary(audit)
            lines = []
            for item in audit:
                lines.append(
                    f"{item['label']} | task={item.get('task', 'n/a')} | format={item['format']} | "
                    f"infer={item['can_infer']} | resume_training={item['can_resume_training']}\n"
                    f"path={item['path']}\n"
                    f"inputs={item.get('input_shapes') or 'n/a'}"
                )
                if item.get("error"):
                    lines.append(f"note={item['error']}")
                lines.append("")
            self.root.after(0, lambda: self.model_audit_summary_var.set(summary))
            self.root.after(0, lambda: self._replace_text(self.model_audit_text, "\n".join(lines)))
            self.root.after(0, lambda: self.status_var.set("Model audit completed"))

        self._run_in_thread(action, "Unable to audit saved models.")

    def _run_signal_importance(self) -> None:
        self.status_var.set("Running signal importance...")
        self.signal_summary_var.set("Signal importance in progress...")
        self._draw_placeholder_signal_plot()

        def action() -> None:
            analysis = operations.compute_signal_importance(
                self.signal_task_var.get(),
                self.signal_dataset_var.get().strip(),
                model_path=self.signal_model_path_var.get().strip() or None,
                repeats=int(self.signal_repeats_var.get()),
                temporal_windows=int(self.signal_windows_var.get()),
            )
            summary = self._format_signal_summary(analysis)
            self.root.after(0, lambda: self.signal_summary_var.set(summary))
            self.root.after(0, lambda: self._draw_signal_importance(analysis))
            self.root.after(0, lambda: self.status_var.set("Signal importance completed"))

        self._run_in_thread(action, "Unable to compute signal importance.")

    def _run_hyperparameter_search(self) -> None:
        self.status_var.set("Running hyperparameter search...")
        self.tuner_summary_var.set("Hyperparameter search in progress...")
        self.tuner_progress_var.set(0.0)
        self.tuner_progress_text_var.set("Preparing search...")
        self.tuner_log.delete("1.0", tk.END)
        self._draw_placeholder_tuner_plot()

        def action() -> None:
            def report_progress(event: dict) -> None:
                trial = event["trial"]
                total = event["total_trials"]
                if event["stage"] == "trial_started":
                    text = f"Running trial {trial}/{total}: {event['parameters']}"
                    percent = ((trial - 1) / total) * 100
                else:
                    result = event["result"]
                    text = (
                        f"Completed trial {trial}/{total}: "
                        f"best_val_accuracy={result['best_val_accuracy']:.3f}"
                    )
                    percent = (trial / total) * 100
                self.root.after(0, lambda: self.tuner_progress_var.set(percent))
                self.root.after(0, lambda: self.tuner_progress_text_var.set(text))

            summary = operations.run_hyperparameter_search(
                self.tuner_task_var.get(),
                self.tuner_dataset_var.get().strip(),
                self.tuner_architecture_var.get(),
                max_trials=int(self.tuner_trials_var.get()),
                epochs=int(self.tuner_epochs_var.get()),
                use_scalar_features=self.train_use_scalar_features_var.get(),
                progress_callback=report_progress,
            )
            if summary.get("best_trial"):
                operations.save_optimized_model_parameters(
                    summary["task"],
                    summary["architecture"],
                    summary["best_trial"],
                )
            formatted = self._format_tuner_summary(summary)
            lines = [formatted, ""]
            for item in summary["results"]:
                lines.append(
                    f"T{item['trial']}: best_val_accuracy={item['best_val_accuracy']:.3f}, "
                    f"final_val_accuracy={item['final_val_accuracy']:.3f}, epochs={item['epochs_ran']}, "
                    f"params={item['parameters']}"
                )
            self.root.after(0, lambda: self.tuner_summary_var.set(formatted))
            self.root.after(0, lambda: self._replace_text(self.tuner_log, "\n".join(lines)))
            self.root.after(0, lambda: self._draw_hyperparameter_results(summary))
            self.root.after(0, self._sync_train_parameter_profile)
            self.root.after(0, lambda: self.tuner_progress_var.set(100.0))
            self.root.after(0, lambda: self.tuner_progress_text_var.set("Search completed."))
            self.root.after(0, lambda: self.status_var.set("Hyperparameter search completed"))

        self._run_in_thread(action, "Unable to run hyperparameter search.")

    def _run_window_benchmark(self) -> None:
        self.status_var.set("Running window benchmark...")
        self.window_progress_var.set(0.0)
        self.window_progress_text_var.set("Preparing benchmark...")
        self.window_log.delete("1.0", tk.END)
        windows = [int(value.strip()) for value in self.window_candidates_var.get().split(",") if value.strip()]

        def action() -> None:
            def report(event: dict) -> None:
                index = event["index"]
                total = event["total"]
                if event["stage"] == "started":
                    percent = ((index - 1) / total) * 100
                    text = f"Training window {index}/{total}: {event['frames']} frames"
                else:
                    percent = (index / total) * 100
                    result = event["result"]
                    text = f"Completed {index}/{total}: {result['frames']} frames, balanced_acc={result['balanced_accuracy']:.3f}"
                self.root.after(0, lambda: self.window_progress_var.set(percent))
                self.root.after(0, lambda: self.window_progress_text_var.set(text))

            summary = operations.benchmark_temporal_windows(
                self.window_task_var.get(),
                self.window_dataset_var.get().strip(),
                self.window_architecture_var.get(),
                windows=windows,
                epochs=int(self.window_epochs_var.get()),
                use_scalar_features=self.train_use_scalar_features_var.get(),
                progress_callback=report,
            )
            best = summary["best"]
            lines = [
                f"Best window: {best['frames']} frames | balanced_acc={best['balanced_accuracy']:.3f} | best_val_acc={best['best_val_accuracy']:.3f}",
                summary["note"],
                "",
            ]
            for item in summary["results"]:
                lines.append(
                    f"{item['frames']} frames: balanced_acc={item['balanced_accuracy']:.3f}, "
                    f"best_val_acc={item['best_val_accuracy']:.3f}, epochs={item['epochs_ran']}"
                )
            self.root.after(0, lambda: self.window_summary_var.set(lines[0]))
            self.root.after(0, lambda: self._replace_text(self.window_log, "\n".join(lines)))
            self.root.after(0, lambda: self.window_progress_var.set(100.0))
            self.root.after(0, lambda: self.window_progress_text_var.set("Benchmark completed."))
            self.root.after(0, lambda: self.status_var.set("Window benchmark completed"))

        self._run_in_thread(action, "Unable to run window benchmark.")

    def _run_window_offset_benchmark(self) -> None:
        self.status_var.set("Running window offset benchmark...")
        self.window_progress_var.set(0.0)
        self.window_progress_text_var.set("Preparing offset benchmark...")
        self.window_log.delete("1.0", tk.END)
        offsets = [int(value.strip()) for value in self.window_offsets_var.get().split(",") if value.strip()]

        def action() -> None:
            def report(event: dict) -> None:
                index = event["index"]
                total = event["total"]
                if event["stage"] == "reexporting":
                    percent = (index / total) * 25
                    text = f"Re-exporting segments {index}/{total}: {event['reexported']} extended"
                elif event["stage"] == "started":
                    percent = ((index - 1) / total) * 100
                    text = f"Training offset {index}/{total}: segment start {event['offset']}"
                else:
                    percent = (index / total) * 100
                    result = event["result"]
                    text = (
                        f"Completed {index}/{total}: "
                        f"[{result['start_relative_to_takeoff']},{result['end_relative_to_takeoff']}] "
                        f"balanced_acc={result['balanced_accuracy']:.3f}"
                    )
                self.root.after(0, lambda: self.window_progress_var.set(percent))
                self.root.after(0, lambda: self.window_progress_text_var.set(text))

            summary = operations.benchmark_temporal_offsets(
                self.window_task_var.get(),
                self.window_dataset_var.get().strip(),
                self.window_architecture_var.get(),
                window_frames=int(self.window_offset_frames_var.get()),
                offsets=offsets,
                epochs=int(self.window_epochs_var.get()),
                use_scalar_features=self.train_use_scalar_features_var.get(),
                progress_callback=report,
            )
            best = summary["best"]
            lines = [
                f"Best offset window: [{best['start_relative_to_takeoff']},{best['end_relative_to_takeoff']}] "
                f"| balanced_acc={best['balanced_accuracy']:.3f} | best_val_acc={best['best_val_accuracy']:.3f}",
                summary["note"],
                "",
            ]
            for item in summary["results"]:
                lines.append(
                    f"[{item['start_relative_to_takeoff']},{item['end_relative_to_takeoff']}] "
                    f"(segment offset {item['offset']}): balanced_acc={item['balanced_accuracy']:.3f}, "
                    f"best_val_acc={item['best_val_accuracy']:.3f}, epochs={item['epochs_ran']}"
                )
            self.root.after(0, lambda: self.window_summary_var.set(lines[0]))
            self.root.after(0, lambda: self._replace_text(self.window_log, "\n".join(lines)))
            self.root.after(0, lambda: self.window_progress_var.set(100.0))
            self.root.after(0, lambda: self.window_progress_text_var.set("Offset benchmark completed."))
            self.root.after(0, lambda: self.status_var.set("Window offset benchmark completed"))

        self._run_in_thread(action, "Unable to run window offset benchmark.")

    def _build_detection_tuning_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(parent, text="Reviewed Detection Errors", padding=12)
        controls.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(2, weight=1)
        refresh_errors_button = ttk.Button(controls, text="Refresh reviewed errors", command=self._refresh_detection_review)
        refresh_errors_button.grid(row=0, column=0, sticky="w")
        self._add_tooltip(refresh_errors_button, "Recharge les faux positifs et faux negatifs marques pendant l'annotation.")
        ttk.Label(controls, textvariable=self.detection_review_summary_var, justify=tk.LEFT, wraplength=320).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(8, 8),
        )
        self.detection_review_listbox = tk.Listbox(controls, listvariable=self.detection_review_records_var, width=48)
        self.detection_review_listbox.grid(row=2, column=0, sticky="nsew")
        threshold_sweep_button = ttk.Button(controls, text="Run threshold sweep", command=self._run_detection_parameter_sweep)
        threshold_sweep_button.grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 0),
        )
        self._add_tooltip(threshold_sweep_button, "Teste plusieurs seuils et sigmas sur les exemples revus pour proposer un compromis.")
        ttk.Label(controls, textvariable=self.detection_tuning_summary_var, justify=tk.LEFT, wraplength=320).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(8, 0),
        )

        notes = ttk.LabelFrame(parent, text="How to use this tab", padding=12)
        notes.grid(row=0, column=1, sticky="nsew")
        notes.columnconfigure(0, weight=1)
        ttk.Label(
            notes,
            justify=tk.LEFT,
            wraplength=700,
            text=(
                "Rows marked 'Not a jump' become false positives; rows added manually become false negatives.\n\n"
                "Use this reviewed set before changing thresholds: false positives show where the detector is too permissive, "
                "false negatives show missed events. A later publication-ready version should replay the raw sessions across a "
                "grid of threshold/smoothing settings and optimize balanced error rates on held-out sessions."
            ),
        ).grid(row=0, column=0, sticky="nw")
        self._refresh_detection_review()

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

    def _on_train_architecture_changed(self, _event=None) -> None:
        self._sync_train_parameter_profile()

    def _on_train_parameter_profile_changed(self, _event=None) -> None:
        self._apply_train_parameter_profile()

    def _on_signal_task_changed(self, _event=None) -> None:
        self.signal_model_path_var.set(operations.latest_model_path_for_task(self.signal_task_var.get()))
        self._refresh_signal_models()

    def _refresh_signal_models(self) -> None:
        task = self.signal_task_var.get()
        models = operations.list_pretrained_training_models(task=task, compatible_only=True)
        labels = ["Active model alias"] + [operations.format_pretrained_model_label(model) for model in models]
        self.signal_model_box.configure(values=labels)
        self.signal_model_choice_var.set(labels[0])

    def _on_signal_model_changed(self, _event=None) -> None:
        selected = self.signal_model_choice_var.get()
        if selected == "Active model alias":
            self.signal_model_path_var.set(operations.latest_model_path_for_task(self.signal_task_var.get()))
            return
        for model in operations.list_pretrained_training_models(task=self.signal_task_var.get(), compatible_only=True):
            if operations.format_pretrained_model_label(model) == selected:
                self.signal_model_path_var.set(model["path"])
                return

    def _on_tuner_task_changed(self, _event=None) -> None:
        self._sync_tuner_architectures()

    def _sync_tuner_architectures(self) -> None:
        options = self.TRAIN_ARCHITECTURES.get(self.tuner_task_var.get(), [])
        self.tuner_architecture_box.configure(values=options)
        if options and self.tuner_architecture_var.get() not in options:
            self.tuner_architecture_var.set(options[0])

    def _on_window_task_changed(self, _event=None) -> None:
        self._sync_window_architectures()
        if self.window_task_var.get() == "type":
            self.window_candidates_var.set("240,200,160,120")
            self.window_offset_frames_var.set("200")
            self.window_offsets_var.set("-60,-40,-20,0")
        else:
            self.window_candidates_var.set("180,160,140,120")
            self.window_offset_frames_var.set("180")
            self.window_offsets_var.set("20,40,60")

    def _sync_window_architectures(self) -> None:
        options = self.TRAIN_ARCHITECTURES.get(self.window_task_var.get(), [])
        self.window_architecture_box.configure(values=options)
        if options and self.window_architecture_var.get() not in options:
            self.window_architecture_var.set(options[0])

    def _on_pretrained_model_changed(self, _event=None) -> None:
        self._sync_pretrained_architecture()
        pretrained_model_id = self._selected_pretrained_model_id()
        if not pretrained_model_id:
            self._draw_placeholder_confusion_matrix()
            return
        self.status_var.set("Loading confusion matrix for selected model...")

        def action() -> None:
            model_entry = next(
                (
                    model
                    for model in operations.list_pretrained_training_models(task=self.train_task_var.get(), compatible_only=True)
                    if model["id"] == pretrained_model_id
                ),
                None,
            )
            if model_entry is None:
                self.root.after(0, self._draw_placeholder_confusion_matrix)
                return
            summary = operations.evaluate_registered_model(model_entry)
            self.root.after(0, lambda: self._draw_confusion_matrix(summary))
            self.root.after(
                0,
                lambda: self.train_quality_summary_var.set(
                    f"Selected model: {model_entry['label']}\n"
                    f"Test accuracy: {summary.get('test_accuracy', 0.0):.3f} | "
                    f"Test samples: {summary.get('test_samples', 0)}"
                ),
            )
            self.root.after(0, lambda: self.status_var.set("Selected model metrics loaded"))

        self._run_in_thread(action, "Unable to evaluate selected pretrained model.")

    def _on_process_file_selected(self, _event=None) -> None:
        selection = self.process_session_files.curselection()
        if selection:
            selected_path = self.process_session_files.get(selection[0])
            self.csv_path_var.set(selected_path)
            self._set_default_output_path(selected_path)
            selected_count = len(selection)
            description = (
                f"{self._describe_selected_file(selected_path)}\n"
                f"Automatic output: {self.output_path_var.get()}"
            )
            if selected_count > 1:
                description += f"\nBatch selection: {selected_count} files"
            self.process_selected_file_info_var.set(description)

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
        suggestion = operations.suggest_session_from_imu_file(representative_file["path"])
        self.new_data_session_suggestion_var.set(
            f"{suggestion['session_id']} -> data/raw/{suggestion['path']} | sync offset {suggestion['sample_time_fine_synchro']}"
        )
        sensors = ", ".join(file_metadata["sensor_id"] for file_metadata in session["files"])
        self.new_data_summary_var.set(
            f"Session: {session['session_key']}\n"
            f"Sensors: {sensors} | {session['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Workflow: {(session.get('workflow') or {}).get('status', 'not processed yet')}"
        )
        workflow = session.get("workflow") or {}
        workflow_status = workflow.get("status", "not processed yet")
        prediction_status = workflow.get("prediction_status", "will use selected models when processed")
        self.new_data_automation_var.set(
            "Automatic actions:\n"
            f"- annotation CSV path reserved automatically\n"
            f"- session target inferred from IMU timestamp\n"
            f"- workflow state: {workflow_status}\n"
            f"- prediction status: {prediction_status}"
        )

    def _add_selected_new_data_session(self) -> None:
        raw_file = self.new_data_selected_file_var.get().strip()
        if not raw_file:
            messagebox.showwarning("Synergie Tools", "Select a new IMU session first.")
            return
        suggestion = operations.suggest_session_from_imu_file(raw_file)
        try:
            metadata = operations.add_session(
                suggestion["session_id"],
                suggestion["path"],
                suggestion["sample_time_fine_synchro"],
            )
        except ValueError as exc:
            messagebox.showinfo("Synergie Tools", str(exc))
            return
        self.session_var.set(suggestion["session_id"])
        self.inspect_session_var.set(suggestion["session_id"])
        self._refresh_all_sessions()
        self.status_var.set(f"Session added automatically: {suggestion['session_id']} -> {metadata['path']}")
        self.new_data_automation_var.set(
            "Automatic session created.\n"
            f"- ID: {suggestion['session_id']}\n"
            f"- raw destination: data/raw/{metadata['path']}\n"
            "- sync offset starts at 0 until corrected or measured"
        )

    def _refresh_annotation_files(self) -> None:
        files = operations.list_pending_annotation_files()
        self._annotation_files_cache = files
        global_progress = operations.summarize_pending_annotation_files()
        per_file = {item["path"]: item for item in global_progress["files"]}
        self.annotation_files_var.set(
            [
                f"{path.name} | pending {per_file[path]['pending']} / {per_file[path]['total']}"
                for path in files
            ]
        )
        self.annotation_global_progress_var.set(
            f"All files pending annotations: {global_progress['pending']} | "
            f"Completed: {global_progress['completed']} / {global_progress['total']}"
        )
        if not files:
            self.annotation_summary_var.set("No pending annotation file found.")
            self.annotation_progress_var.set("Pending annotations: 0")
            self.annotation_jump_listbox.delete(0, tk.END)
            self.annotation_dataframe = None
            self.annotation_file_path = None
            self.annotation_metadata = {}
            self.annotation_video_path_var.set("")
            self.annotation_video_directory_var.set("")
            self.annotation_video_matches_var.set([])
            self._annotation_video_match_cache = []
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
        prefilled_count = 0
        if "prediction_source" in self.annotation_dataframe:
            prefilled_count = int(self.annotation_dataframe["prediction_source"].fillna("").astype(str).ne("").sum())
        self.annotation_summary_var.set(
            f"{file_path.name}\nEntries: {len(self.annotation_dataframe)} | Sensors: {sensor_count} | "
            f"Model-prefilled: {prefilled_count}\n"
            f"{'Predicted labels will initialize the controls.' if prefilled_count else 'No model predictions stored: labels start blank until reviewed.'}"
        )
        self._refresh_annotation_progress()
        video_path = self.annotation_metadata.get("video_path", "")
        video_directory = self.annotation_metadata.get("video_directory", "")
        self.annotation_video_path_var.set(video_path)
        self.annotation_video_directory_var.set(video_directory)
        self.annotation_video_matches_var.set([])
        self._annotation_video_match_cache = []
        if video_path and Path(video_path).exists():
            self._load_annotation_video(video_path, persist=False)
        else:
            self._stop_annotation_playback()
            self._release_annotation_video()
            self._draw_placeholder_annotation_video()
            self.annotation_video_info_var.set("No video loaded.")
            if video_directory and Path(video_directory).exists():
                self._find_annotation_video_matches(auto_load=True)
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
            if str(row.get("prediction_source", "") or ""):
                jump_type = int(float(row.get("type", 8)))
                success = int(float(row.get("success", 2)))
                label += f" | model: {operations.JUMP_TYPE_LABELS.get(jump_type, jump_type)} / success {success}"
            self.annotation_jump_listbox.insert(tk.END, label)

    def _refresh_annotation_progress(self) -> None:
        if self.annotation_dataframe is None:
            self.annotation_progress_var.set("Pending annotations: 0")
            return
        progress = operations.summarize_annotation_progress(self.annotation_dataframe)
        self.annotation_progress_var.set(
            f"Pending annotations: {progress['pending']} | Completed: {progress['completed']} / {progress['total']}"
        )
        global_progress = operations.summarize_pending_annotation_files()
        self.annotation_global_progress_var.set(
            f"All files pending annotations: {global_progress['pending']} | "
            f"Completed: {global_progress['completed']} / {global_progress['total']}"
        )

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
        type_key = next((key for key, _label, value in operations.ANNOTATION_JUMP_TYPE_OPTIONS if value == type_value), "")
        self.annotation_type_var.set(type_key)
        self.annotation_turn_var.set(operations.annotation_turn_value_for_ui(type_key, row.get("turns", "")))
        self.annotation_success_var.set(str(int(float(row.get("success", 2)))))
        self.annotation_review_status_var.set(review_status)
        self.annotation_athlete_var.set(str(row.get("athlete_id", row.get("skater", ""))))
        self.annotation_combination_var.set(bool(row.get("combination", False)))
        self._sync_annotation_turn_options()
        self._sync_annotation_review_controls()
        self._refresh_annotation_video_context()
        self._draw_annotation_segment(row)

    def _sync_annotation_turn_options(self) -> None:
        options = operations.annotation_turn_options(self.annotation_type_var.get())
        if self.annotation_turn_var.get() not in options:
            self.annotation_turn_var.set(options[0] if options and self.annotation_type_var.get() else "")
        self._sync_annotation_review_controls()

    def _sync_annotation_review_controls(self) -> None:
        excluded = self.annotation_review_status_var.get() in {"not_seen_on_video", "weird_signal", "not_a_jump"}
        hint = ""
        if self.annotation_review_status_var.get() == "not_seen_on_video":
            hint = "This jump will be excluded from training because it is marked as unseen on video."
        elif self.annotation_review_status_var.get() == "weird_signal":
            hint = "This jump will be excluded from training because the signal or takeoff/landing bounds look unreliable."
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

    def _current_tab_text(self) -> str:
        try:
            return str(self.notebook.tab(self.notebook.select(), "text"))
        except Exception:
            return ""

    def _annotate_tab_active(self) -> bool:
        return self._current_tab_text() == "Data - Annotate"

    def _dataset_review_tab_active(self) -> bool:
        return self._current_tab_text() == "Review - Dataset"

    def _on_global_keypress(self, event) -> None:
        widget = event.widget
        if isinstance(widget, (tk.Entry, tk.Text, scrolledtext.ScrolledText)):
            return

        key = (event.keysym or event.char or "").lower()
        if not key:
            return

        if self._dataset_review_tab_active():
            if key == "x" and self._selected_dataset_review_record() is not None:
                self._exclude_selected_training_jump()
            return

        if not self._annotate_tab_active():
            return
        if self.annotation_dataframe is None or self._selected_annotation_index() is None:
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
            self.annotation_review_status_var.set("weird_signal")
            self.status_var.set("Annotation review status selected: weird signal / bad bounds")
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

    def _open_annotation_video_popup(self) -> None:
        if self.annotation_video_popup is not None and self.annotation_video_popup.winfo_exists():
            self.annotation_video_popup.lift()
            self.annotation_video_popup.focus_force()
            return

        popup = tk.Toplevel(self.root)
        popup.title("Choose session video")
        popup.geometry("900x420")
        popup.transient(self.root)
        popup.columnconfigure(1, weight=1)
        popup.rowconfigure(2, weight=1)
        popup.protocol("WM_DELETE_WINDOW", self._close_annotation_video_popup)

        ttk.Label(popup, text="Video folder").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        ttk.Entry(popup, textvariable=self.annotation_video_directory_var).grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 6))
        folder_buttons = ttk.Frame(popup)
        folder_buttons.grid(row=0, column=2, sticky="e", padx=12, pady=(12, 6))
        ttk.Button(folder_buttons, text="Browse folder", command=self._browse_annotation_video_directory).grid(row=0, column=0, sticky="w")
        ttk.Button(folder_buttons, text="Find match", command=self._find_annotation_video_matches).grid(row=0, column=1, sticky="w", padx=(6, 0))

        popup_info = ttk.Label(popup, textvariable=self.annotation_video_info_var, justify=tk.LEFT, wraplength=840)
        popup_info.grid(row=1, column=0, columnspan=3, sticky="w", padx=12, pady=(0, 8))

        matches = tk.Listbox(
            popup,
            listvariable=self.annotation_video_matches_var,
            exportselection=False,
            height=12,
        )
        matches.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=12, pady=(0, 8))
        matches.bind("<<ListboxSelect>>", self._on_annotation_video_match_selected)

        bottom_buttons = ttk.Frame(popup)
        bottom_buttons.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 12))
        ttk.Button(bottom_buttons, text="Load selected", command=self._load_selected_annotation_video_match).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom_buttons, text="Close", command=self._close_annotation_video_popup).grid(row=0, column=1, sticky="w", padx=(6, 0))

        self.annotation_video_popup = popup
        self.annotation_video_popup_info_label = popup_info
        self.annotation_video_matches_listbox = matches

    def _close_annotation_video_popup(self) -> None:
        if self.annotation_video_popup is not None and self.annotation_video_popup.winfo_exists():
            self.annotation_video_popup.destroy()
        self.annotation_video_popup = None
        self.annotation_video_popup_info_label = None
        self.annotation_video_matches_listbox = None

    def _browse_annotation_video_directory(self) -> None:
        directory = filedialog.askdirectory(title="Select video folder")
        if not directory:
            return
        self.annotation_video_directory_var.set(directory)
        if self.annotation_file_path is not None:
            self.annotation_metadata = operations.set_annotation_video_directory(self.annotation_file_path, directory)
        self._find_annotation_video_matches(auto_load=True)

    def _find_annotation_video_matches(self, auto_load: bool = False) -> None:
        if self.annotation_file_path is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation file first.")
            return
        video_directory = self.annotation_video_directory_var.get().strip()
        if not video_directory:
            messagebox.showwarning("Synergie Tools", "Select a video folder first.")
            return
        directory_path = Path(video_directory)
        if not directory_path.exists():
            messagebox.showerror("Synergie Tools", f"Video folder not found:\n{directory_path}")
            return

        if self.annotation_file_path is not None:
            self.annotation_metadata = operations.set_annotation_video_directory(self.annotation_file_path, directory_path)

        result = operations.find_matching_videos(
            self.annotation_file_path,
            directory_path,
            annotation_rows=self.annotation_dataframe,
            recursive=True,
            limit=8,
        )
        self._annotation_video_match_cache = result["matches"]

        if not result["matches"]:
            self.annotation_video_matches_var.set(["No video found in selected folder."])
            self.annotation_video_info_var.set("No matching video found.")
            return

        entries = []
        for match in result["matches"]:
            delta_seconds = match["delta_seconds"]
            delta_text = "unknown gap" if delta_seconds is None else f"{int(round(delta_seconds))} s gap"
            entries.append(
                f"{match['name']} | {match['recorded_at'].strftime('%Y-%m-%d %H:%M:%S')} | {match['recorded_at_source']} | {delta_text}"
            )
        self.annotation_video_matches_var.set(entries)

        reference_datetime = result["reference_datetime"]
        if reference_datetime is None:
            self.annotation_video_info_var.set(f"{len(result['matches'])} video candidates found.")
        else:
            best_match = result["matches"][0]
            self.annotation_video_info_var.set(
                f"Reference: {reference_datetime.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"best match: {best_match['name']} ({best_match['recorded_at_source']})"
            )

        if self.annotation_video_matches_listbox is not None:
            self.annotation_video_matches_listbox.selection_clear(0, tk.END)
            if result["matches"]:
                self.annotation_video_matches_listbox.selection_set(0)
        if auto_load:
            self._load_annotation_video(result["matches"][0]["path"], persist=True)

    def _on_annotation_video_match_selected(self, _event=None) -> None:
        if self.annotation_video_matches_listbox is None:
            return
        selection = self.annotation_video_matches_listbox.curselection()
        if not selection or selection[0] >= len(self._annotation_video_match_cache):
            return
        self._load_annotation_video(self._annotation_video_match_cache[selection[0]]["path"], persist=True)

    def _load_selected_annotation_video_match(self) -> None:
        if self.annotation_video_matches_listbox is None:
            return
        selection = self.annotation_video_matches_listbox.curselection()
        if not selection or selection[0] >= len(self._annotation_video_match_cache):
            messagebox.showwarning("Synergie Tools", "Select a video match first.")
            return
        self._load_annotation_video(self._annotation_video_match_cache[selection[0]]["path"], persist=True)

    def _release_annotation_video(self) -> None:
        if self.annotation_video_capture is not None:
            self.annotation_video_capture.release()
            self.annotation_video_capture = None
        self.annotation_video_fps = 0.0
        self.annotation_video_frame_count = 0
        self.annotation_video_duration_ms = 0.0
        self.annotation_video_current_ms = 0.0
        self._set_annotation_play_button_state(False)

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
        self.annotation_video_directory_var.set(str(path.parent))
        self.annotation_video_slider.configure(to=max(duration_ms, 1.0))
        self.annotation_video_info_var.set(f"{path.name} | fps={fps:.2f}")
        if persist and self.annotation_file_path is not None:
            self.annotation_metadata = operations.set_annotation_video_path(self.annotation_file_path, path)
            self.annotation_metadata = operations.set_annotation_video_directory(self.annotation_file_path, path.parent)
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

    def _set_annotation_play_button_state(self, is_playing: bool) -> None:
        self.annotation_play_button_var.set("⏸" if is_playing else "▶")

    def _play_annotation_video(self) -> None:
        if self.annotation_video_capture is None:
            messagebox.showwarning("Synergie Tools", "Load the session video first.")
            return
        if self._annotation_playback_after_id is not None:
            self._stop_annotation_playback()
            return

        self._stop_annotation_playback()
        self._set_annotation_play_button_state(True)
        base_frame_ms = 1000.0 / self.annotation_video_fps if self.annotation_video_fps > 0 else 40.0
        step_ms = max(base_frame_ms, 20.0)
        delay_ms = max(int(round(base_frame_ms)), 20)
        target_ms = max(self.annotation_video_duration_ms, self.annotation_video_current_ms)

        def advance() -> None:
            if self.annotation_video_capture is None:
                self._annotation_playback_after_id = None
                self._set_annotation_play_button_state(False)
                return
            next_ms = self.annotation_video_current_ms + step_ms
            if next_ms >= target_ms:
                self._display_annotation_video_frame(target_ms)
                self._annotation_playback_after_id = None
                self._set_annotation_play_button_state(False)
                return
            self._display_annotation_video_frame(next_ms)
            self._annotation_playback_after_id = self.root.after(delay_ms, advance)

        self._annotation_playback_after_id = self.root.after(delay_ms, advance)

    def _play_annotation_to_current_jump(self) -> None:
        if self.annotation_video_capture is None:
            messagebox.showwarning("Synergie Tools", "Load the session video first.")
            return
        target_ms = self._current_jump_video_time_ms()
        if target_ms is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation entry first.")
            return

        self._stop_annotation_playback()
        self._set_annotation_play_button_state(True)
        start_ms = max(target_ms - 5000.0, 0.0)
        base_frame_ms = 1000.0 / self.annotation_video_fps if self.annotation_video_fps > 0 else 40.0
        step_ms = max(base_frame_ms * 5.0, 100.0)
        delay_ms = max(int(round(base_frame_ms)), 20)
        self._display_annotation_video_frame(start_ms)

        def advance() -> None:
            if self.annotation_video_capture is None:
                self._annotation_playback_after_id = None
                self._set_annotation_play_button_state(False)
                return
            next_ms = self.annotation_video_current_ms + step_ms
            if next_ms >= target_ms:
                self._display_annotation_video_frame(target_ms)
                self._annotation_playback_after_id = None
                self._set_annotation_play_button_state(False)
                return
            self._display_annotation_video_frame(next_ms)
            self._annotation_playback_after_id = self.root.after(delay_ms, advance)

        self._annotation_playback_after_id = self.root.after(delay_ms, advance)

    def _stop_annotation_playback(self) -> None:
        if self._annotation_playback_after_id is not None:
            self.root.after_cancel(self._annotation_playback_after_id)
            self._annotation_playback_after_id = None
        self._set_annotation_play_button_state(False)

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
        gyro_column = "Gyr_X_smoothed" if "Gyr_X_smoothed" in dataframe else "Gyr_X"
        raw_line = self.annotation_ax.plot(dataframe["ms"], dataframe["Gyr_X"], label="Gyr_X raw", linewidth=0.8, alpha=0.35, color="#1f77b4")[0]
        gyro_line = self.annotation_ax.plot(dataframe["ms"], dataframe[gyro_column], label="Gyr_X smoothed", linewidth=1.2, color="#1f77b4")[0]
        acc_line = self.annotation_acc_ax.plot(dataframe["ms"], dataframe["Acc_X"], label="Acc_X", linewidth=0.9, alpha=0.8, color="#ff7f0e")[0]
        derivative_line = None
        if "X_gyr_second_derivative" in dataframe:
            derivative_line = self.annotation_acc_ax.plot(
                dataframe["ms"],
                dataframe["X_gyr_second_derivative"],
                label="2nd derivative",
                linewidth=0.9,
                alpha=0.8,
                color="crimson",
            )[0]
            self.annotation_acc_ax.axhline(DEFAULT_DETECTION_THRESHOLD, color="crimson", linestyle="--", linewidth=0.9, label="Detection threshold")
        if row.get("start_ms", "") != "":
            self.annotation_ax.axvline(float(row["start_ms"]), color="black", linestyle="--", linewidth=1.0, label="Takeoff")
        if row.get("end_ms", "") != "":
            self.annotation_ax.axvline(float(row["end_ms"]), color="black", linestyle=":", linewidth=1.0, label="Landing")
        self.annotation_ax.set_title(
            f"{video_time_label} | "
            f"{row.get('athlete_id', row.get('skater', 'unknown'))} | "
            f"sensor {sensor_id} | {row.get('source_file', '')}"
        )
        self.annotation_ax.set_xlabel("ms")
        self.annotation_ax.set_ylabel("Gyroscope")
        self.annotation_acc_ax.set_ylabel("Acceleration")
        handles = [raw_line, gyro_line, acc_line]
        labels = ["Gyr_X raw", "Gyr_X smoothed", "Acc_X"]
        if derivative_line is not None:
            handles.append(derivative_line)
            labels.append("2nd derivative")
        self.annotation_ax.legend(handles, labels, loc="upper right", fontsize=8)
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
        is_excluded_by_status = self.annotation_review_status_var.get() in {"not_seen_on_video", "weird_signal", "not_a_jump"}
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
        self.annotation_dataframe.at[index, "combination"] = bool(self.annotation_combination_var.get())
        self.annotation_dataframe.at[index, "annotation_status"] = "annotated"
        self.annotation_dataframe.to_csv(self.annotation_file_path, index=False)
        self._refresh_annotation_jump_list()
        self._refresh_annotation_progress()
        self.annotation_jump_listbox.selection_clear(0, tk.END)
        self.annotation_jump_listbox.selection_set(index)
        self.status_var.set("Annotation saved")

    def _finalize_current_annotation_file(self) -> None:
        if self.annotation_file_path is None:
            messagebox.showwarning("Synergie Tools", "Select an annotation file first.")
            return
        progress = operations.summarize_annotation_progress(self.annotation_dataframe)
        if progress["pending"] > 0:
            messagebox.showwarning(
                "Synergie Tools",
                f"Cannot finalize yet: {progress['pending']} annotations are still pending.",
            )
            return
        if not messagebox.askyesno(
            "Synergie Tools",
            "Finalize this annotated file, archive the old jumplist, and merge labelled jumps into the training set?",
        ):
            return
        annotation_path = self.annotation_file_path
        self.status_var.set("Finalizing annotation file...")

        def action() -> None:
            result = operations.finalize_annotation_file(annotation_path, dataset_path=self.dataset_var.get())
            self.root.after(0, lambda: self._apply_finalized_annotation_file(result))

        self._run_in_thread(action, "Unable to finalize annotation file.")

    def _apply_finalized_annotation_file(self, result: dict) -> None:
        self.annotation_dataframe = None
        self.annotation_file_path = None
        self._refresh_annotation_files()
        self._refresh_training_dataset_stats()
        self.status_var.set(
            f"Finalized annotation file: {result['rows_added']} rows added; retraining recommended"
        )
        messagebox.showinfo(
            "Synergie Tools",
            f"Finalization complete.\n\n"
            f"Rows added: {result['rows_added']}\n"
            f"Raw IMU files moved: {result['raw_files_moved']}\n"
            "Automatic actions completed:\n"
            "- labelled segments moved into data/annotated\n"
            "- completed annotation file archived with the session\n"
            "- workflow status marked finalized\n"
            f"Previous jumplist archived at: {result['archive_path']}\n\n"
            "The training dataset changed; retraining is recommended.",
        )

    def _refresh_detection_review(self) -> None:
        analysis = operations.analyze_detection_review_labels()
        self.detection_review_summary_var.set(
            f"Reviewed detections: {analysis['reviewed_detected']}\n"
            f"False positives: {analysis['false_positive_count']}\n"
            f"False negatives: {analysis['false_negative_count']}"
        )
        labels = []
        for record in analysis["false_positives"]:
            labels.append(f"FP | sensor {record['sensor_id']} | {record['annotation_file']} | row {record['row_index'] + 1}")
        for record in analysis["false_negatives"]:
            labels.append(f"FN | sensor {record['sensor_id']} | {record['annotation_file']} | row {record['row_index'] + 1}")
        self.detection_review_records_var.set(labels)

    def _run_detection_parameter_sweep(self) -> None:
        self.status_var.set("Running detection parameter sweep...")
        self.detection_tuning_summary_var.set("Threshold sweep in progress...")

        def action() -> None:
            result = operations.optimize_detection_parameters()
            best = result["best"]
            if best is None:
                summary = "No reviewed segments available for threshold tuning."
            else:
                operations.save_optimized_detection_parameters(best)
                summary = (
                    f"Reviewed windows: {result['reviewed_segments']}\n"
                    f"Best threshold: {best['threshold']:.2f} | sigma: {best['smoothing_sigma']:.0f}\n"
                    f"Balanced error: {best['balanced_error']:.3f} | "
                    f"FP: {best['false_positive']} | FN: {best['false_negative']}"
                )
            self.root.after(0, lambda: self.detection_tuning_summary_var.set(summary))
            self.root.after(0, self._refresh_detection_parameter_modes)
            self.root.after(0, lambda: self.status_var.set("Detection parameter sweep completed"))

        self._run_in_thread(action, "Unable to optimize detection parameters.")

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

    def _replace_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.see(tk.END)

    def _run_in_thread(self, target, on_error_message: str) -> None:
        def runner() -> None:
            try:
                target()
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Synergie Tools", f"{on_error_message}\n\n{exc}"))
                self.root.after(0, lambda: self.status_var.set("Error"))

        threading.Thread(target=runner, daemon=True).start()

    def _run_batch_process_files(self) -> None:
        selections = self.process_session_files.curselection()
        if not selections:
            messagebox.showwarning("Synergie Tools", "Select one or more session CSV files first.")
            return

        session = operations.session_metadata(self.session_var.get())
        input_files = [
            {
                "path": self.process_session_files.get(index),
                "session_name": self.session_var.get(),
                "sample_time_fine_synchro": session["sample_time_fine_synchro"],
            }
            for index in selections
        ]
        self._run_batch_process_paths(input_files, label="selected")

    def _run_global_batch_process_files(self) -> None:
        input_files = operations.list_all_session_csv_files()
        if not input_files:
            messagebox.showwarning("Synergie Tools", "No CSV files are available across configured sessions.")
            return
        self._run_batch_process_paths(input_files, label="all sessions")

    def _run_batch_process_paths(self, input_files: list[dict], *, label: str) -> None:
        type_path = self._process_prediction_model_path("type", self.process_type_model_var.get())
        success_path = self._process_prediction_model_path("success", self.process_success_model_var.get())
        self.status_var.set(f"Batch processing {label}: {len(input_files)} files...")
        self.process_batch_progress_var.set(f"Preparing batch {label}: 0/{len(input_files)} files completed.")
        self.process_log.delete("1.0", tk.END)
        self._log(self.process_log, f"Batch {label} started: {len(input_files)} CSV files to process.")
        for summary in self._summarize_batch_sessions(input_files):
            self._log(
                self.process_log,
                (
                    f"- Session {summary['session_name']}: {summary['file_count']} file(s), "
                    f"sync offset={summary['sample_time_fine_synchro']}"
                ),
            )
        self._log(
            self.process_log,
            (
                f"Prediction models: type={Path(type_path).name if type_path else 'none'}, "
                f"success={Path(success_path).name if success_path else 'none'}"
            ),
        )

        def action() -> None:
            created = []
            total = len(input_files)
            for index, item in enumerate(input_files, start=1):
                csv_path = item["path"]
                file_name = f"{item['session_name']} / {Path(csv_path).name}"
                output_path = self._suggest_output_path(csv_path)
                self.root.after(
                    0,
                    lambda current=index, count=total, name=file_name, source=csv_path, output=output_path, offset=item["sample_time_fine_synchro"]: self._on_batch_file_started(
                        current,
                        count,
                        name,
                        source,
                        output,
                        offset,
                    ),
                )
                result = operations.process_csv_file(
                    csv_path,
                    synchro=item["sample_time_fine_synchro"],
                    output_path=output_path,
                    type_model_path=type_path,
                    success_model_path=success_path,
                )
                created.append(result["path"])
                self.root.after(
                    0,
                    lambda current=index, count=total, path=result["path"]: self._on_batch_file_completed(current, count, path),
                )
                if result["prediction_status"] != "predicted":
                    self.root.after(0, lambda: self._log(self.process_log, "Prediction skipped: TensorFlow/Keras is not available in this environment."))
            self.root.after(0, lambda: self._on_batch_completed(label, len(created)))

        self._run_in_thread(action, "Unable to batch process files.")

    def _summarize_batch_sessions(self, input_files: list[dict]) -> list[dict]:
        summaries: dict[str, dict] = {}
        for item in input_files:
            summary = summaries.setdefault(
                item["session_name"],
                {
                    "session_name": item["session_name"],
                    "file_count": 0,
                    "sample_time_fine_synchro": item["sample_time_fine_synchro"],
                },
            )
            summary["file_count"] += 1
        return [summaries[key] for key in sorted(summaries)]

    def _on_batch_file_started(
        self,
        current: int,
        total: int,
        file_name: str,
        source_path: str | Path,
        output_path: str | Path,
        sync_offset: int,
    ) -> None:
        self.status_var.set(f"Batch processing: {current}/{total} - {file_name}")
        self.process_batch_progress_var.set(f"Processing {current}/{total}: {file_name}")
        self._log(self.process_log, f"[{current}/{total}] Processing: {file_name}")
        self._log(self.process_log, f"    Source: {source_path}")
        self._log(self.process_log, f"    Output: {output_path}")
        self._log(self.process_log, f"    Sync offset: {sync_offset}")

    def _on_batch_file_completed(self, current: int, total: int, path: str | Path) -> None:
        self.process_batch_progress_var.set(f"Completed {current}/{total}: {Path(path).name}")
        self._log(self.process_log, f"[{current}/{total}] Created: {path}")

    def _on_batch_completed(self, label: str, created_count: int) -> None:
        self.status_var.set(f"Batch processing completed: {created_count} files")
        self.process_batch_progress_var.set(f"Batch {label} completed: {created_count} files.")
        self._log(self.process_log, f"Batch {label} completed: {created_count} output file(s) created.")

    def _run_train(self) -> None:
        self.status_var.set("Training started...")
        self.train_log.delete("1.0", tk.END)
        self.train_quality_summary_var.set("Training in progress...")
        self._draw_placeholder_training_plot()
        self._draw_placeholder_confusion_matrix()

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
            model_overrides, batch_size = self._training_overrides_from_gui()
            summary = operations.train_model(
                task,
                self.dataset_var.get(),
                epochs,
                architecture,
                pretrained_model_id=pretrained_model_id,
                model_overrides=model_overrides,
                batch_size=batch_size,
                use_scalar_features=self.train_use_scalar_features_var.get(),
            )
            formatted_summary = self._format_training_quality_summary(summary)
            self.root.after(0, lambda: self._draw_training_history(summary))
            self.root.after(0, lambda: self._draw_confusion_matrix(summary))
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
        session_suggestion = operations.suggest_session_from_imu_file(raw_file)
        selected_session = next(
            (item for item in self._new_data_files_cache if item["session_key"] == session_suggestion["session_id"]),
            None,
        )
        workflow = (selected_session or {}).get("workflow") or {}
        if workflow.get("status") == "pending_annotation":
            if not messagebox.askyesno(
                "Synergie Tools",
                "This session is already available in pending annotation files.\n\n"
                f"Existing annotation CSV: {workflow.get('annotation_csv', 'unknown')}\n"
                "Processing again will generate another pending annotation file for the same source session.\n\n"
                "Process it again anyway?",
            ):
                return

        self.status_var.set("Processing new IMU file for annotation...")
        self.new_data_log.delete("1.0", tk.END)
        self._log(self.new_data_log, f"Selected source session: {session_suggestion['session_id']}")
        self._log(self.new_data_log, f"Automatic annotation CSV: {self.new_data_output_var.get()}")
        self._log(self.new_data_log, f"Automatic raw destination after finalization: data/raw/{session_suggestion['path']}")

        def action() -> None:
            type_path = self._process_prediction_model_path("type", self.process_type_model_var.get())
            success_path = self._process_prediction_model_path("success", self.process_success_model_var.get())
            sample_time_fine_synchro = 0
            if session_suggestion["session_id"] in operations.list_sessions():
                sample_time_fine_synchro = operations.session_synchro(session_suggestion["session_id"])
            result = operations.process_new_imu_file_for_annotation(
                raw_file,
                output_path=self.new_data_output_var.get().strip() or None,
                sample_time_fine_synchro=sample_time_fine_synchro,
                type_model_path=type_path,
                success_model_path=success_path,
            )
            self.root.after(0, lambda: self._log(self.new_data_log, f"Created annotation CSV: {result['annotation_csv']}"))
            self.root.after(0, lambda: self._log(self.new_data_log, f"Created jump segments in: {result['segment_directory']}"))
            self.root.after(0, lambda: self._log(self.new_data_log, f"Jumps ready for annotation: {result['jump_count']} across {result['sensor_count']} sensors"))
            self.root.after(0, lambda: self._log(self.new_data_log, f"Session sync offset used: {sample_time_fine_synchro}"))
            self.root.after(
                0,
                lambda: self._log(
                    self.new_data_log,
                    (
                        f"Initial model predictions: {result['prediction_status']} | "
                        f"updated={result['predictions_updated']} | skipped={result['predictions_skipped']}"
                    ),
                ),
            )
            self.root.after(
                0,
                lambda: self._log(
                    self.new_data_log,
                    "Workflow recorded: this session is now marked pending_annotation until finalization.",
                ),
            )
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
        segment_start_idx = jump.start - SEGMENT_FRAMES_BEFORE_TAKEOFF
        type_start_idx = max(0, segment_start_idx + TYPE_WINDOW_START)
        type_end_idx = min(len(session_df) - 1, type_start_idx + TYPE_WINDOW_FRAMES - 1)
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
        start_idx = max(0, jump.start - SEGMENT_FRAMES_BEFORE_TAKEOFF)
        end_idx = min(len(session_df), jump.start + len(jump.df))
        window = session_df.iloc[start_idx:end_idx]["Gyr_X_unfiltered"].abs()
        return bool((window >= GYRO_SATURATION_WARNING_THRESHOLD).any())

    def _draw_jump_windows(self, axis, session_df, jump, alpha_scale: float = 1.0) -> None:
        bounds = self._jump_window_bounds_ms(session_df, jump)
        axis.axvspan(bounds["type"][0], bounds["type"][1], color="royalblue", alpha=0.10 * alpha_scale)
        axis.axvspan(bounds["success"][0], bounds["success"][1], color="seagreen", alpha=0.10 * alpha_scale)
        axis.axvline(bounds["detected"][0], color="black", linestyle="--", linewidth=1.0)
        axis.axvline(bounds["detected"][1], color="black", linestyle=":", linewidth=1.0)
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


def _string_or_empty(value) -> str:
    return "" if value is None else str(value)


def launch() -> None:
    root = tk.Tk()
    SynergieToolsApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
