# GUI Refactor Architecture

This note documents the intended direction for `tools_gui.py` and the services used by the annotation interface.

## Goal

Keep `synergie/tool_gui.py` focused on UI orchestration:

- create Tk widgets
- read widget values
- call service functions through `synergie.operations`
- display results, warnings, and plots

Move data rules and calculations into services under `synergie/services/` so they can be tested without opening the GUI.

## Current Service Boundaries

- `add_jump_service.py`: manual ADD JUMP candidate detection, manual segment building, manual annotation rows, and timeline ordering.
- `annotation_service.py`: annotation metadata, label/status mappings, robust CSV-to-widget value mapping, row updates, and CSV persistence when saving GUI annotation values.
- `annotation_shortcut_service.py`: annotation keyboard-shortcut mapping and the Annotation header help text.
- `annotation_sync_service.py`: video/IMU time conversion, block sync offsets, review-video target selection, and sync status text.
- `annotation_timeline_service.py`: rows displayed in the Data | Annotate timeline.
- `annotation_video_player.py`: OpenCV video capture wrapper used by the annotation player.
- `annotation_source_service.py`: lookup of raw source CSV files for annotation entries.
- `annotation_plot_service.py`: robust annotation segment path parsing, loading and preparing annotation segment signals before Matplotlib drawing, including lookup of the first IMU timestamp used by the jump-type model.
- `annotation_impact_repair_service.py`: re-detection of IMU impact offsets in annotation files.
- `formatting_service.py`: display formatting and parsing for video timestamps and file sizes.
- `numeric_utils.py`: shared numeric conversion helpers.
- `data_validation_service.py`: cross-folder checks and next-action summaries for data consistency.
- `detection_explanation_service.py`: threshold-crossing and biomechanical diagnostics for why a jump was detected.
- `inspect_signal_service.py`: pure Inspect IMU axis/legend/text/signal-series specs plus drag-zoom selection, jump-window, center marker, zoom-window, range-selection, and gyroscope-saturation calculations.
- `sync_impact_service.py`: sync-impact detection plus raw-session loading and acceleration-window preparation for review plots.

`synergie.operations` is the stable facade imported by the GUI. New GUI-facing services should usually be re-exported there.

## Refactor Rules

1. Add or update a unit test before moving logic.
2. Extract pure functions first: no Tk widgets, no message boxes, no direct plotting.
3. Keep file IO in services only when the behavior is part of the workflow and can be tested with temporary files.
4. Keep `tool_gui.py` wrappers temporarily when that reduces risk.
5. Run targeted tests, then the full suite.

Useful commands:

```sh
python -m unittest tests.test_add_jump_service tests.test_annotation_sync_service tests.test_formatting_service
python -m unittest tests.test_gui_smoke
python -m unittest discover -s tests
```

## Remaining Refactor Targets

- Matplotlib drawing in `Data | Annotate` and `Review | Inspect IMU`.
- Video cache and optimization progress plumbing.
- Data validation presentation and repair actions.
- Annotation file load/save orchestration.
- Inspect IMU selection/zoom state.

For each target, prefer small service functions plus tests rather than one large move.

## Smoke Coverage

`tests/test_gui_smoke.py` verifies that the GUI module imports without starting a Tk mainloop and that key service helpers remain exported through `synergie.operations`.
