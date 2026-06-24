from __future__ import annotations


INSPECT_AXIS_LABELS = {
    "x": "ms",
    "left_y": "Gyroscope",
    "right_y": "2nd derivative",
}

INSPECT_JUMP_LEGEND_SPECS = [
    {"color": "royalblue", "linewidth": 6, "alpha": 0.35, "label": "Type window"},
    {"color": "seagreen", "linewidth": 6, "alpha": 0.35, "label": "Success window"},
    {"color": "crimson", "linewidth": 2, "linestyle": ":", "label": "Gyro saturation"},
    {"color": "darkorange", "linewidth": 1.5, "linestyle": "-.", "label": "Sync impact"},
]

INSPECT_CENTER_MARKER_LEGEND_SPEC = {
    "marker": "o",
    "markersize": 6,
    "color": "black",
    "markerfacecolor": "white",
    "linestyle": "None",
    "label": "Detected jump center",
}

INSPECT_SELECTED_MARKER_LEGEND_SPEC = {
    "marker": "*",
    "markersize": 12,
    "color": "gold",
    "markeredgecolor": "black",
    "linestyle": "None",
    "label": "Selected jump",
}

INSPECT_TEXT = {
    "placeholder_overview_title": "Overview of IMU jump localisation signals",
    "placeholder_zoom_title": "Zoom on selected jump",
    "placeholder_overview_message": "Load a CSV in Inspect IMU",
    "placeholder_zoom_message": "Select a detected jump to zoom",
    "overview_title": "Signals used to localise jumps",
    "zoom_title": "Zoom on selected jump",
    "zoom_select_message": "Select a jump in the list",
    "zoom_empty_jump_message": "Selected jump range contains no data",
    "zoom_empty_range_message": "Selected range contains no data",
    "zoom_range_title": "Zoom on selected range",
    "sync_impact_title": "Sync impact acceleration",
}


def inspect_text(key: str) -> str:
    """Return a standard Inspect IMU UI string."""
    return INSPECT_TEXT[str(key)]


def inspect_selected_jump_title(jump_number: int, *, has_gyro_saturation: bool = False) -> str:
    """Return the zoom title for a selected detected jump."""
    saturation_text = " | Gyro saturated" if has_gyro_saturation else ""
    return f"Zoom on selected jump #{int(jump_number)}{saturation_text}"


def inspect_zoom_range_title(start_ms: float, end_ms: float) -> str:
    """Return the title for a manually selected Inspect IMU zoom range."""
    return f"Zoom on selected range: {float(start_ms):.0f}-{float(end_ms):.0f} ms"


def inspect_signal_series_specs(dataframe) -> list[dict]:
    """Return plot series specs for Inspect IMU gyroscope localization signals."""
    return [
        {
            "axis": "left",
            "x": dataframe["ms"],
            "y": dataframe["Gyr_X_unfiltered"],
            "style": {"label": "Gyr_X raw", "linewidth": 0.8, "alpha": 0.4},
        },
        {
            "axis": "left",
            "x": dataframe["ms"],
            "y": dataframe["Gyr_X_smoothed"],
            "style": {"label": "Gyr_X smoothed", "linewidth": 1.3},
        },
        {
            "axis": "right",
            "x": dataframe["ms"],
            "y": dataframe["X_gyr_second_derivative"],
            "style": {"label": "2nd derivative", "linewidth": 1.0, "color": "crimson"},
        },
    ]


def inspect_detection_threshold_spec(threshold: float) -> dict:
    """Return the Inspect IMU detection-threshold line spec."""
    return {
        "y": float(threshold),
        "style": {"color": "crimson", "linestyle": "--", "label": "Detection threshold"},
    }


def inspect_axis_labels() -> dict[str, str]:
    """Return the standard axis labels for Inspect IMU plots."""
    return dict(INSPECT_AXIS_LABELS)


def inspect_jump_legend_specs(*, include_center: bool = False, include_selected: bool = False) -> list[dict]:
    """Return Matplotlib-compatible legend placeholder specs for Inspect IMU plots."""
    specs = [dict(spec) for spec in INSPECT_JUMP_LEGEND_SPECS]
    if include_center:
        specs.append(dict(INSPECT_CENTER_MARKER_LEGEND_SPEC))
    if include_selected:
        specs.append(dict(INSPECT_SELECTED_MARKER_LEGEND_SPEC))
    return specs


def inspect_jump_center_markers(
    session_df,
    jumps,
    *,
    segment_frames_before_takeoff: int,
    type_window_start: int,
    type_window_frames: int,
) -> list[dict]:
    """Return overview marker positions for detected jumps."""
    markers: list[dict] = []
    for index, jump in enumerate(jumps):
        center_ms = inspect_jump_center_ms(
            session_df,
            jump,
            segment_frames_before_takeoff=segment_frames_before_takeoff,
            type_window_start=type_window_start,
            type_window_frames=type_window_frames,
        )
        center_index = jump.start + ((jump.end - jump.start) // 2)
        center_index = max(0, min(len(session_df) - 1, int(center_index)))
        markers.append(
            {
                "index": index,
                "center_ms": center_ms,
                "center_y": float(session_df.iloc[center_index]["Gyr_X_smoothed"]),
            }
        )
    return markers


def inspect_jump_zoom_view(session_df, jump, *, padding_frames: int = 140):
    """Return the dataframe slice and x limits for a selected jump zoom."""
    start_idx = max(0, int(jump.start) - int(padding_frames))
    end_idx = min(len(session_df) - 1, int(jump.end) + int(padding_frames))
    view_df = session_df.iloc[start_idx : end_idx + 1]
    if view_df.empty:
        return {"view_df": view_df, "x_min_ms": None, "x_max_ms": None}
    return {
        "view_df": view_df,
        "x_min_ms": float(view_df["ms"].iloc[0]),
        "x_max_ms": float(view_df["ms"].iloc[-1]),
    }


def inspect_zoom_range_view(session_df, start_ms: float, end_ms: float):
    """Return the dataframe slice and normalized x limits for a selected time range."""
    x_min_ms, x_max_ms = sorted((float(start_ms), float(end_ms)))
    view_df = session_df[(session_df["ms"] >= x_min_ms) & (session_df["ms"] <= x_max_ms)]
    return {"view_df": view_df, "x_min_ms": x_min_ms, "x_max_ms": x_max_ms}


def inspect_drag_zoom_selection(start_ms: float, end_ms: float | None, *, min_width_ms: float = 5.0) -> dict:
    """Return the zoom selection represented by an overview drag gesture."""
    if end_ms is None:
        return {"accepted": False, "range_ms": None, "status": ""}
    start_value = float(start_ms)
    end_value = float(end_ms)
    if abs(end_value - start_value) < float(min_width_ms):
        return {"accepted": False, "range_ms": None, "status": ""}
    range_ms = tuple(sorted((start_value, end_value)))
    return {
        "accepted": True,
        "range_ms": range_ms,
        "status": f"Inspect IMU zoom: {range_ms[0]:.0f}-{range_ms[1]:.0f} ms",
    }


def inspect_jump_window_bounds_ms(
    session_df,
    jump,
    *,
    segment_frames_before_takeoff: int,
    type_window_start: int,
    type_window_frames: int,
) -> dict[str, tuple[float, float]]:
    """Return the relevant Inspect IMU time windows for one detected jump."""
    segment_start_idx = jump.start - int(segment_frames_before_takeoff)
    type_start_idx = max(0, segment_start_idx + int(type_window_start))
    type_end_idx = min(len(session_df) - 1, type_start_idx + int(type_window_frames) - 1)
    success_start_idx = max(0, jump.start)
    success_end_idx = min(len(session_df) - 1, jump.start + (len(jump.df_success) - 1))
    return {
        "type": (float(session_df.iloc[type_start_idx]["ms"]), float(session_df.iloc[type_end_idx]["ms"])),
        "success": (float(session_df.iloc[success_start_idx]["ms"]), float(session_df.iloc[success_end_idx]["ms"])),
        "detected": (float(session_df.iloc[jump.start]["ms"]), float(session_df.iloc[jump.end]["ms"])),
    }


def inspect_jump_center_ms(
    session_df,
    jump,
    *,
    segment_frames_before_takeoff: int,
    type_window_start: int,
    type_window_frames: int,
) -> float:
    """Return the midpoint of the detected jump window in milliseconds."""
    bounds = inspect_jump_window_bounds_ms(
        session_df,
        jump,
        segment_frames_before_takeoff=segment_frames_before_takeoff,
        type_window_start=type_window_start,
        type_window_frames=type_window_frames,
    )
    return (bounds["detected"][0] + bounds["detected"][1]) / 2.0


def inspect_jump_has_gyro_saturation(
    session_df,
    jump,
    *,
    segment_frames_before_takeoff: int,
    saturation_threshold: float,
) -> bool:
    """Return whether the jump window includes saturated gyroscope values."""
    if "Gyr_X_unfiltered" not in session_df:
        return False
    start_idx = max(0, jump.start - int(segment_frames_before_takeoff))
    end_idx = min(len(session_df), jump.start + len(jump.df))
    window = session_df.iloc[start_idx:end_idx]["Gyr_X_unfiltered"].abs()
    return bool((window >= float(saturation_threshold)).any())
