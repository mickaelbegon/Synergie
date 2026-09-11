from __future__ import annotations


ANNOTATION_JUMP_TYPE_SHORTCUTS = {
    "t": "toe_loop",
    "f": "flip",
    "z": "lutz",
    "s": "salchow",
    "a": "axel",
}

ANNOTATION_SUCCESS_SHORTCUTS = {"c": "0", "r": "1", "n": "2"}

ANNOTATION_SHORTCUTS_HELP_TEXT = (
    "Shortcuts: t/f/z/s/a = jump type | 1-4 = turns\n"
    "c/r/n = chute/reussi/inconnu | u = unseen | x = weird signal\n"
    "Space = play/pause | \u2190/\u2192 = frame | l = legend | Ctrl+S = save"
)


def resolve_annotation_shortcut(key: str, *, ctrl: bool = False, turn_options=None) -> dict | None:
    """Resolve one normalized keypress into a GUI annotation action."""
    normalized = str(key or "").strip().lower()
    if not normalized:
        return None
    if normalized == "space":
        return {"action": "play_pause"}
    if normalized == "left":
        return {"action": "seek_frame", "delta": -1}
    if normalized == "right":
        return {"action": "seek_frame", "delta": 1}
    if normalized == "s" and ctrl:
        return {"action": "save"}
    if normalized == "l":
        return {"action": "toggle_legend"}
    if normalized in ANNOTATION_JUMP_TYPE_SHORTCUTS:
        return {"action": "jump_type", "value": ANNOTATION_JUMP_TYPE_SHORTCUTS[normalized]}
    if normalized == "u":
        return {"action": "review_status", "value": "not_seen_on_video", "status": "unseen on video"}
    if normalized == "x":
        return {"action": "review_status", "value": "weird_signal", "status": "weird signal / bad bounds"}
    if normalized in {"1", "2", "3", "4"}:
        options = set(turn_options or [])
        if normalized in options:
            return {"action": "turns", "value": normalized}
        return None
    if normalized in ANNOTATION_SUCCESS_SHORTCUTS:
        return {"action": "success", "value": ANNOTATION_SUCCESS_SHORTCUTS[normalized]}
    return None
