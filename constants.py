from enum import Enum

from synergie.session_store import load_sessions

treshold = -0.2

modeltype_filepath = "core/model/saved_models/checkpoint.keras"
modelsuccess_filepath = "core/model/saved_models/success.keras"

fields_to_keep = ["Euler_X","Euler_Y","Euler_Z","Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z", "Combination"]

sessions = load_sessions()


def get_session(session_name: str) -> dict:
    """Return metadata for a configured recording session."""
    try:
        return sessions[session_name]
    except KeyError as exc:
        available = ", ".join(sorted(sessions))
        raise KeyError(f"Unknown session '{session_name}'. Available sessions: {available}") from exc

class jumpType(Enum):
    """
    figure skating jump type enum
    """

    # toe jumps
    TOE_LOOP = 0
    FLIP = 1
    LUTZ = 2

    # edge jumps
    SALCHOW = 3
    LOOP = 4
    AXEL = 5

    NONE = 8  # none is intended to be used when the annotation could not be completed (ice skater off frame at the time of the jump)

class jumpSuccess(Enum):
    FALL = 0
    SUCCESS = 1
    NONE = 2 # used before the annotation
