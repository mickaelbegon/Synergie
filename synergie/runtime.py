from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path


def _ensure_writable_tempdir() -> None:
    current_temp = Path(tempfile.gettempdir())
    try:
        current_temp.mkdir(parents=True, exist_ok=True)
        probe_dir = Path(tempfile.mkdtemp(dir=current_temp))
        probe_path = probe_dir / "synergie-temp-write-check.tmp"
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink(missing_ok=True)
        probe_dir.rmdir()
        return
    except OSError:
        pass

    fallback_temp = Path(__file__).resolve().parent.parent / ".tmp" / "runtime"
    fallback_temp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(fallback_temp)
    os.environ["TMP"] = str(fallback_temp)
    os.environ["TMPDIR"] = str(fallback_temp)
    tempfile.tempdir = str(fallback_temp)


def configure_runtime() -> None:
    _ensure_writable_tempdir()

    # Hide TensorFlow CPU capability info logs in normal usage.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Silence google-api-core Python end-of-support warning while the local
    # environment is still being migrated to Python 3.11.
    warnings.filterwarnings(
        "ignore",
        message=r"You are using a Python version .*",
        category=FutureWarning,
        module=r"google\.api_core\._python_version_support",
    )


configure_runtime()
