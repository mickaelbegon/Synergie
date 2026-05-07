from __future__ import annotations

import os
import warnings


def configure_runtime() -> None:
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
