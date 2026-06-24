from __future__ import annotations

import threading
from collections.abc import Callable


def run_tk_background(root, work: Callable[[], object], on_success: Callable[[object], None], on_error: Callable[[Exception], None]) -> None:
    """Run blocking work off the Tk thread and marshal callbacks back onto Tk."""

    def runner() -> None:
        try:
            result = work()
        except Exception as exc:
            root.after(0, lambda error=exc: on_error(error))
            return
        root.after(0, lambda: on_success(result))

    threading.Thread(target=runner, daemon=True).start()
