"""
Shared Python workspace.

Holds a single :class:`Workspace` instance that both the block diagram's
parameter evaluator and the MATLAB-style workspace widget read from and
write to. Putting this in its own module avoids a circular import
between ``stm32_block_ide.py`` and ``matlab_workspace.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


class Workspace:
    """Holds Python variables that can be referenced by block parameters
    and that are displayed in the MATLAB-style Workspace widget.

    Blocks may specify a parameter as a literal (e.g. ``1000``) or as a
    workspace symbol (e.g. ``Ts`` or ``f_pwm * 2``). At code-generation
    time each expression is ``eval``'d against ``globals``.
    """

    def __init__(self) -> None:
        self.globals: Dict[str, Any] = {
            "np": np, "math": math, "pi": math.pi,
        }

    def eval_param(self, expr: str) -> Any:
        """Evaluate a parameter string against the workspace.

        Numeric literals pass through; anything else is ``eval``'d with
        numpy and math available.
        """
        expr = str(expr).strip()
        if not expr:
            return 0
        try:
            if "." in expr or "e" in expr.lower():
                return float(expr)
            return int(expr)
        except ValueError:
            pass
        try:
            return eval(expr, self.globals, {})
        except Exception as exc:
            raise ValueError(f"Cannot resolve '{expr}': {exc}")


# Single global instance shared throughout the application.
WORKSPACE = Workspace()
