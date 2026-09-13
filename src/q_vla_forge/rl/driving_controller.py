"""Simple baseline controller for the driving RL proxy."""

from __future__ import annotations

import numpy as np


def zero_action_controller(
    observation: np.ndarray,
) -> np.ndarray:
    """Return a deterministic no-op driving action."""
    del observation

    return np.array(
        [
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
