"""
Generates realistic IMEP target profiles composed of linear ramps, quadratic
curves, and occasional step drops (simulating abrupt throttle release).

Segments are generated lazily one at a time.  Each new segment starts where
the previous one ended, so the target evolves smoothly -- except when a step
drop is selected, which produces an instantaneous downward jump.  Passing a
*seed* makes the entire sequence fully reproducible.
"""

from __future__ import annotations

import numpy as np


class IMEPTargetCurveGenerator:
    """Continuous IMEP target-profile generator.

    Parameters
    ----------
    low, high : float
        IMEP bounds (inclusive).
    seed : int | None
        RNG seed for reproducibility.
    step_drop_probability : float
        Probability that any given segment is a step drop rather than a
        smooth ramp.
    min_segment_len, max_segment_len : int
        Range of segment lengths (in rollouts) for linear / quadratic pieces.
    """

    def __init__(
        self,
        low: float = 1.6,
        high: float = 4.1,
        seed: int | None = None,
        step_drop_probability: float = 0.15,
        min_segment_len: int = 20,
        max_segment_len: int = 80,
    ) -> None:
        self.low = low
        self.high = high
        self.step_drop_probability = step_drop_probability
        self.min_segment_len = min_segment_len
        self.max_segment_len = max_segment_len

        self._rng = np.random.default_rng(seed)
        self._segment: list[float] = []
        self._seg_len: int = 0
        self._index: int = 0
        self._current_val: float = self._rng.uniform(self.low, self.high)
        self._generate_segment()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Return the next target value, generating a new segment when
        the current one is exhausted."""
        if self._index >= self._seg_len:
            self._generate_segment()
        value = self._segment[self._index]
        self._index += 1
        return value

    def current(self) -> float:
        """Return the current target value without advancing the index."""
        if self._index >= self._seg_len:
            self._generate_segment()
        return self._segment[self._index]

    def reset(self, seed: int | None = None) -> None:
        """Re-initialise the generator, optionally with a new seed."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_val = self._rng.uniform(self.low, self.high)
        self._generate_segment()

    # ------------------------------------------------------------------
    # Segment construction
    # ------------------------------------------------------------------

    def _generate_segment(self) -> None:
        """Build the next segment, continuing from the last value."""
        is_step_drop = self._rng.random() < self.step_drop_probability

        if is_step_drop:
            seg = self._make_step_drop(self._current_val)
        else:
            kind = self._rng.choice(["linear", "quadratic"])
            seg_len = int(self._rng.integers(
                self.min_segment_len,
                self.max_segment_len + 1,
            ))
            end_val = self._rng.uniform(self.low, self.high)

            if kind == "linear":
                seg = self._make_linear(self._current_val, end_val, seg_len)
            else:
                seg = self._make_quadratic(self._current_val, end_val, seg_len)

        self._segment = seg.tolist()
        self._seg_len = len(self._segment)
        self._current_val = self._segment[-1]
        self._index = 0

    # ---- segment builders ------------------------------------------------

    @staticmethod
    def _make_linear(start: float, end: float, length: int) -> np.ndarray:
        return np.linspace(start, end, length)

    def _make_quadratic(
        self, start: float, end: float, length: int
    ) -> np.ndarray:
        t = np.linspace(0.0, 1.0, length)
        # Randomly choose between "ease-in" (t^2) and "ease-out" (1-(1-t)^2)
        if self._rng.random() < 0.5:
            alpha = t ** 2
        else:
            alpha = 1.0 - (1.0 - t) ** 2
        return start + (end - start) * alpha

    def _make_step_drop(self, current_val: float) -> np.ndarray:
        """Instantaneous drop followed by a short hold."""
        drop_target = self._rng.uniform(self.low, current_val)

        hold_len = int(self._rng.integers(
            max(1, self.min_segment_len // 4),
            max(2, self.min_segment_len // 2) + 1,
        ))
        return np.full(hold_len, drop_target)
