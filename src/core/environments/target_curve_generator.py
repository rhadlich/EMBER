"""
Generates IMEP target profiles as sequences of random flat holds connected
by smooth transitions (linear, quadratic, or blends).

The generator produces a single stream of targets -- training and evaluation
draw from the same distribution.  Roughly 40 % of the time is spent at
randomly located flat holds and ~60 % on transitions between them.

Each call to ``next()`` advances by one step.  When the current segment
(one flat hold + one transition) is exhausted, a new segment is generated
automatically.

Passing a *seed* makes the entire sequence fully reproducible.
"""

from __future__ import annotations

import numpy as np


class IMEPTargetCurveGenerator:
    """Continuous IMEP target-profile generator.

    Produces segments of [flat hold | transition curve].  When a segment is
    consumed, a new one is built starting from where the last one ended.
    Both training and evaluation use the same distribution.

    Parameters
    ----------
    low, high : float
        IMEP bounds (inclusive).
    seed : int | None
        RNG seed for reproducibility.
    min_hold_len, max_hold_len : int
        Range of hold durations (in steps) at each flat level.
    min_transition_len, max_transition_len : int
        Range of transition durations (in steps) between flats.
    """

    _CURVE_TYPES = ("linear", "quadratic_accel", "quadratic_decel", "blend")

    def __init__(
        self,
        low: float = 1.6,
        high: float = 4.1,
        seed: int | None = None,
        min_hold_len: int = 15,
        max_hold_len: int = 60,
        min_transition_len: int = 20,
        max_transition_len: int = 90,
    ) -> None:
        self.low = low
        self.high = high
        self.min_hold_len = min_hold_len
        self.max_hold_len = max_hold_len
        self.min_transition_len = min_transition_len
        self.max_transition_len = max_transition_len

        self._rng = np.random.default_rng(seed)
        self._mode = "train"

        self._segment: list[float] = []
        self._seg_len: int = 0
        self._index: int = 0
        self._current_val: float = self._rng.uniform(self.low, self.high)

        self._generate_segment()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next(self) -> float:
        """Return the next target value, generating a new segment when needed."""
        if self._index >= self._seg_len:
            self._generate_segment()
        value = self._segment[self._index]
        self._index += 1
        return value

    def current(self) -> float:
        """Return the current target value without advancing."""
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
        """Build one segment: flat hold at current value, then transition
        to a new random value."""
        hold_len = int(
            self._rng.integers(self.min_hold_len, self.max_hold_len + 1)
        )
        next_val = float(self._rng.uniform(self.low, self.high))
        trans_len = int(
            self._rng.integers(
                self.min_transition_len, self.max_transition_len + 1
            )
        )

        seg: list[float] = [self._current_val] * hold_len
        transition = self._make_transition(
            self._current_val, next_val, trans_len
        )
        seg.extend(transition.tolist())

        self._segment = seg
        self._seg_len = len(seg)
        self._index = 0
        self._current_val = next_val

    # ------------------------------------------------------------------
    # Transition builders
    # ------------------------------------------------------------------

    def _make_transition(
        self, start: float, end: float, length: int
    ) -> np.ndarray:
        """Create a transition curve of a randomly chosen type."""
        curve_type = self._rng.choice(self._CURVE_TYPES)
        t = np.linspace(0.0, 1.0, length)

        if curve_type == "linear":
            alpha = t
        elif curve_type == "quadratic_accel":
            alpha = t**2
        elif curve_type == "quadratic_decel":
            alpha = 1.0 - (1.0 - t) ** 2
        else:  # blend: weighted mix of linear and quadratic
            w = self._rng.uniform(0.2, 0.8)
            quad = t**2 if self._rng.random() < 0.5 else 1.0 - (1.0 - t) ** 2
            alpha = w * t + (1.0 - w) * quad

        return start + (end - start) * alpha
