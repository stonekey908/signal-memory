"""Half-life decay model (split model — housekeeping only).

Produces a **filing/housekeeping weight** for a signal (how "live" it is for
consolidation/archival — Wave 2), **NOT** a retrieval multiplier (the AI judge weighs
recency at retrieval, STO-2724). Pins bypass decay; the weight is floored so a faded
signal is never 0 — still searchable, never deleted.

Two adjustable dials:

* ``half_life`` (in days, or whatever age unit the caller uses) — how long until influence
  halves. The Wave-3 tuning dial (STO-2734).
* the **curve** — a pluggable ``DecayCurve``: ``ExponentialDecay`` (with a steepness knob) or
  ``LinearDecay``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from signal_engine.signal import Signal

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_FLOOR = 0.01


@runtime_checkable
class DecayCurve(Protocol):
    def factor(self, age: float, half_life: float) -> float:
        """Decay multiplier in (0, 1] for a given age and half-life (same unit)."""
        ...


@dataclass(frozen=True)
class ExponentialDecay:
    """Classic half-life fade: ``0.5 ** ((age/half_life) ** steepness)``.

    ``steepness = 1.0`` is the standard exponential. >1 = gentler early then sharper;
    <1 = sharper early then gentler. The half-life point (``age == half_life`` → 0.5) is
    invariant to steepness.
    """

    steepness: float = 1.0

    def factor(self, age: float, half_life: float) -> float:
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        if age <= 0:
            return 1.0
        return 0.5 ** ((age / half_life) ** self.steepness)


@dataclass(frozen=True)
class LinearDecay:
    """Straight-line fade: ``1 - 0.5*(age/half_life)``, clamped to [0, 1].

    Reaches 0.5 at one half-life and 0 at two half-lives (the model's floor then keeps
    the weight > 0).
    """

    def factor(self, age: float, half_life: float) -> float:
        if half_life <= 0:
            raise ValueError("half_life must be > 0")
        if age <= 0:
            return 1.0
        return max(0.0, 1.0 - 0.5 * (age / half_life))


@dataclass
class DecayModel:
    """Turns age into a filing weight, using a pluggable curve + adjustable half-life."""

    curve: DecayCurve = field(default_factory=ExponentialDecay)
    half_life: float = DEFAULT_HALF_LIFE_DAYS
    floor: float = DEFAULT_FLOOR

    def filing_weight(self, signal: Signal, age: float) -> float:
        """How 'live' a signal is for filing/consolidation. Pins never fade; result floored."""
        if signal.pinned:
            return signal.importance_base          # pins bypass decay entirely
        decayed = signal.importance_base * self.curve.factor(age, self.half_life)
        return max(decayed, self.floor)
