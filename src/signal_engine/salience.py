"""Salience filter (write-time throwaway rejection).

The keep-vs-throwaway gate between extraction (STO-2720) and storage: drops
non-durable signals so chit-chat and content-free notes never get persisted —
which keeps tokens/query down and lifts retrieval precision.

Mechanism (``docs/DECISIONS.md``): a threshold on the scribbler-assigned
``importance_base``. **Err toward keep** (low default threshold); **pins are never
dropped**; **every drop is logged** with a reason (over-filtering is never silent).
The threshold is the single tunable "aggressiveness" dial (tuned in Wave 3, STO-2734).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Sequence

from signal_engine.signal import Signal

DEFAULT_KEEP_THRESHOLD = 0.2   # low -> err toward keep

_log = logging.getLogger("signal_engine.salience")


@dataclass
class DropRecord:
    signal: Signal
    reason: str


@dataclass
class FilterResult:
    kept: List[Signal] = field(default_factory=list)
    dropped: List[DropRecord] = field(default_factory=list)

    @property
    def n_kept(self) -> int:
        return len(self.kept)

    @property
    def n_dropped(self) -> int:
        return len(self.dropped)


class SalienceFilter:
    """Keep-vs-throwaway gate. Drops low-``importance_base`` signals; pins are exempt.

    ``threshold`` is the single tunable aggressiveness dial: higher drops more; the
    default is deliberately low to **err toward keep**.
    """

    def __init__(self, threshold: float = DEFAULT_KEEP_THRESHOLD):
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def keep(self, signal: Signal) -> bool:
        """A signal is kept if it is pinned or clears the importance threshold."""
        return signal.pinned or signal.importance_base >= self.threshold

    def filter(self, signals: Sequence[Signal]) -> FilterResult:
        result = FilterResult()
        for s in signals:
            if self.keep(s):
                result.kept.append(s)
            else:
                reason = (
                    f"importance_base {s.importance_base:.2f} < keep-threshold {self.threshold:.2f}"
                )
                result.dropped.append(DropRecord(signal=s, reason=reason))
                _log.info("salience drop: kind=%s title=%r — %s", s.kind, s.title, reason)
        return result
