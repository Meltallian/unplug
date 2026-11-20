"""Beats, downbeats and the bar grid.

beat_this (CPJKU, 2024) gives beat and downbeat times. Its raw downbeats are noisy (1-, 2- and
3-beat "bars" appear), so fit_grid() lays a regular grid over the beats: every beat gets a
position in its bar (0 = the "1"), bars are `meter` beats long, and a bar may be cut short only
at a cost. The meter is 4 unless 3 fits the downbeats clearly better.
"""

import os
from functools import lru_cache

import numpy as np

CHECKPOINT = os.environ.get("BEAT_THIS_CHECKPOINT", "final0")
DOWNBEAT_TOL = 0.07   # s: a beat counts as a detected downbeat if one is this close
SHORT_BAR_COST = 3.0  # cost of cutting a bar short (a missed beat, an odd bar)
MIN_GAP = 0.08        # s: beats closer than this are duplicates


@lru_cache(maxsize=1)
def _tracker():
    import torch
    from beat_this.inference import Audio2Beats

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # two songs may run at once
    return Audio2Beats(checkpoint_path=CHECKPOINT, device="cpu", dbn=False)


def track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times in seconds."""
    beats, downs = _tracker()(y.astype(np.float32), sr)
    beats = np.asarray(beats, dtype=float)
    if len(beats):
        beats = beats[np.concatenate([[True], np.diff(beats) > MIN_GAP])]
    return beats, np.asarray(downs, dtype=float)


def _fit(near: np.ndarray, meter: int) -> tuple[float, np.ndarray]:
    """Viterbi over bar positions. Score: +1 for a grid "1" on a detected downbeat, -1 for a grid "1"
    without one or a detected downbeat elsewhere."""
    n = len(near)
    emit = np.zeros((n, meter))
    emit[:, 0] = np.where(near, 1.0, -1.0)
    emit[:, 1:] = np.where(near, -1.0, 0.0)[:, None]
    score = emit[0].copy()
    back = np.zeros((n, meter), dtype=int)
    for i in range(1, n):
        prev = np.roll(np.arange(meter), 1)          # normal step: position p-1 → p
        new = score[prev].copy()
        back[i] = prev
        if meter > 2:                                 # cut a bar short: any middle position → "1"
            mids = np.arange(1, meter - 1)
            best = mids[np.argmax(score[mids])]
            if score[best] - SHORT_BAR_COST > new[0]:
                new[0], back[i, 0] = score[best] - SHORT_BAR_COST, best
        score = new + emit[i]
    pos = np.zeros(n, dtype=int)
    pos[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return float(score.max()), pos


def fit_grid(beats: np.ndarray, downbeats: np.ndarray) -> tuple[int, np.ndarray]:
    """(meter, position of every beat in its bar). Meter is 4 or 3."""
    if len(beats) == 0:
        return 4, np.zeros(0, dtype=int)
    if len(downbeats):
        idx = np.searchsorted(downbeats, beats)
        lo = downbeats[np.clip(idx - 1, 0, len(downbeats) - 1)]
        hi = downbeats[np.clip(idx, 0, len(downbeats) - 1)]
        near = np.minimum(np.abs(beats - lo), np.abs(beats - hi)) < DOWNBEAT_TOL
    else:
        near = np.zeros(len(beats), dtype=bool)
    s4, p4 = _fit(near, 4)
    s3, p3 = _fit(near, 3)
    return (3, p3) if s3 > s4 else (4, p4)


def tempo(beats: np.ndarray) -> float:
    return float(60.0 / np.median(np.diff(beats))) if len(beats) > 1 else 0.0
