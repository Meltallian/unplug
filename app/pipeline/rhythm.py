"""Beats, downbeats and the bar grid.

beat_this (CPJKU, 2024) gives beat and downbeat times. Its raw downbeats are noisy (1-, 2- and
3-beat "bars" appear), so fit_grid() lays a regular grid over the beats: every beat gets a
position in its bar (0 = the "1"), bars are `meter` beats long, and a bar may be cut short only
at a cost. The meter is 4 unless 3 or 6 fits the downbeats clearly better (6 = a 6/8 song whose
beats the model counts in eighth notes).
"""

import os
from functools import lru_cache

import numpy as np

CHECKPOINT = os.environ.get("BEAT_THIS_CHECKPOINT", "final0")
DOWNBEAT_TOL = 0.07   # s: a beat counts as a detected downbeat if one is this close
SHORT_BAR_COST = 3.0  # cost of cutting a bar short (a missed beat, an odd bar)
MIN_GAP = 0.08        # s: beats closer than this are duplicates
METERS = (4, 3, 6)    # 4 first: the default
# Another meter must fit the downbeats better than it contradicts them (score > 0: random
# downbeats never reach that, at most -0.12 per beat in 300 trials) and beat 4/4 by this much
# per beat (random downbeats give up to ~0.05 in favour of 3).
METER_MARGIN = 0.05


FPS = 50                 # frames per second of the tracker's activations
ACTIVATIONS_VERSION = "a1"


@lru_cache(maxsize=1)
def _network():
    import torch
    from beat_this.inference import Audio2Frames

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # two songs may run at once
    return Audio2Frames(checkpoint_path=CHECKPOINT, device="cpu")


def activations(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat activations (logits, FPS frames per second): the slow part of tracking."""
    beat, down = _network()(y.astype(np.float32), sr)
    return beat.numpy().astype(np.float32), down.numpy().astype(np.float32)


def pick(beat_act: np.ndarray, down_act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times in seconds from the activations (peaks above 50 %)."""
    import torch
    from beat_this.model.postprocessor import Postprocessor

    beats, downs = Postprocessor(type="minimal", fps=FPS)(torch.from_numpy(beat_act), torch.from_numpy(down_act))
    beats = np.asarray(beats, dtype=float)
    if len(beats):
        beats = beats[np.concatenate([[True], np.diff(beats) > MIN_GAP])]
    return beats, np.asarray(downs, dtype=float)


def track(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times in seconds."""
    return pick(*activations(y, sr))


def cached_activations(wav, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """activations(), kept next to the wav so that later stages and experiments never re-run the network."""
    cache = wav.parent / f"beat-activations-{ACTIVATIONS_VERSION}.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                return z["beat"].astype(np.float32), z["downbeat"].astype(np.float32)
        except Exception:  # corrupt or partial cache: recompute
            pass
    beat, down = activations(y, sr)
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, beat=beat.astype(np.float16), downbeat=down.astype(np.float16))
        tmp.replace(cache)
    finally:
        tmp.unlink(missing_ok=True)
    return beat, down


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
    """(meter, position of every beat in its bar). Meter is 4, 3 or 6."""
    if len(beats) == 0:
        return 4, np.zeros(0, dtype=int)
    if len(downbeats):
        idx = np.searchsorted(downbeats, beats)
        lo = downbeats[np.clip(idx - 1, 0, len(downbeats) - 1)]
        hi = downbeats[np.clip(idx, 0, len(downbeats) - 1)]
        near = np.minimum(np.abs(beats - lo), np.abs(beats - hi)) < DOWNBEAT_TOL
    else:
        near = np.zeros(len(beats), dtype=bool)
    fits = {m: _fit(near, m) for m in METERS}
    best = 4
    for m in METERS[1:]:
        score = fits[m][0]
        if score > 0 and score > fits[best][0] and score > fits[4][0] + max(2.0, METER_MARGIN * len(beats)):
            best = m
    return best, fits[best][1]


def time_signature(meter: int) -> str:
    return {3: "3/4", 6: "6/8"}.get(meter, "4/4")


def tempo(beats: np.ndarray) -> float:
    return float(60.0 / np.median(np.diff(beats))) if len(beats) > 1 else 0.0
