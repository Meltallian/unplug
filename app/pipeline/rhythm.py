"""Beats, downbeats, the bar grid and the tempo.

beat_this (CPJKU, 2024) gives beat and downbeat times. Its raw downbeats are noisy (1-, 2- and
3-beat "bars" appear), so fit_grid() lays a regular grid over the beats: every beat gets a
position in its bar (0 = the "1"), bars are `meter` beats long, and a bar may be cut short only
at a cost. The meter is 4 unless 3 or 6 fits the downbeats clearly better (6 = a 6/8 song whose
beats the model counts in eighth notes).

bar_grid() then settles what a guitarist counts: fast beats in threes are the eighth notes of
6/8 (two slow beats per bar), not a waltz at 150 bpm. The tempo is always the tempo of the beat
the time signature names, so tempo, time signature and bar length agree with each other.
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
COMPOUND_BPM = 138.0  # beats in threes at least this fast are eighth notes of 6/8, slower ones a waltz
HARMONY_WEIGHT = 0.0  # weight of "the chord changes here" when placing the "1" (a downbeat counts 1)
PAIRING_WEIGHT = 1.0  # same, when pairing short bars into 6/8 bars (the downbeats cannot tell)
MAX_FILL = 16         # beats: longer holes in the beat track are left alone
FILL_TOL = 0.12       # a hole is filled if its beats would be within this share of the local beat length


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


def regular_beats(beats: np.ndarray, duration: float) -> np.ndarray:
    """The tracker's beats with their holes filled and continued to both ends of the song.

    The tracker goes quiet in breaks and often starts a few bars late; the song (and its chords)
    goes on, so the grid does too: a hole that is a whole number of beats long gets that many
    beats, and the grid runs on at the local tempo before the first and after the last beat."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        return beats
    gaps = np.diff(beats)
    out = [beats[0]]
    for i, gap in enumerate(gaps):
        local = float(np.median(gaps[max(0, i - 8):i + 9]))
        k = int(round(gap / local))
        if 2 <= k <= MAX_FILL and abs(gap / k - local) <= FILL_TOL * local:
            out += [beats[i] + gap * j / k for j in range(1, k)]
        out.append(beats[i + 1])
    first, last = float(np.median(gaps[:8])), float(np.median(gaps[-8:]))
    head = np.arange(out[0] - first, 0.5 * first, -first)[::-1]
    tail = np.arange(out[-1] + last, duration - 0.5 * last, last)
    return np.concatenate([head, out, tail])


def _near(beats: np.ndarray, downbeats: np.ndarray) -> np.ndarray:
    """For every beat: did the tracker see a downbeat there?"""
    if not len(downbeats):
        return np.zeros(len(beats), dtype=bool)
    idx = np.searchsorted(downbeats, beats)
    lo = downbeats[np.clip(idx - 1, 0, len(downbeats) - 1)]
    hi = downbeats[np.clip(idx, 0, len(downbeats) - 1)]
    return np.minimum(np.abs(beats - lo), np.abs(beats - hi)) < DOWNBEAT_TOL


def _fit(near: np.ndarray, meter: int, bonus: np.ndarray | None = None, half_ok: bool = False) -> tuple[float, np.ndarray]:
    """Viterbi over bar positions. Score: +1 for a grid "1" on a detected downbeat, -1 for a grid "1"
    without one or a detected downbeat elsewhere. `bonus` adds to the score of a "1" on that beat;
    with `half_ok` a detected downbeat in the middle of the bar costs nothing."""
    n = len(near)
    emit = np.zeros((n, meter))
    emit[:, 0] = np.where(near, 1.0, -1.0)
    emit[:, 1:] = np.where(near, -1.0, 0.0)[:, None]
    if half_ok:
        emit[:, meter // 2] = 0.0
    if bonus is not None:
        emit[:, 0] += bonus
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


def _harmony_bonus(novelty: np.ndarray | None, weight: float) -> np.ndarray | None:
    """Score for putting a "1" on each beat, from how much the chord changes there. Centred, so
    that a grid in a random place gains nothing whatever its bar length."""
    if novelty is None or not weight:
        return None
    novelty = np.asarray(novelty, dtype=float)
    return weight * (novelty - novelty.mean())


def fit_grid(beats: np.ndarray, downbeats: np.ndarray, novelty: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """(meter, position of every beat in its bar). Meter is 4, 3 or 6.
    `novelty` (how much the chord changes on each beat) helps to place the "1"."""
    if len(beats) == 0:
        return 4, np.zeros(0, dtype=int)
    near = _near(beats, downbeats)
    bonus = _harmony_bonus(novelty, HARMONY_WEIGHT)
    fits = {m: _fit(near, m, bonus) for m in METERS}
    best = 4
    for m in METERS[1:]:
        score = fits[m][0]
        if score > 0 and score > fits[best][0] and score > fits[4][0] + max(2.0, METER_MARGIN * len(beats)):
            best = m
    return best, fits[best][1]


def bar_grid(beats: np.ndarray, downbeats: np.ndarray, novelty: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """fit_grid(), then the bar a guitarist counts.

    Beats in threes faster than COMPOUND_BPM are the eighth notes of 6/8: two of the tracker's
    short bars make one bar. Which two is decided by where the harmony moves (`novelty`: how much
    the chord changes on each beat), since the tracker hears a "1" on both. The other way round,
    six slow beats to the bar are two bars of 3/4."""
    meter, pos = fit_grid(beats, downbeats, novelty)
    bpm = tempo(beats)
    if meter == 3 and bpm >= COMPOUND_BPM:
        bonus = _harmony_bonus(novelty, max(HARMONY_WEIGHT, PAIRING_WEIGHT))
        return 6, _fit(_near(beats, downbeats), 6, bonus=bonus, half_ok=True)[1]
    if meter == 6 and bpm < COMPOUND_BPM:
        return 3, pos % 3
    return meter, pos


def notated_tempo(beats: np.ndarray, meter: int) -> float:
    """Tempo (bpm) of the beat the time signature names: the dotted quarter in 6/8, else the quarter."""
    bpm = tempo(beats)
    return bpm / 3 if meter == 6 else bpm


def time_signature(meter: int) -> str:
    return {3: "3/4", 6: "6/8"}.get(meter, "4/4")


def tempo(beats: np.ndarray) -> float:
    """Beats per minute. Beat times come on a 20 ms raster, so one beat length is only known to a
    few percent: the typical length is the mean of all those close to the median."""
    if len(beats) < 2:
        return 0.0
    gaps = np.diff(beats)
    median = float(np.median(gaps))
    typical = gaps[np.abs(gaps - median) <= 0.1 * median]
    return float(60.0 / typical.mean())
