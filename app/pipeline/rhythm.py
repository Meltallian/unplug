"""Beats, downbeats, the bar grid and the tempo.

beat_this (CPJKU, 2024) gives beat and downbeat times. Three things are settled here on top:

1. One pulse for the whole song (regular_beats). The tracker goes quiet in breaks, starts late,
   and sometimes counts one section twice as slowly as the rest; the grid is filled so that every
   beat is the same note value from the first second to the last.
2. Bars (bar_grid). The raw downbeats are noisy (1-, 2- and 3-beat "bars" appear), so a regular
   grid is laid over the beats: every beat gets a position in its bar (0 = the "1"), bars may be
   cut short only at a cost, and the song may go from counting in threes to counting in fours
   (or back) at a bar line, also at a cost: a verse in 4/4 and a chorus in 6/8 each get their bars.
3. What a guitarist counts. Fast beats in threes are the eighth notes of 6/8 (two slow beats per
   bar), not a waltz at 150 bpm; in such a song the parts in fours are 4/4 counted in the same
   eighth notes. The tempo is always that of the beat the time signature names, so tempo, time
   signature and bar length agree with each other.
"""

import os
from functools import lru_cache

import numpy as np

CHECKPOINT = os.environ.get("BEAT_THIS_CHECKPOINT", "final0")
DOWNBEAT_TOL = 0.07   # s: a beat counts as a detected downbeat if one is this close
SHORT_BAR_COST = 3.0  # cost of cutting a bar short (a missed beat, an odd bar)
SWITCH_COST = 4.0     # cost of going from bars in threes to bars in fours or back (several bars of evidence)
REGROUP_COST = 1.0    # cost of going from bars of 3 to bars of 6 or back (the same count, marked differently)
# Cost per bar of each length, so that without clear downbeats a song is in four: a grid of
# threes or sixes must earn its place.
BAR_COST = {4: 0.0, 3: 0.15, 6: 0.3}
DENSITY_WINDOW = 48   # beats over which the share of detected downbeats is measured
MIN_GAP = 0.08        # s: beats closer than this are duplicates
MIN_DOWNBEATS = 4     # with fewer detected downbeats than this the song is simply counted in four
MIN_STRETCH = 24      # beats: a shorter stretch counted in threes among fours (or the reverse) is absorbed
COMPOUND_BPM = 138.0  # beats in threes at least this fast are eighth notes of 6/8, slower ones a waltz
HARMONY_WEIGHT = 1.0  # weight of "the chord changes here" when placing the "1" (a downbeat counts 1)
PAIRING_WEIGHT = 1.0  # same, when pairing short bars into 6/8 or 4/4 bars (the downbeats often cannot tell)
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


def _typical(gaps: np.ndarray) -> float:
    """The pulse of the song: the time between two beats that most of the song (by duration, not
    by number of beats) is counted in."""
    order = np.argsort(gaps)
    covered = np.cumsum(gaps[order])
    return float(gaps[order][np.searchsorted(covered, covered[-1] / 2)])


def regular_beats(beats: np.ndarray, duration: float) -> np.ndarray:
    """The tracker's beats on one pulse, with their holes filled, continued to both ends of the song.

    The tracker sometimes counts a section twice or three times as fast as the rest (the eighth
    notes or triplets instead of the beat), or half as fast, and goes quiet in breaks. Here the
    pulse is the one most of the song is counted in: extra beats in between are dropped, a hole
    that is a whole number of pulses long gets that many beats, and the grid runs on at the local
    tempo before the first and after the last beat."""
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        return beats
    pulse = _typical(np.diff(beats))

    kept, recent = [beats[0]], []                       # 1. drop beats between two pulses
    for i in range(1, len(beats)):
        local = float(np.median(recent[-8:])) if recent else pulse
        gap = beats[i] - kept[-1]
        ahead = beats[i + 1:i + 4] - kept[-1]
        if gap < 0.75 * local and np.any(np.abs(ahead - local) <= 0.15 * local):
            continue                                    # a beat one pulse on exists: this one is in between
        kept.append(beats[i])
        if abs(gap - pulse) <= 0.25 * pulse:
            recent.append(gap)
    beats = np.array(kept)

    gaps = np.diff(beats)                               # 2. fill the holes
    whole = np.maximum(np.round(gaps / pulse), 1)
    local_all = np.where(np.abs(gaps / whole - pulse) <= 0.15 * pulse, gaps / whole, np.nan)
    out = [beats[0]]
    for i, gap in enumerate(gaps):
        near = local_all[max(0, i - 8):i + 9]
        local = float(np.nanmedian(near)) if np.any(~np.isnan(near)) else pulse
        k = int(round(gap / local))
        if 2 <= k <= MAX_FILL and abs(gap / k - local) <= FILL_TOL * local:
            out += [beats[i] + gap * j / k for j in range(1, k)]
        out.append(beats[i + 1])
    out = np.array(out)

    gaps = np.diff(out)                                 # 3. continue to both ends
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


def _harmony_bonus(novelty: np.ndarray | None, weight: float, n: int) -> np.ndarray:
    """Score for putting a "1" on each beat, from how much the chord changes there. Centred, so
    that a grid in a random place gains nothing whatever its bar length."""
    if novelty is None or not weight:
        return np.zeros(n)
    novelty = np.asarray(novelty, dtype=float)
    return weight * (novelty - novelty.mean())


GROUPS = (4, 3, 6)   # beats per bar the grid may count; 4 first: the default


def _fit(near: np.ndarray, bonus: np.ndarray, groups: tuple = GROUPS, quiet: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Viterbi over (bars of 4 | 3 | 6 beats) × position in the bar. Score: +1 for a "1" on a
    detected downbeat, -1 for a detected downbeat elsewhere, -miss for a "1" without one, plus
    `bonus` on a "1". A bar may be cut short, and the bar length may change at a bar line, each
    at a cost (small between 3 and 6, which are both counted in threes).

    `miss` follows how many downbeats the tracker sees around there (2d/(1-d) for a share d of
    beats): with that, downbeats scattered at random score the same for every bar length, so a
    bar length wins on the pattern of the downbeats, not on how many "1"s it has to offer.
    On `quiet` beats (no music playing) nothing is scored: what the tracker reports there is noise.
    Returns, for every beat, the length of its bar (4, 3 or 6) and its position in it."""
    n = len(near)
    offsets = np.concatenate([[0], np.cumsum(groups)])            # state index of each bar length's "1"
    k = int(offsets[-1])
    window = np.ones(min(n, DENSITY_WINDOW))
    density = np.convolve(near.astype(float), window, mode="same") / np.convolve(np.ones(n), window, mode="same")
    miss = np.clip(2 * density / (1 - np.minimum(density, 0.9)), 0.2, 1.0)
    emit = np.tile(np.where(near, -1.0, 0.0)[:, None], (1, k))
    for g, o in zip(groups, offsets):
        emit[:, o] = np.where(near, 1.0, -miss) + bonus - BAR_COST[g]
    if quiet is not None:
        emit[quiet] = 0.0          # no music, no evidence: the grid just runs on
    score = emit[0].copy()
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        new = np.empty(k)
        for g, o in zip(groups, offsets):
            new[o + 1:o + g] = score[o:o + g - 1]                # normal step inside the bar
            back[i, o + 1:o + g] = np.arange(o, o + g - 1)
            best, src = score[o + g - 1], o + g - 1               # ... and from the last beat to the "1"
            mid = o + 1 + int(np.argmax(score[o + 1:o + g - 1]))  # cut a bar short: a middle position → "1"
            if score[mid] - SHORT_BAR_COST > best:
                best, src = score[mid] - SHORT_BAR_COST, mid
            for g2, o2 in zip(groups, offsets):                   # another bar length from this bar line on
                cost = SWITCH_COST if (g == 4) != (g2 == 4) else REGROUP_COST
                if g2 != g and score[o2 + g2 - 1] - cost > best:
                    best, src = score[o2 + g2 - 1] - cost, o2 + g2 - 1
            new[o], back[i, o] = best, src
        score = new + emit[i]
    state = np.zeros(n, dtype=int)
    state[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        state[i - 1] = back[i, state[i]]
    which = np.searchsorted(offsets, state, side="right") - 1
    return np.array(groups)[which], state - offsets[which]


class Grid:
    """The bars of a song. For every beat: its position in its bar (0 = the "1"), the number of
    beats in that bar's time signature (3, 4, 6 or 8) and the time signature (3 = 3/4, 4 = 4/4,
    6 = 6/8). A bar of 8 beats is a 4/4 bar counted in eighth notes."""

    def __init__(self, beats: np.ndarray, pos: np.ndarray, size: np.ndarray, meter: np.ndarray):
        self.beats, self.pos, self.size, self.meter = beats, pos, size, meter

    @property
    def downbeats(self) -> np.ndarray:
        return self.beats[self.pos == 0]

    @property
    def unit(self) -> np.ndarray:
        """Tracked beats per counted beat, for every beat: 3 in 6/8 (the dotted quarter), 2 in a 4/4
        bar counted in eighth notes, else 1."""
        return np.where(self.meter == 6, 3, np.where(self.meter == 4, self.size // 4, 1))

    def sections(self) -> list[dict]:
        """Stretches of bars in one time signature: [{"start": s, "meter": code, "tempo": bpm}].
        A stretch starts at a bar line (the first one starts with the song)."""
        if not len(self.beats):
            return [{"start": 0.0, "meter": 4, "tempo": 0.0}]
        out, start = [], 0
        for i in range(1, len(self.beats) + 1):
            if i == len(self.beats) or (self.pos[i] == 0 and (self.meter[i], self.size[i]) != (self.meter[start], self.size[start])):
                beats, meter, size = self.beats[start:i + 1], int(self.meter[start]), int(self.size[start])
                bpm = tempo(beats if len(beats) > 2 else self.beats) / int(self.unit[start])
                if out and out[-1]["meter"] == meter and abs(out[-1]["tempo"] - bpm) < 0.02 * bpm:
                    pass                                                      # same signature and tempo: one stretch
                else:
                    out.append({"start": round(float(self.beats[start]), 3) if out else 0.0, "meter": meter, "tempo": round(bpm, 2)})
                start = i
        return out

    @property
    def main_meter(self) -> int:
        """The time signature most of the song is in."""
        codes, counts = np.unique(self.meter, return_counts=True)
        return int(codes[np.argmax(counts)]) if len(codes) else 4


def _pair(evidence: np.ndarray) -> np.ndarray:
    """Two short bars make one long bar: which short bars start a long one? `evidence` says, for
    each short bar, how much speaks for a long bar starting there (downbeat seen, chord change).
    Normally they alternate; a long bar may be left with a single short bar, at the cost of a
    bar cut short, when the evidence turns round for good (after a break, an odd bar)."""
    m = len(evidence)
    start = np.full(m, -np.inf)     # best score with a long bar starting on short bar j
    second = np.full(m, -np.inf)    # ... with short bar j as the second half
    came_from_start = np.zeros(m, dtype=bool)
    start[0] = evidence[0]                            # a stretch begins with a bar line
    for j in range(1, m):
        normal, lonely = second[j - 1], start[j - 1] - SHORT_BAR_COST
        came_from_start[j] = lonely > normal
        start[j] = evidence[j] + max(normal, lonely)
        second[j] = start[j - 1]
    is_start = np.zeros(m, dtype=bool)
    j, at_start = m - 1, bool(start[m - 1] >= second[m - 1])
    while j >= 0:
        is_start[j] = at_start
        at_start = came_from_start[j] if at_start else True
        j -= 1
    return is_start


def bar_grid(beats: np.ndarray, downbeats: np.ndarray, novelty: np.ndarray | None = None,
             quiet: np.ndarray | None = None) -> Grid:
    """The bars a guitarist counts (see the module docstring). `novelty`: how much the chord
    changes on each beat; it helps where the downbeats cannot tell. `quiet`: beats on which no
    music plays (a video intro or outro); downbeats reported there are ignored."""
    n = len(beats)
    if n == 0:
        return Grid(beats, np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    near = _near(beats, downbeats)
    if quiet is not None:
        near &= ~quiet
    groups = GROUPS if near.sum() >= MIN_DOWNBEATS else (4,)      # no downbeats to speak of: in four
    bonus = _harmony_bonus(novelty, HARMONY_WEIGHT, n)
    group, pos = _fit(near, bonus, groups, quiet)
    # A few bars in another count (often in a fade-out, where the tracker wavers) are no section:
    # they are counted like the stretch before them (or after, at the very start).
    edges = [0] + [i for i in range(1, n) if pos[i] == 0 and (group[i] == 4) != (group[i - 1] == 4)] + [n]
    for j, (a, b) in enumerate(zip(edges, edges[1:])):
        if b - a < MIN_STRETCH and len(edges) > 2:
            like = group[a - 1] if j > 0 else group[b]
            group[a:b], pos[a:b] = _fit(near[a:b], bonus[a:b], (4,) if like == 4 else (3, 6),
                                        None if quiet is None else quiet[a:b])
    fast = tempo(beats) >= COMPOUND_BPM
    compound = fast and bool(np.any(group != 4))        # the pulse is the eighth note of 6/8 somewhere
    evidence = near.astype(float) + _harmony_bonus(novelty, PAIRING_WEIGHT, n)

    size, meter = group.copy(), np.where(group == 4, 4, 3)
    bar_line = np.concatenate([[True], pos[1:] == 0])
    edges = [0] + [i for i in range(1, n) if bar_line[i] and group[i] != group[i - 1]] + [n]
    for a, b in zip(edges, edges[1:]):                   # each stretch of one bar length
        g = int(group[b - 1])
        if g == 6 and not fast:                          # six slow beats: two bars of a waltz
            pos[a:b] %= 3
            size[a:b] = 3
        elif g == 6:                                     # six fast beats: a bar of 6/8
            meter[a:b] = 6
        elif (g == 3 and fast) or (g == 4 and compound):  # two short bars make one of 6/8 or of 4/4 in eighths
            starts = a + np.flatnonzero(pos[a:b] == 0)
            if len(starts):
                long_start = _pair(evidence[starts])
                for j, s0 in enumerate(starts):
                    if not long_start[j]:
                        pos[s0:(starts[j + 1] if j + 1 < len(starts) else b)] += g   # second half of a long bar
            size[a:b], meter[a:b] = 2 * g, (6 if g == 3 else 4)
    return Grid(beats, pos, size, meter)


def fit_grid(beats: np.ndarray, downbeats: np.ndarray, novelty: np.ndarray | None = None) -> tuple[int, np.ndarray]:
    """(time signature most of the song is in: 3, 4 or 6; position of every beat in its bar)."""
    grid = bar_grid(np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float), novelty)
    return (grid.main_meter if len(beats) else 4), grid.pos


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
