"""Stages 3-4: beats and bars, chord probabilities, and the decoding that joins them.

The chord model (btc.py) says, frame by frame, how likely each of 170 chords is. The beat tracker
(rhythm.py) says where the beats and the bars are. decode() joins the two: one chord per beat,
chosen so that chords change on the "1" of a bar, less often mid-bar, rarely elsewhere. It first
decides root and major/minor, then says more (7, m7, sus4, ...) only where the model is sure.

Two parts so methods can be compared quickly:
  features()  slow, cached in cache/features-<FEATURES_VERSION>.npz
  decode()    fast, turns features into chords with a given set of parameters
analyze() runs both and caches the result in cache/chords.json (per VERSION).
"""

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import librosa
import numpy as np

from app.music import ALL_CHORDS, chord_name, choose_capo, diatonic, estimate_key, key_name, triad_pcs
from app.pipeline import btc, rhythm

FEATURES_VERSION = "f5"
RMS_HOP = 512


@dataclass(frozen=True)
class Params:
    change_bar: float = 0.3         # HMM: chance the chord changes on the "1" of a bar
    change_half: float = 0.05       # ... in the middle of a 4/4 or 6/8 bar
    change_off: float = 0.001       # ... on any other beat
    out_of_key_prior: float = 0.25  # relative prior weight for chords outside the detected key
    sharpness: float = 3.0          # exponent on the chord model's probabilities (per beat)
    silence_ratio: float = 0.08     # beats quieter than this × median RMS become N.C.
    quality_share: float = 0.5      # write G7, Am7, Dsus4 ... when the model gives that reading at
                                    # least this share among the chords of the same root and family


PARAMS = Params()
VERSION = "btc-1"

# The decoder first chooses among the 24 major and minor chords; every richer chord of the model
# counts towards the plain chord a guitarist would strum instead. A sus chord has no third, so it
# speaks for the major and the minor chord of its root alike.
_FAMILY = {"maj": ("maj", "maj6", "maj7", "7", "aug", "sus2", "sus4"),
           "min": ("min", "min6", "min7", "minmaj7", "dim", "dim7", "hdim7", "sus2", "sus4")}
# ... and these are written out when the model is sure of them (sixths stay plain chords)
_WRITTEN = ("7", "maj7", "min7", "sus4", "sus2", "dim", "dim7", "hdim7", "aug")


def _family_matrix() -> np.ndarray:
    """(24 × 170): which model chords count towards each major/minor chord."""
    m = np.zeros((len(ALL_CHORDS), btc.N_CHORDS))
    for i, (root, quality) in enumerate(ALL_CHORDS):
        for q in _FAMILY[quality]:
            m[i, btc.chord_index(root, q)] = 1.0
    return m


def plain_probabilities(probs: np.ndarray) -> np.ndarray:
    """(frames × 25): the model's probabilities summed into the 24 major and minor chords, plus "no chord"."""
    return np.hstack([probs @ _family_matrix().T, probs[:, btc.N_INDEX:btc.N_INDEX + 1]])


def features(wav: Path) -> dict:
    """Chord probabilities, beat activations and loudness of a song. Cached next to the wav."""
    cache = wav.parent / f"features-{FEATURES_VERSION}.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                feats = {k: z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k] for k in z.files}
            if {"probs", "rms", "beat_act", "down_act", "tuning", "duration"} <= feats.keys():
                return feats
        except Exception:  # corrupt or partial cache (e.g. BadZipFile): recompute
            pass

    y, sr = librosa.load(wav, sr=None, mono=True)
    beat_act, down_act = rhythm.cached_activations(wav, y, sr)
    feats = {
        "probs": btc.chord_probabilities(y, sr),
        "rms": librosa.feature.rms(y=y, hop_length=RMS_HOP)[0],
        "rms_step": np.array(RMS_HOP / sr),
        "beat_act": beat_act,
        "down_act": down_act,
        "tuning": np.array(float(librosa.estimate_tuning(y=y, sr=sr))),  # fraction of a semitone
        "duration": np.array(len(y) / sr),
    }
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **{k: v.astype(np.float16) if k in ("probs", "beat_act", "down_act") else v
                                    for k, v in feats.items()})
        tmp.replace(cache)
    finally:
        tmp.unlink(missing_ok=True)
    return feats


def span_means(frames: np.ndarray, step: float, bounds: np.ndarray) -> np.ndarray:
    """Mean of frame values (frames × ...) over each span between consecutive bounds (seconds);
    frame i covers [i, i+1) × step. Frames cut by a bound count for the part inside."""
    total = np.concatenate([np.zeros((1,) + frames.shape[1:]), np.cumsum(frames, axis=0, dtype=float)])
    x = np.clip(np.asarray(bounds, dtype=float) / step, 0, len(frames))
    i = np.minimum(x.astype(int), len(frames) - 1)
    frac = (x - i).reshape((-1,) + (1,) * (frames.ndim - 1))
    at = total[i] + frac * frames[i]
    width = np.maximum(np.diff(x), 1e-9).reshape((-1,) + (1,) * (frames.ndim - 1))
    return (at[1:] - at[:-1]) / width


def _change_prob(pos: np.ndarray, size, p: Params) -> np.ndarray:
    """Chance of a chord change on each beat, from its position in a bar of `size` beats
    (one number, or one per beat where the time signature changes)."""
    size = np.broadcast_to(np.asarray(size), pos.shape)
    c = np.full(len(pos), p.change_off)
    c[pos <= 0] = p.change_bar                       # the "1" (and the stretch before the first beat)
    c[(size % 2 == 0) & (pos == size // 2)] = p.change_half
    return c


def _viterbi(log_lik: np.ndarray, change: np.ndarray) -> np.ndarray:
    """Most likely state path when, at step t, the state changes with probability change[t]
    (spread evenly over the other states)."""
    k, n = log_lik.shape
    stay, move = np.log1p(-change), np.log(change / (k - 1))
    score = log_lik[:, 0].copy()
    back = np.zeros((n, k), dtype=int)
    for t in range(1, n):
        best = int(np.argmax(score))
        jump = score[best] + move[t]
        cont = score + stay[t]
        take_jump = jump > cont
        back[t] = np.where(take_jump, best, np.arange(k))
        score = np.where(take_jump, jump, cont) + log_lik[:, t]
    path = np.zeros(n, dtype=int)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _written_quality(chord: tuple, probs: np.ndarray, p: Params) -> str:
    """"maj"/"min", or a richer reading of the same chord when the model is sure of it."""
    root, quality = chord
    family = {q: float(probs[btc.chord_index(root, q)]) for q in _FAMILY[quality]}
    best = max(family, key=family.get)
    if best in _WRITTEN and family[best] >= p.quality_share * (sum(family.values()) + 1e-12):
        return best
    return quality


# A guitarist plays one shape for "the A minor of this song": where the model writes Am7 here and
# Am there, the song gets the one it mostly is (if clearly so), else the plain chord.
_SAME_SHAPE = (("min", "min7"), ("maj", "maj7"))
SAME_SHAPE_SHARE = 0.7


def one_spelling(segments: list[dict]) -> None:
    """Make each chord of a song one of (m, m7), and one of (plain, maj7), in place."""
    for pair in _SAME_SHAPE:
        seconds: dict[int, dict[str, float]] = {}
        for s in segments:
            if s["chord"] != "N" and s["chord"][1] in pair:
                by_quality = seconds.setdefault(s["chord"][0], dict.fromkeys(pair, 0.0))
                by_quality[s["chord"][1]] += s["end"] - s["start"]
        for s in segments:
            if s["chord"] != "N" and s["chord"][1] in pair:
                by_quality = seconds[s["chord"][0]]
                rich = by_quality[pair[1]] >= SAME_SHAPE_SHARE * sum(by_quality.values())
                s["chord"] = (s["chord"][0], pair[1] if rich else pair[0])


def merge_short(segments: list[dict], min_beats: np.ndarray) -> list[dict]:
    """Chords shorter than one counted beat (min_beats[i] tracked beats at beat i: three eighth
    notes in 6/8) join the chord before them; then equal neighbours are joined."""
    out: list[dict] = []
    for s in segments:
        s = dict(s)
        if out and s["chord"] != "N" and s["beats"] < min_beats[s["first"]] and out[-1]["chord"] != "N":
            s["chord"] = out[-1]["chord"]
        if out and out[-1]["chord"] == s["chord"]:
            out[-1]["end"], out[-1]["beats"] = s["end"], out[-1]["beats"] + s["beats"]
        else:
            out.append(s)
    return out


def decode_beats(probs: np.ndarray, rms: np.ndarray, times: np.ndarray, pos: np.ndarray, size,
                 p: Params = PARAMS, min_beats: np.ndarray | None = None) -> tuple[list[dict], tuple]:
    """Chords from per-beat model probabilities (beats × 170), loudness, the beat bounds (beats + 1
    times), each beat's position in its bar and the number of beats of that bar (one number, or
    one per beat). `min_beats`: per beat, how many tracked beats make one counted beat there.
    Returns (segments, key)."""
    family = probs @ _family_matrix().T                                        # beats × 24
    weights = rms / (rms.sum() or 1.0)
    profile = np.zeros(12)
    for i, chord in enumerate(ALL_CHORDS):
        for k, pc in enumerate(triad_pcs(chord)):
            profile[pc] += float(family[:, i] @ weights) * (1.5 if k == 0 else 1.0)
    key = estimate_key(profile)

    prior = np.array([1.0 if c in diatonic(key) else p.out_of_key_prior for c in ALL_CHORDS])
    lik = (family.T + 1e-6) ** p.sharpness * prior[:, None]                    # 24 × beats
    silent = rms < p.silence_ratio * np.median(rms)
    none = (probs[:, btc.N_INDEX] + 1e-6) ** p.sharpness
    lik = np.vstack([lik, np.where(silent, lik.max(axis=0) * 10, none)])
    path = _viterbi(np.log(lik / lik.sum(axis=0, keepdims=True)), _change_prob(pos, size, p))

    segments: list[dict] = []
    for i, state in enumerate(path):
        if segments and segments[-1]["state"] == state:
            segments[-1]["end"] = float(times[i + 1])
            segments[-1]["beats"] += 1
            segments[-1]["last"] = i
        else:
            segments.append({"state": int(state), "start": float(times[i]), "end": float(times[i + 1]),
                             "beats": 1, "on_bar": bool(pos[i] <= 0), "first": i, "last": i})
    for s in segments:
        state, first, last = s.pop("state"), s["first"], s.pop("last")
        if state == len(ALL_CHORDS):
            s["chord"] = "N"
        else:
            root, quality = ALL_CHORDS[state]
            span = probs[first:last + 1]
            s["chord"] = (root, _written_quality((root, quality), span.T @ weights[first:last + 1], p))
    one_spelling(segments)
    segments = merge_short(segments, np.ones(len(pos)) if min_beats is None else min_beats)
    for s in segments:
        del s["first"]
    return segments, key


def novelty(probs: np.ndarray, sharpness: float = PARAMS.sharpness) -> np.ndarray:
    """How much the harmony moves on each beat (0 = same chord as the beat before, 1 = another one)."""
    family = (probs @ _family_matrix().T + 1e-9) ** sharpness
    family /= family.sum(axis=1, keepdims=True)
    same = np.sum(np.sqrt(family[1:] * family[:-1]), axis=1)
    return np.concatenate([[0.0], 1.0 - same])


def decode(feats: dict, p: Params = PARAMS) -> dict:
    duration = float(feats["duration"])
    beats, downbeats = rhythm.pick(feats["beat_act"], feats["down_act"])
    beats = rhythm.regular_beats(beats[(beats > 0.05) & (beats < duration - 0.05)], duration)
    times = np.concatenate([[0.0], beats, [duration]])
    probs = span_means(feats["probs"], btc.FRAME_S, times)
    rms = span_means(feats["rms"], float(feats["rms_step"]), times)
    grid = rhythm.bar_grid(beats, downbeats, novelty(probs, p.sharpness)[1:])
    pos = np.concatenate([[-1], grid.pos])           # the stretch before the first beat has no bar position
    size = np.concatenate([grid.size[:1], grid.size]) if len(beats) else np.array([4])
    unit = np.concatenate([grid.unit[:1], grid.unit]) if len(beats) else np.array([1])
    segments, key = decode_beats(probs, rms, times, pos, size, p, unit)

    meters = grid.sections()
    weighted = [(s["chord"], s["end"] - s["start"]) for s in segments if s["chord"] != "N"]
    return {
        "version": VERSION,
        "params": asdict(p),
        "tuning_semitones": round(float(feats["tuning"]), 3),
        "tempo": meters[0]["tempo"],                  # of the beat the time signature names, at the start
        "meter": meters[0]["meter"],                  # 3 = 3/4, 4 = 4/4, 6 = 6/8, at the start
        "meters": meters,                             # ... and every change of time signature after that
        "downbeats": [round(float(t), 3) for t in grid.downbeats],
        "key": {"tonic": key[0], "mode": key[1], "name": key_name(key)},
        "capo": choose_capo(weighted),
        "segments": [
            {**s, "name": chord_name(s["chord"]), "chord": None if s["chord"] == "N" else list(s["chord"])}
            for s in segments
        ],
    }


def analyze(wav: Path, out: Path, params: Params | None = None) -> dict:
    """Chords for a song. With explicit params (experiments) the result is returned but not cached."""
    if params is None and out.exists():
        try:
            cached = json.loads(out.read_text())
            if cached.get("version") == VERSION and cached.get("params") == asdict(PARAMS):
                return cached
        except (json.JSONDecodeError, OSError):
            pass  # corrupt cache: recompute
    result = decode(features(wav), params or PARAMS)
    if params is None:
        out.parent.mkdir(exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out)
    return result


def with_overrides(overrides: dict[str, float]) -> Params:
    return replace(PARAMS, **overrides)


def load_segments(result: dict) -> list[tuple]:
    """Segments as (chord_or_N, start, end, beats) with chords back as tuples."""
    return [("N" if s["chord"] is None else tuple(s["chord"]), s["start"], s["end"], s["beats"])
            for s in result["segments"]]
