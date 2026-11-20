"""Stages 3-4: tuning, beats + bar grid, chroma (+ bass chroma), key-aware template matching,
HMM smoothing in which chords change on the "1" of a bar, less often mid-bar, rarely elsewhere.

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

from app.pipeline import rhythm
from app.music import ALL_CHORDS, chord_name, choose_capo, diatonic, estimate_key, key_name, triad_pcs

FEATURES_VERSION = "f4"
HOP = 512


@dataclass(frozen=True)
class Params:
    change_bar: float = 0.3         # HMM: chance the chord changes on the "1" of a bar
    change_half: float = 0.05       # ... in the middle of a 4/4 or 6/8 bar
    change_off: float = 0.001       # ... on any other beat
    out_of_key_prior: float = 0.25  # relative prior weight for chords outside the detected key
    sharpness: float = 12.0         # softmax temperature on template similarity
    silence_ratio: float = 0.08     # beats quieter than this × median RMS become N.C.
    bass_weight: float = 0.5        # how much the bass note pulls towards chords with that root
    third_weight: float = 0.8       # template weight of the third (root = 1)
    fifth_weight: float = 0.8       # template weight of the fifth


PARAMS = Params()
VERSION = "bars-2"


def _templates(p: Params) -> np.ndarray:
    t = np.zeros((len(ALL_CHORDS), 12))
    for i, chord in enumerate(ALL_CHORDS):
        root, third, fifth = triad_pcs(chord)
        t[i, [root, third, fifth]] = [1.0, p.third_weight, p.fifth_weight]
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def features(wav: Path) -> dict:
    """Beat-synchronous chroma, bass chroma and loudness. Cached next to the wav."""
    cache = wav.parent / f"features-{FEATURES_VERSION}.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                feats = {k: z[k] for k in z.files}
            if {"chroma", "bass", "rms", "times", "pos", "meter", "tuning", "tempo"} <= feats.keys():
                return feats
        except Exception:  # corrupt or partial cache (e.g. BadZipFile): recompute
            pass

    y, sr = librosa.load(wav, sr=None, mono=True)
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))  # fraction of a semitone
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning, bins_per_octave=36)
    # Bass: E1 (41 Hz) to E3 (165 Hz) — bass guitar and the low guitar strings
    bass_cqt = np.abs(librosa.cqt(y_harm, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("E1"),
                                  n_bins=72, bins_per_octave=36, tuning=tuning))
    bass = librosa.feature.chroma_cqt(C=bass_cqt, sr=sr, hop_length=HOP, bins_per_octave=36,
                                      fmin=librosa.note_to_hz("E1"))
    n = min(chroma.shape[1], bass.shape[1])
    chroma, bass = chroma[:, :n], bass[:, :n]
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][:n]

    beat_times, downbeat_times = rhythm.track(y, sr)
    meter, beat_pos = rhythm.fit_grid(beat_times, downbeat_times)
    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    keep = (frames > 0) & (frames < n) & np.concatenate([[True], np.diff(frames) > 0])
    frames, beat_pos = frames[keep], beat_pos[keep]
    bounds = np.concatenate([[0], frames, [n]])
    # the stretch before the first beat has no bar position (-1)
    pos = np.concatenate([[-1], beat_pos])
    feats = {
        "chroma": librosa.util.sync(chroma, bounds, aggregate=np.median),
        "bass": librosa.util.sync(bass, bounds, aggregate=np.median),
        "rms": librosa.util.sync(rms[np.newaxis, :], bounds, aggregate=np.mean)[0],
        "times": librosa.frames_to_time(bounds, sr=sr, hop_length=HOP),
        "pos": pos,
        "meter": np.array(meter),
        "tuning": np.array(tuning),
        "tempo": np.array(rhythm.tempo(beat_times)),
    }
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp, **feats)
        tmp.replace(cache)
    finally:
        tmp.unlink(missing_ok=True)
    return feats


def display_tempo(bpm: float) -> float:
    """Beat trackers often lock onto double time; strummed songs rarely exceed ~135 bpm per beat."""
    while 135 < bpm < 10_000:
        bpm /= 2
    return bpm


def _change_prob(pos: np.ndarray, meter: int, p: Params) -> np.ndarray:
    """Chance of a chord change on each beat, from its position in the bar."""
    c = np.full(len(pos), p.change_off)
    c[pos <= 0] = p.change_bar                       # the "1" (and the stretch before the first beat)
    if meter % 2 == 0:
        c[pos == meter // 2] = p.change_half
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


def decode(feats: dict, p: Params = PARAMS) -> dict:
    chroma, bass, rms, times = feats["chroma"], feats["bass"], feats["rms"], feats["times"]

    weights = rms / (rms.sum() or 1.0)
    key = estimate_key(chroma @ weights)

    norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    sim = _templates(p) @ norm                                                   # 24 × n_beats
    prior = np.array([1.0 if c in diatonic(key) else p.out_of_key_prior for c in ALL_CHORDS])
    lik = np.exp(p.sharpness * sim) * prior[:, None]
    if p.bass_weight:
        b = bass / (bass.max(axis=0, keepdims=True) + 1e-9)                     # 1 = strongest bass note
        roots = np.array([c[0] for c in ALL_CHORDS])
        lik *= (0.1 + b[roots, :]) ** p.bass_weight
    silent = rms < p.silence_ratio * np.median(rms)
    n_row = np.where(silent, lik.max(axis=0) * 10, lik.min(axis=0) * 0.1)
    lik = np.vstack([lik, n_row])
    prob = lik / lik.sum(axis=0, keepdims=True)

    pos = feats.get("pos", np.zeros(prob.shape[1], dtype=int))
    meter = int(feats.get("meter", 4))
    path = _viterbi(np.log(prob + 1e-12), _change_prob(pos, meter, p))
    labels = [ALL_CHORDS[i] if i < len(ALL_CHORDS) else "N" for i in path]
    segments: list[dict] = []
    for i, label in enumerate(labels):
        if segments and segments[-1]["chord"] == label:
            segments[-1]["end"] = float(times[i + 1])
            segments[-1]["beats"] += 1
        else:
            segments.append({"chord": label, "start": float(times[i]), "end": float(times[i + 1]), "beats": 1,
                             "on_bar": bool(pos[i] <= 0)})

    weighted = [(s["chord"], s["end"] - s["start"]) for s in segments if s["chord"] != "N"]
    return {
        "version": VERSION,
        "params": asdict(p),
        "tuning_semitones": round(float(feats["tuning"]), 3),
        "tempo": display_tempo(float(feats["tempo"])),
        "meter": meter,
        "downbeats": [round(float(t), 3) for t, q in zip(times, pos) if q == 0],
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
