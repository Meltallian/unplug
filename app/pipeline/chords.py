"""Stages 3-4: tuning, beats, chroma (+ bass chroma), key-aware template matching, HMM smoothing.

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

FEATURES_VERSION = "f2"
HOP = 512


@dataclass(frozen=True)
class Params:
    self_transition: float = 0.85   # HMM: how strongly a chord tends to continue to the next beat
    out_of_key_prior: float = 0.25  # relative prior weight for chords outside the detected key
    sharpness: float = 12.0         # softmax temperature on template similarity
    silence_ratio: float = 0.08     # beats quieter than this × median RMS become N.C.
    bass_weight: float = 0.5        # how much the bass note pulls towards chords with that root
    third_weight: float = 0.8       # template weight of the third (root = 1)
    fifth_weight: float = 0.8       # template weight of the fifth


PARAMS = Params()
VERSION = "bass-1"


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
                return {k: z[k] for k in z.files}
        except (OSError, ValueError):
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

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    bounds = librosa.util.fix_frames(beats, x_min=0, x_max=n)
    feats = {
        "chroma": librosa.util.sync(chroma, bounds, aggregate=np.median),
        "bass": librosa.util.sync(bass, bounds, aggregate=np.median),
        "rms": librosa.util.sync(rms[np.newaxis, :], bounds, aggregate=np.mean)[0],
        "times": librosa.frames_to_time(bounds, sr=sr, hop_length=HOP),
        "tuning": np.array(tuning),
        "tempo": np.array(float(np.atleast_1d(tempo)[0])),
    }
    tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **feats)
    tmp.replace(cache)
    return feats


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

    trans = librosa.sequence.transition_loop(len(ALL_CHORDS) + 1, p.self_transition)
    path = librosa.sequence.viterbi_discriminative(prob, trans)

    labels = [ALL_CHORDS[i] if i < len(ALL_CHORDS) else "N" for i in path]
    segments: list[dict] = []
    for i, label in enumerate(labels):
        if segments and segments[-1]["chord"] == label:
            segments[-1]["end"] = float(times[i + 1])
            segments[-1]["beats"] += 1
        else:
            segments.append({"chord": label, "start": float(times[i]), "end": float(times[i + 1]), "beats": 1})

    weighted = [(s["chord"], s["end"] - s["start"]) for s in segments if s["chord"] != "N"]
    return {
        "version": VERSION,
        "params": asdict(p),
        "tuning_semitones": round(float(feats["tuning"]), 3),
        "tempo": float(feats["tempo"]),
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
            if cached.get("version") == VERSION:
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
