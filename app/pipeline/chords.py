"""Stages 3-4 (baseline): tuning, beats, chroma, key-aware template matching with HMM smoothing.

Writes cache/chords.json. Results are cached per VERSION, so changing the method
(and bumping VERSION) recomputes only this stage.
"""

import json
from pathlib import Path

import librosa
import numpy as np

from app.music import ALL_CHORDS, chord_name, choose_capo, diatonic, estimate_key, key_name, triad_pcs

VERSION = "baseline-2"
HOP = 512
SELF_TRANSITION = 0.85     # HMM: how strongly a chord tends to continue to the next beat
OUT_OF_KEY_PRIOR = 0.25    # relative prior weight for chords outside the detected key
SHARPNESS = 12.0           # softmax temperature on template similarity
SILENCE_RATIO = 0.08       # beats quieter than this × median RMS become N.C.


def _templates() -> np.ndarray:
    t = np.zeros((len(ALL_CHORDS), 12))
    for i, chord in enumerate(ALL_CHORDS):
        root, third, fifth = triad_pcs(chord)
        t[i, [root, third, fifth]] = [1.0, 0.8, 0.8]
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def analyze(wav: Path, out: Path) -> dict:
    if out.exists():
        cached = json.loads(out.read_text())
        if cached.get("version") == VERSION:
            return cached

    y, sr = librosa.load(wav, sr=None, mono=True)
    tuning = float(librosa.estimate_tuning(y=y, sr=sr))  # fraction of a semitone
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP, tuning=tuning, bins_per_octave=36)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    bounds = librosa.util.fix_frames(beats, x_min=0, x_max=chroma.shape[1])
    beat_chroma = librosa.util.sync(chroma, bounds, aggregate=np.median)          # 12 × n_beats
    beat_rms = librosa.util.sync(rms[np.newaxis, :], bounds, aggregate=np.mean)[0]
    times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP)

    # Key from the energy-weighted average chroma
    weights = beat_rms / (beat_rms.sum() or 1.0)
    key = estimate_key(beat_chroma @ weights)

    # Per-beat chord likelihoods: cosine similarity to triad templates, key prior, N for silence
    norm = beat_chroma / (np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9)
    sim = _templates() @ norm                                                   # 24 × n_beats
    prior = np.array([1.0 if c in diatonic(key) else OUT_OF_KEY_PRIOR for c in ALL_CHORDS])
    lik = np.exp(SHARPNESS * sim) * prior[:, None]
    silent = beat_rms < SILENCE_RATIO * np.median(beat_rms)
    n_row = np.where(silent, lik.max(axis=0) * 10, lik.min(axis=0) * 0.1)
    lik = np.vstack([lik, n_row])
    prob = lik / lik.sum(axis=0, keepdims=True)

    trans = librosa.sequence.transition_loop(len(ALL_CHORDS) + 1, SELF_TRANSITION)
    path = librosa.sequence.viterbi_discriminative(prob, trans)

    labels = [ALL_CHORDS[i] if i < len(ALL_CHORDS) else "N" for i in path]
    segments = []
    for i, label in enumerate(labels):
        if segments and segments[-1]["chord"] == label:
            segments[-1]["end"] = float(times[i + 1])
            segments[-1]["beats"] += 1
        else:
            segments.append({"chord": label, "start": float(times[i]), "end": float(times[i + 1]), "beats": 1})

    weighted = [(s["chord"], s["end"] - s["start"]) for s in segments if s["chord"] != "N"]
    capo = choose_capo(weighted)
    result = {
        "version": VERSION,
        "tuning_semitones": round(tuning, 3),
        "tempo": float(np.atleast_1d(tempo)[0]),
        "key": {"tonic": key[0], "mode": key[1], "name": key_name(key)},
        "capo": capo,
        "segments": [
            {**s, "name": chord_name(s["chord"]),
             "chord": None if s["chord"] == "N" else list(s["chord"])}
            for s in segments
        ],
    }
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    return result


def load_segments(result: dict) -> list[tuple]:
    """Segments as (chord_or_N, start, end, beats) with chords back as tuples."""
    return [("N" if s["chord"] is None else tuple(s["chord"]), s["start"], s["end"], s["beats"])
            for s in result["segments"]]
