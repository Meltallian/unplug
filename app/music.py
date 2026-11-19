"""Music theory helpers: chord names, keys, transposition and capo choice.

A chord is a (root, quality) pair: root is a pitch class 0-11 (C=0), quality is
"maj" or "min". Version 1 only knows major and minor triads; power chords and
extensions are reduced to these.
"""

import re

import numpy as np

NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

Chord = tuple[int, str]

_TOKEN = re.compile(r"^([A-G])([#b]?)([^/]*?)(?:/([A-G][#b]?))?\*?$")
_SUFFIX = re.compile(
    r"^(m|min|-|maj|M|dim|aug|\+|sus|add|5|6|7|9|11|13|b5|#5|b9|#9|#11|maj7|M7|m7|m6|m9|"
    r"maj9|mmaj7|7sus4|sus2|sus4|add9|add2|add11|dim7|m7b5|\(|\)|\d)*$"
)


def pc(name: str) -> int:
    base = _PC[name[0]]
    if len(name) > 1:
        base += {"#": 1, "b": -1}[name[1]]
    return base % 12


def parse_chord(token: str) -> Chord | None | str:
    """Parse a chord symbol. Returns (root, quality), "N" for no-chord, or None if not a chord."""
    t = token.strip()
    if t.upper() in ("N.C.", "NC", "N.C", "N"):
        return "N"
    m = _TOKEN.match(t)
    if not m:
        return None
    letter, acc, suffix, _bass = m.groups()
    if not _SUFFIX.match(suffix):
        return None
    root = pc(letter + acc)
    is_minor = (suffix.startswith("m") and not suffix.startswith("maj")) or suffix.startswith(("min", "-", "dim"))
    return (root, "min" if is_minor else "maj")


def chord_name(chord: Chord | str) -> str:
    if chord == "N":
        return "N.C."
    root, quality = chord
    return NAMES[root] + ("m" if quality == "min" else "")


def transpose(chord: Chord | str, semitones: int) -> Chord | str:
    if chord == "N":
        return chord
    return ((chord[0] + semitones) % 12, chord[1])


def triad_pcs(chord: Chord) -> tuple[int, int, int]:
    root, quality = chord
    third = 3 if quality == "min" else 4
    return (root, (root + third) % 12, (root + 7) % 12)


ALL_CHORDS: list[Chord] = [(r, q) for q in ("maj", "min") for r in range(12)]

# --- keys -------------------------------------------------------------------
# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

Key = tuple[int, str]  # (tonic pc, "major"|"minor")


def key_name(key: Key) -> str:
    return NAMES[key[0]] + ("m" if key[1] == "minor" else "")


def estimate_key(chroma_profile: np.ndarray) -> Key:
    """Best key for a 12-bin pitch-class profile."""
    best, best_r = (0, "major"), -2.0
    for tonic in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(chroma_profile, np.roll(prof, tonic))[0, 1]
            if r > best_r:
                best, best_r = (tonic, mode), r
    return best


def key_from_chords(chords: list[tuple[Chord, float]]) -> Key:
    """Key from (chord, weight) pairs, e.g. a reference sheet."""
    prof = np.zeros(12)
    for chord, w in chords:
        for i, p in enumerate(triad_pcs(chord)):
            prof[p] += w * (1.5 if i == 0 else 1.0)
    return estimate_key(prof)


def diatonic(key: Key) -> set[Chord]:
    """Chords that fit the key. In minor, the major V is allowed too."""
    t, mode = key
    if mode == "major":
        degrees = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min")]
    else:
        degrees = [(0, "min"), (3, "maj"), (5, "min"), (7, "min"), (8, "maj"), (10, "maj"), (7, "maj")]
    return {((t + d) % 12, q) for d, q in degrees}


def same_scale(a: Key, b: Key) -> bool:
    """True when the keys share their notes (identical or relative major/minor)."""
    def major_tonic(k: Key) -> int:
        return k[0] if k[1] == "major" else (k[0] + 3) % 12
    return major_tonic(a) == major_tonic(b)


# --- capo -------------------------------------------------------------------
# How hard each shape is for an acoustic player: 0 = easy open shape.
_SHAPE_COST = {
    (0, "maj"): 0, (2, "maj"): 0, (4, "maj"): 0, (7, "maj"): 0, (9, "maj"): 0,   # C D E G A
    (9, "min"): 0, (2, "min"): 0, (4, "min"): 0,                                 # Am Dm Em
    (5, "maj"): 0.35, (11, "min"): 0.6,                                          # F, Bm (common barres)
}


def shape_cost(chord: Chord) -> float:
    return _SHAPE_COST.get(chord, 1.0)


# A capo must save real effort: each fret costs as much as 6% of the song on a hard shape.
CAPO_PENALTY = 0.06


def choose_capo(chords: list[tuple[Chord, float]], max_capo: int = 7) -> int:
    """Capo position that puts the most playing time on easy open shapes (ties → lower capo)."""
    def cost(capo: int) -> float:
        total = sum(w for _, w in chords) or 1.0
        return sum(w * shape_cost(transpose(c, -capo)) for c, w in chords) / total + CAPO_PENALTY * capo
    return min(range(max_capo + 1), key=cost)
