"""Music theory helpers: chord names, keys, transposition and capo choice.

A chord is a (root, quality) pair: root is a pitch class 0-11 (C=0), quality is "maj", "min" or
a richer chord of one of these two families ("7", "min7", "sus4", ... see QUALITY_SUFFIX).
plain() reduces any chord to the major or minor chord a guitarist would strum instead.
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


# How each quality is written after the root, and the family it belongs to.
QUALITY_SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "sus4": "sus4", "sus2": "sus2",
                  "dim": "dim", "dim7": "dim7", "hdim7": "m7b5", "aug": "aug", "maj6": "6", "min6": "m6",
                  "minmaj7": "mMaj7"}
_MINOR_FAMILY = {"min", "min7", "min6", "minmaj7", "dim", "dim7", "hdim7"}


def plain(chord: Chord) -> Chord:
    """The major or minor chord of the same root."""
    return (chord[0], "min" if chord[1] in _MINOR_FAMILY else "maj")


def chord_name(chord: Chord | str) -> str:
    if chord == "N":
        return "N.C."
    root, quality = chord
    return NAMES[root] + QUALITY_SUFFIX.get(quality, "")


def transpose(chord: Chord | str, semitones: int) -> Chord | str:
    if chord == "N":
        return chord
    return ((chord[0] + semitones) % 12, chord[1])


def triad_pcs(chord: Chord) -> tuple[int, int, int]:
    root, quality = plain(chord)
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


# Richer chords that are easier (or harder) than their plain chord in open position.
_SHAPE_COST.update({(5, "maj7"): 0.0, (11, "7"): 0.2, (11, "min7"): 0.3, (6, "min7"): 0.6})


def shape_cost(chord: Chord) -> float:
    return _SHAPE_COST.get(chord, _SHAPE_COST.get(plain(chord), 1.0))


# A capo must save real effort: each fret costs as much as 6% of the song on a hard shape.
CAPO_PENALTY = 0.06


def choose_capo(chords: list[tuple[Chord, float]], max_capo: int = 7) -> int:
    """Capo position that puts the most playing time on easy open shapes (ties → lower capo)."""
    def cost(capo: int) -> float:
        total = sum(w for _, w in chords) or 1.0
        return sum(w * shape_cost(transpose(c, -capo)) for c, w in chords) / total + CAPO_PENALTY * capo
    return min(range(max_capo + 1), key=cost)


# --- chord names as written on charts (with any suffix) ---------------------------------------
_NAME = re.compile(r"^([A-G])([#b]?)(.*?)(?:/([A-G])([#b]?))?$")

# Same shapes as the chart page (static/chart.js), low E (left) to high e; x = not played.
SHAPES = {
    "C": "x32010", "C#": "x46664", "D": "xx0232", "Eb": "x68886", "E": "022100", "F": "133211",
    "F#": "244322", "G": "320003", "Ab": "466544", "A": "x02220", "Bb": "x13331", "B": "x24442",
    "Cm": "x35543", "C#m": "x46654", "Dm": "xx0231", "Ebm": "x68876", "Em": "022000", "Fm": "133111",
    "F#m": "244222", "Gm": "355333", "Abm": "466444", "Am": "x02210", "Bbm": "x13321", "Bm": "x24432",
    # richer chords with a common open (or easy) shape of their own
    "A7": "x02020", "B7": "x21202", "C7": "x32310", "D7": "xx0212", "E7": "020100", "G7": "320001",
    "F#7": "242322", "Am7": "x02010", "Bm7": "x20202", "Dm7": "xx0211", "Em7": "022030", "F#m7": "242222",
    "C#m7": "x46454", "Cmaj7": "x32000", "Dmaj7": "xx0222", "Emaj7": "021100", "Fmaj7": "xx3210",
    "Gmaj7": "320002", "Amaj7": "x02120", "Asus4": "x02230", "Dsus4": "xx0233", "Esus4": "022200",
    "Gsus4": "330013", "Csus4": "x33011", "Asus2": "x02200", "Dsus2": "xx0230", "Esus2": "024400",
    "Csus2": "x30013", "Gsus2": "300033", "Fsus2": "xx3011", "Bm7b5": "x2323x", "F#m7b5": "2x221x",
}


def move_chord_name(name: str, semitones: int) -> str:
    """Transpose a written chord (keeping its suffix and bass note); anything else is returned as is."""
    m = _NAME.match(name.strip())
    if not m:
        return name
    root = NAMES[(pc(m.group(1) + m.group(2)) + semitones) % 12]
    bass = f"/{NAMES[(pc(m.group(4) + m.group(5)) + semitones) % 12]}" if m.group(4) else ""
    return root + m.group(3) + bass


def shape_key(name: str) -> str | None:
    """The diagram to show for a written chord: its own if we have one (Em7, Dsus4, B7 ...),
    else the plain major or minor chord of its root (Bbm7 -> Bbm). None if it is no chord."""
    m = _NAME.match(name.strip())
    if not m:
        return None
    root = NAMES[pc(m.group(1) + m.group(2))]
    if root + m.group(3) in SHAPES:
        return root + m.group(3)
    minor = re.match(r"^(m(?!aj)|min|-)", m.group(3)) is not None
    return root + ("m" if minor else "")


def diagram_svg(key: str) -> str:
    """Chord diagram as inline SVG (same drawing as the chart page)."""
    frets = SHAPES.get(key)
    if not frets:
        return ""
    f = [-1 if c == "x" else int(c) for c in frets]
    played = [v for v in f if v > 0]
    base = min(played) if played and max(played) > 4 else 1

    def x(i: int) -> float:
        return 10 + i * 9

    def y(row: float) -> float:
        return 18 + row * 12

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 84" width="48" height="63">']
    for i in range(6):
        parts.append(f'<line x1="{x(i)}" y1="{y(0)}" x2="{x(i)}" y2="{y(5)}" stroke="#888" stroke-width="1"/>')
    for r in range(6):
        w = 3 if r == 0 and base == 1 else 1
        parts.append(f'<line x1="{x(0)}" y1="{y(r)}" x2="{x(5)}" y2="{y(r)}" stroke="{"#222" if w == 3 else "#888"}" stroke-width="{w}"/>')
    if base > 1:
        parts.append(f'<text x="0" y="{y(0.8)}" font-size="9" fill="#666">{base}</text>')
    for i, v in enumerate(f):
        if v < 0:
            parts.append(f'<text x="{x(i) - 3}" y="12" font-size="9" fill="#666">×</text>')
        elif v == 0:
            parts.append(f'<circle cx="{x(i)}" cy="10" r="2.6" fill="none" stroke="#222" stroke-width="1.2"/>')
        else:
            parts.append(f'<circle cx="{x(i)}" cy="{y(v - base + 0.5)}" r="3.6" fill="#222"/>')
    parts.append("</svg>")
    return "".join(parts)
