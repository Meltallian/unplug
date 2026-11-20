"""Stage 7-8: build the draft chart (ChordPro) from detected chords and lyrics.

Chords are written as the shapes to play with the suggested capo (Ultimate Guitar convention).
Sections: instrumental stretches become Intro / Interlude / Outro; the lyric block that repeats
most becomes Chorus; other lyric blocks are Verse 1, 2, ...
"""

import re
from collections import Counter

from app.music import chord_name, transpose
from app.pipeline.rhythm import time_signature

MIN_BEATS = 2            # chords shorter than this are merged into the previous one on the chart
MAX_LINE = 8.0           # a lyric line "owns" at most this many seconds after its start
INSTRUMENTAL_GAP = 6.0   # a gap this long without lyrics becomes its own section
CHORDS_PER_ROW = 4
BARS_PER_ROW = 4
REPEAT_BAR = "%"         # repeat the previous bar (only written after a single-chord bar)
NO_CHORD = "N.C."        # a silent bar inside a stretch of bars


def _clean_segments(segments: list[tuple]) -> list[tuple]:
    """Drop no-chord and very short chords (merging them into the previous chord)."""
    out: list[list] = []
    for chord, start, end, beats in segments:
        if chord == "N":
            continue
        if out and (chord == out[-1][0] or beats < MIN_BEATS):
            out[-1][2] = end
            continue
        out.append([chord, start, end, beats])
    return [tuple(s) for s in out]


def _snap_to_word(text: str, pos: int) -> int:
    pos = max(0, min(pos, len(text)))
    while pos > 0 and pos < len(text) and not text[pos - 1].isspace():
        pos -= 1
    return pos


def _place(text: str, chords: list[tuple[float, str]]) -> str:
    """Insert [Chord] marks into text at the given fractional positions (0..1)."""
    marks: list[tuple[int, str]] = []
    for frac, name in chords:
        pos = _snap_to_word(text, round(frac * len(text)))
        if marks and pos <= marks[-1][0]:
            nxt = text.find(" ", marks[-1][0] + 1)
            pos = nxt + 1 if nxt != -1 else len(text)
        marks.append((pos, name))
    for pos, name in reversed(marks):
        text = text[:pos] + f"[{name}]" + text[pos:]
    return text


def _chord_rows(names: list[str]) -> list[str]:
    rows = [names[i:i + CHORDS_PER_ROW] for i in range(0, len(names), CHORDS_PER_ROW)]
    return ["  ".join(f"[{n}]" for n in row) for row in rows]


def _bar_edges(downbeats: list[float], start: float, end: float) -> list[float]:
    """Bar lines covering [start, end): the detected downbeats, continued at the typical bar length
    where the grid stops short of the stretch (beat tracking often starts late or ends early)."""
    if len(downbeats) < 2:
        return []
    length = float(sorted(b - a for a, b in zip(downbeats, downbeats[1:]))[(len(downbeats) - 1) // 2])
    if length <= 0:
        return []
    inside = [d for d in downbeats if start - 0.25 * length <= d < end - 0.25 * length]
    if not inside:          # stretch falls entirely before or after the detected grid
        anchor = downbeats[0] if start < downbeats[0] else downbeats[-1]
        k = (start - anchor) / length
        inside = [anchor + round(k) * length] if anchor + round(k) * length < end else [anchor + (round(k) - 1) * length]
    edges = list(inside)
    while edges[0] - length >= start - 0.25 * length:            # continue the grid backwards
        edges.insert(0, edges[0] - length)
    later = [d for d in downbeats if d > edges[-1]]
    while True:                                                  # ... and forwards, closing the last bar
        nxt = later.pop(0) if later else edges[-1] + length
        edges.append(nxt)
        if nxt >= end - 0.25 * length:
            return edges


def _bar_rows(downbeats: list[float], start: float, end: float, segs: list[tuple], shape) -> list[str] | None:
    """Instrumental stretch as bar lines: "| [Am] | % | [C] [G] | [F] |". None when there is no grid.
    "%" repeats the previous bar, so it is only used after a bar holding that single chord."""
    edges = _bar_edges(downbeats, start, end)
    if not edges:
        return None
    cells, prev = [], None
    for b0, b1 in zip(edges, edges[1:]):
        lo = max(b0, start)
        names = [shape(c) for c, s, e, _ in segs if s <= lo + 0.1 < e]
        names += [shape(c) for c, s, *_ in segs if lo + 0.1 < s < min(b1, end) - 0.1]
        names = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
        if not names:
            cells.append(None)
        elif len(names) == 1 and names == prev:
            cells.append(REPEAT_BAR)
        else:
            cells.append(" ".join(f"[{n}]" for n in names))
        prev = names
    while cells and cells[0] is None:       # silence before the first chord / after the last
        cells.pop(0)
    while cells and cells[-1] is None:
        cells.pop()
    if not cells:
        return []
    cells = [NO_CHORD if c is None else c for c in cells]
    return ["| " + " | ".join(cells[i:i + BARS_PER_ROW]) + " |" for i in range(0, len(cells), BARS_PER_ROW)]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def build_chordpro(*, title: str, artist: str, analysis: dict, segments: list[tuple],
                   lyrics: dict, duration: float) -> str:
    capo = analysis["capo"]
    segs = _clean_segments(segments)

    def shape(chord) -> str:
        return chord_name(transpose(chord, -capo))

    header = [f"{{title: {title}}}", f"{{artist: {artist}}}" if artist else None,
              f"{{key: {analysis['key']['name']}}}", f"{{capo: {capo}}}",
              f"{{tempo: {round(analysis['tempo'])}}}"]
    out = [h for h in header if h]

    offset = lyrics.get("offset", 0.0)
    timed = [(ln["t"] + offset, ln["text"]) for ln in lyrics.get("lines", []) if ln.get("t") is not None]
    if not timed and lyrics.get("lines"):
        # plain lyrics: spread the lines evenly over the part of the song that has chords
        texts = [ln["text"] for ln in lyrics["lines"]]
        start = segs[0][1] if segs else 0.0
        end = segs[-1][2] if segs else duration
        step = (end - start) / max(1, len(texts))
        timed = [(start + i * step, t) for i, t in enumerate(texts)]

    downbeats = analysis.get("downbeats") or []
    if analysis.get("meter"):
        out.append(f"{{time: {time_signature(analysis['meter'])}}}")

    if not timed:
        rows = _bar_rows(downbeats, 0.0, duration, segs, shape)
        out += ["", "{c: Chords}"] + (rows if rows is not None else _chord_rows([shape(c) for c, *_ in segs]))
        return "\n".join(out) + "\n"

    # 1. Split the timeline into sections: instrumental gaps and lyric blocks.
    sung = [(t, text) for t, text in timed if text]
    blocks: list[dict] = []
    prev_end = 0.0
    for i, (t, text) in enumerate(timed):
        if not text:  # an empty LRC line marks a break between blocks
            if blocks and blocks[-1]["kind"] == "lyrics":
                blocks[-1]["closed"] = True
            continue
        nxt = next((t2 for t2, x in timed[i + 1:] if x), duration)
        end = min(nxt, t + MAX_LINE)
        if t - prev_end >= INSTRUMENTAL_GAP or (not blocks and t > 2.0):
            blocks.append({"kind": "inst", "start": prev_end, "end": t})
        if not blocks or blocks[-1]["kind"] != "lyrics" or blocks[-1].get("closed"):
            blocks.append({"kind": "lyrics", "lines": []})
        blocks[-1]["lines"].append((t, end, text))
        prev_end = end
    if duration - prev_end >= 2.0:
        blocks.append({"kind": "inst", "start": prev_end, "end": duration})

    # 2. Name the sections.
    lyric_blocks = [b for b in blocks if b["kind"] == "lyrics"]
    keys = Counter(_norm(" ".join(x for *_, x in b["lines"])) for b in lyric_blocks)
    chorus_key = None
    if keys:
        candidate, count = max(keys.items(), key=lambda kv: (kv[1], len(kv[0])))
        if count >= 2:
            chorus_key = candidate
    verse = 0
    inst_seen = 0
    n_inst = sum(b["kind"] == "inst" for b in blocks)
    for b in blocks:
        if b["kind"] == "inst":
            inst_seen += 1
            b["name"] = "Intro" if b is blocks[0] else ("Outro" if b is blocks[-1] else "Interlude")
            if n_inst == 1 and b is not blocks[0] and b is not blocks[-1]:
                b["name"] = "Interlude"
        elif _norm(" ".join(x for *_, x in b["lines"])) == chorus_key:
            b["name"] = "Chorus"
        else:
            verse += 1
            b["name"] = f"Verse {verse}"

    # 3. Write chords over lyrics.
    def active_at(t: float):
        for chord, s, e, _ in segs:
            if s <= t < e:
                return chord
        return None

    for b in blocks:
        if b["kind"] == "inst":
            rows = _bar_rows(downbeats, b["start"], b["end"], segs, shape)
            if rows is not None:
                if rows:
                    out += ["", f"{{c: {b['name']}}}"] + rows
                continue
            names = []
            first = active_at(b["start"])
            if first is not None:
                names.append(shape(first))
            names += [shape(c) for c, s, *_ in segs if b["start"] < s < b["end"]]
            names = [n for i, n in enumerate(names) if i == 0 or n != names[i - 1]]
            if not names:
                continue
            out += ["", f"{{c: {b['name']}}}"] + _chord_rows(names)
            continue
        out += ["", f"{{c: {b['name']}}}"]
        last_shown = None
        for t, end, text in b["lines"]:
            span = max(end - t, 0.5)
            inside = [((s - t) / span, shape(c)) for c, s, *_ in segs if t <= s < end]
            # Show the chord already ringing at the line start only when it isn't simply
            # carried over from the previous line (Ultimate Guitar style).
            if not inside or inside[0][0] > 0.15:
                current = active_at(t)
                if current is not None and shape(current) != last_shown:
                    inside.insert(0, (0.0, shape(current)))
            if inside:
                last_shown = inside[-1][1]
            out.append(_place(text, inside))
    return "\n".join(out) + "\n"


# --- editing one chord in place ---------------------------------------------------------------

CHORD_SYMBOL = re.compile(r"^[A-G][#b]?[A-Za-z0-9#b+()]{0,8}(/[A-G][#b]?)?$")


def replace_chord(text: str, index: int, new: str | None, expected: str | None = None) -> str:
    """Replace (or with new=None remove) the index-th inline chord, counting exactly like the chart
    page numbers them: directive lines are skipped and empty brackets "[ ]" are not chords.
    With `expected`, refuse (ValueError) if that chord isn't the one currently at `index`."""
    out, count = [], 0
    for line in text.splitlines(keepends=True):
        if _DIRECTIVE.match(line.strip()):
            out.append(line)
            continue

        def swap(m: re.Match) -> str:
            nonlocal count
            if not m.group(1).strip():
                return m.group(0)
            hit = count == index
            count += 1
            if not hit:
                return m.group(0)
            if expected is not None and m.group(1).strip() != expected:
                raise ValueError("chart changed")
            return "" if new is None else f"[{new}]"
        out.append(_INLINE.sub(swap, line))
    if index >= count:
        raise IndexError(index)
    return "".join(out)


# --- reading ChordPro back (for the chart view) --------------------------------------------------

_DIRECTIVE = re.compile(r"^\{\s*([a-zA-Z_]+)\s*(?::\s*(.*?))?\s*\}$")
_INLINE = re.compile(r"\[([^\]]+)\]")


_BAR_MARKS = (REPEAT_BAR, NO_CHORD)


def line_kind(parts: list[tuple]) -> str:
    """CSS class for a parsed chart line: "bars" (| [Am] | % |), "chords-only", or "" (lyrics)."""
    text = "".join(t for _, t in parts)
    if "|" in text:
        rest = text.replace("|", " ")
        for mark in _BAR_MARKS:
            rest = rest.replace(mark, " ")
        if not rest.strip():
            return "bars"
    return "chords-only" if not text.strip() else ""


def bar_cells(parts: list[tuple]) -> list[list[tuple[str, str]]]:
    """A "bars" line split into bars; each bar is a list of ("chord", name) or ("mark", "%"/"N.C.").
    A bar emptied in the editor stays as an empty bar, so the columns don't shift."""
    cells: list[list[tuple[str, str]]] = [[]]
    for chord, text in parts:
        if chord:
            cells[-1].append(("chord", chord))
        for i, piece in enumerate(text.split("|")):
            if i:
                cells.append([])
            if piece.strip() in _BAR_MARKS:
                cells[-1].append(("mark", piece.strip()))
    # the pieces before the first "|" and after the last one are not bars
    lead = "".join(t for _, t in parts).lstrip()
    if lead.startswith("|") and not cells[0]:
        cells.pop(0)
    if not cells[-1]:
        cells.pop()
    return cells


def parse_chordpro(text: str) -> dict:
    """{"meta": {...}, "sections": [{"name", "lines": [[(chord|None, text), ...], ...]}]}"""
    meta: dict[str, str] = {}
    sections: list[dict] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _DIRECTIVE.match(line.strip()):
            name, value = m.group(1).lower(), (m.group(2) or "")
            if name in ("c", "comment", "section", "start_of_verse", "start_of_chorus", "sov", "soc"):
                default = {"start_of_chorus": "Chorus", "soc": "Chorus", "start_of_verse": "Verse", "sov": "Verse"}
                sections.append({"name": value or default.get(name, ""), "lines": []})
            else:
                meta[name] = value
            continue
        if not line.strip():
            continue
        if not sections:
            sections.append({"name": "", "lines": []})
        parts: list[tuple[str | None, str]] = []
        pos, pending = 0, None
        for m in _INLINE.finditer(line):
            before = line[pos:m.start()]
            if before or pending is not None:
                parts.append((pending, before))
            pending, pos = m.group(1).strip(), m.end()
        tail = line[pos:]
        if tail or pending is not None:
            parts.append((pending, tail))
        sections[-1]["lines"].append(parts)
    return {"meta": meta, "sections": sections}
