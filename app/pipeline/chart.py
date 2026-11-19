"""Stage 7-8: build the draft chart (ChordPro) from detected chords and lyrics.

Chords are written as the shapes to play with the suggested capo (Ultimate Guitar convention).
Sections: instrumental stretches become Intro / Interlude / Outro; the lyric block that repeats
most becomes Chorus; other lyric blocks are Verse 1, 2, ...
"""

import re
from collections import Counter

from app.music import chord_name, transpose

MIN_BEATS = 2            # chords shorter than this are merged into the previous one on the chart
MAX_LINE = 8.0           # a lyric line "owns" at most this many seconds after its start
INSTRUMENTAL_GAP = 6.0   # a gap this long without lyrics becomes its own section
CHORDS_PER_ROW = 4


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

    if not timed:
        out += ["", "{c: Chords}"] + _chord_rows([shape(c) for c, *_ in segs])
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


# --- reading ChordPro back (for the chart view) --------------------------------------------------

_DIRECTIVE = re.compile(r"^\{\s*([a-zA-Z_]+)\s*(?::\s*(.*?))?\s*\}$")
_INLINE = re.compile(r"\[([^\]]+)\]")


def parse_chordpro(text: str) -> dict:
    """{"meta": {...}, "sections": [{"name", "lines": [[(chord|None, text), ...], ...]}]}"""
    meta: dict[str, str] = {}
    sections: list[dict] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _DIRECTIVE.match(line.strip()):
            name, value = m.group(1).lower(), (m.group(2) or "")
            if name in ("c", "comment", "section", "start_of_verse", "start_of_chorus", "sov", "soc"):
                sections.append({"name": value or name, "lines": []})
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
