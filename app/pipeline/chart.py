"""Stage 7-8: build the draft chart (ChordPro) from detected chords and lyrics, and read it back.

Chords are written as the shapes to play with the suggested capo (Ultimate Guitar convention).
Sections: instrumental stretches become Intro / Interlude / Outro; the lyric block that repeats
most becomes Chorus; other lyric blocks are Verse 1, 2, ...

Every music line is a row of bars between "|" (see SCORE_SPEC.md, section 2):

    {c: Verse 1}
    | [Em7]Today is gonna be the | [G]day that they're gonna | [Dsus4]throw it back to | [A7sus4]you |
    By now you | [Em7]should've somehow | [G]realised what you gotta | [Dsus4]do | |
    | [C] . . [D] | [Em] | % |

Words written before the first "|" are a pickup (sung at the end of the bar before), "." is a beat on
which the chord keeps ringing, "%" repeats the bar before, "N.C." is silence. Charts written before
bars existed (lines without "|") still read: they simply carry no bar information.
"""

import bisect
import re
from collections import Counter
from difflib import SequenceMatcher

from app.music import chord_name, transpose
from app.pipeline.rhythm import time_signature

MIN_BEATS = 2            # chords shorter than this are merged into the previous one on the chart
MAX_LINE = 8.0           # a lyric line "owns" at most this many seconds after its start
INSTRUMENTAL_GAP = 6.0   # a gap this long without lyrics becomes its own section
CHORDS_PER_ROW = 4
BARS_PER_ROW = 4
REPEAT_BAR = "%"         # repeat the previous bar (only written after a single-chord bar)
NO_CHORD = "N.C."        # a silent bar inside a stretch of bars
DOT = "."                # one beat on which the previous chord keeps ringing
PICKUP_FROM = 0.5        # a sung line that starts later than this far into a bar is a pickup to the next bar

_TIME = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})\s*$")


def meter_of(time: str | None) -> tuple[int, int, int]:
    """(count, unit, beats in a bar) of a time signature: "4/4" -> (4, 4, 4), "6/8" -> (6, 8, 2).
    The beat of 6/8, 9/8 and 12/8 is the dotted quarter. Anything unreadable is 4/4."""
    m = _TIME.match(time or "")
    count, unit = (int(m.group(1)), int(m.group(2))) if m else (4, 4)
    if not (1 <= count <= 12 and unit in (2, 4, 8, 16)):
        count, unit = 4, 4
    compound = unit == 8 and count >= 6 and count % 3 == 0
    return count, unit, count // 3 if compound else count


def _default_beats(n: int, beats: int) -> list[int]:
    """Where n chords written without dots fall in a bar: one on the "1", two on the "1" and the
    middle (beat 3 of 4, beat 3 of 3, the second dotted beat of 6/8), more spread evenly."""
    if n == 2:
        return [0, (beats + 1) // 2]
    return [i * beats // n for i in range(n)]


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


def _bar_edges(downbeats: list[float], start: float, end: float, changes: tuple[float, ...] = ()) -> list[float]:
    """Bar lines covering [start, end): the detected downbeats, continued at the typical bar length
    where the grid stops short of the stretch (beat tracking often starts late or ends early).
    `changes` are the moments the meter changes: bars are longer or shorter from there on, so
    "typical" is judged within each stretch of one meter."""
    if len(downbeats) < 2:
        return []

    def median(lengths: list[float]) -> float:
        return float(sorted(lengths)[(len(lengths) - 1) // 2])

    length = median([b - a for a, b in zip(downbeats, downbeats[1:])])
    if length <= 0:
        return []
    stretches = [float("-inf"), *changes, float("inf")]
    typical = [median([b - a for a, b in zip(downbeats, downbeats[1:]) if lo <= a + 0.05 < hi] or [length])
               for lo, hi in zip(stretches, stretches[1:])]

    def local(t: float) -> float:
        """The typical bar length around t."""
        return typical[sum(c <= t + 0.05 for c in changes)]

    inside = [d for d in downbeats if start - 0.25 * length <= d < end - 0.25 * length]
    if not inside:          # stretch falls entirely before or after the detected grid
        anchor = downbeats[0] if start < downbeats[0] else downbeats[-1]
        k = (start - anchor) / length
        inside = [anchor + round(k) * length] if anchor + round(k) * length < end else [anchor + (round(k) - 1) * length]
    edges = list(inside)
    while edges[0] > start + 0.1:                                # continue the grid backwards
        edges.insert(0, edges[0] - local(edges[0] - 0.1))        # (a pickup bar is clipped at start)
    later = [d for d in downbeats if d > edges[-1]]
    while edges[-1] < end - 0.25 * local(edges[-1]):             # ... and forwards, closing the last bar
        step = local(edges[-1])
        edges.append(later.pop(0) if later and later[0] - edges[-1] <= 1.5 * step else edges[-1] + step)
        later = [d for d in later if d > edges[-1] + 0.5 * step]
    # where the beat tracker lost the beat, fill the gap with evenly spaced bars
    filled = [edges[0]]
    for a, b in zip(edges, edges[1:]):
        k = max(1, round((b - a) / local(a)))
        filled += [a + (b - a) * i / k for i in range(1, k + 1)]
    return filled


class _Meters:
    """The meter in force at any moment of the song (it may change between sections: a chorus in
    6/8, verses in 4/4), and the {time}/{tempo} lines to write in the chart where it changes."""

    def __init__(self, analysis: dict):
        first = {"start": 0.0, "meter": analysis.get("meter"), "tempo": analysis.get("tempo")}
        self.changes = []           # (start, time signature or None, beats in a bar, tempo or None)
        for i, m in enumerate(analysis.get("meters") or [first]):
            time = time_signature(m["meter"]) if m.get("meter") else None
            tempo = str(round(m["tempo"])) if m.get("tempo") else None
            self.changes.append((float(m.get("start") or 0.0) if i else 0.0, time, meter_of(time)[2], tempo))
        # what the header of the chart says
        self.written = (time_signature(analysis["meter"]) if analysis.get("meter") else None,
                        str(round(analysis["tempo"])) if analysis.get("tempo") else None)

    @property
    def starts(self) -> tuple[float, ...]:
        """The moments the meter changes."""
        return tuple(c[0] for c in self.changes[1:])

    def _at(self, t: float) -> tuple:
        return next((c for c in reversed(self.changes) if c[0] <= t + 0.05), self.changes[0])

    def beats(self, t: float) -> int:
        """Beats in the bar that starts at t."""
        return self._at(t)[2]

    def changes_at(self, t: float) -> bool:
        """True when the bar that starts at t is the first one of a new meter or tempo."""
        return any(abs(start - t) <= 0.05 for start in self.starts)

    def lines(self, t: float) -> list[str]:
        """The directive lines to write before a bar that starts at t (none while nothing changes)."""
        _, time, _, tempo = self._at(t)
        out = []
        if time and time != self.written[0]:
            out.append(f"{{time: {time}}}")
        if tempo and (tempo != self.written[1] or out):
            out.append(f"{{tempo: {tempo}}}")
        self.written = (time or self.written[0], tempo or self.written[1])
        return out


def _bar_chords(edges: list[float], segs: list[tuple], shape, meters: _Meters,
                start: float | None = None, end: float | None = None) -> list[list[tuple]]:
    """The chords heard in each bar as (beat, name, time): the one ringing at the bar line (beat 0),
    then those that start inside the bar, on the nearest beat. `start` and `end` clip the stretch."""
    out = []
    for b0, b1 in zip(edges, edges[1:]):
        beats = meters.beats(b0)
        lo = b0 if start is None else max(b0, start)
        hi = b1 if end is None else min(b1, end)
        found = [(0, shape(c), lo) for c, s, e, _ in segs if s <= lo + 0.1 < e]
        for c, s, *_ in segs:
            if lo + 0.1 < s < hi - 0.1:
                beat = min(beats - 1, max(1, round((s - b0) / (b1 - b0) * beats))) if found else 0
                found.append((beat, shape(c), s))
        out.append([f for i, f in enumerate(found) if i == 0 or f[1] != found[i - 1][1]])
    return out


def _marks(chords: list[tuple], beats: int) -> list[str]:
    """One bar's chords as tokens: "[C] [G]", or with dots ("[C] . . [G]") when a chord is not on
    the beat where a reader would put it by default."""
    at = [beat for beat, *_ in chords]
    names = [f"[{name}]" for _, name, *_ in chords]
    if at == _default_beats(len(at), beats) or at[0] != 0 or any(b <= a for a, b in zip(at, at[1:])):
        return names
    row = [DOT] * beats
    for beat, name in zip(at, names):
        row[beat] = name
    return row


def _cells(edges: list[float], bars: list[list[tuple]], meters: _Meters) -> list[tuple[float, str | None]]:
    """Bars without lyrics as (bar start, cell): "[Am]", "%", "[C] [G]"; None for a silent bar.
    "%" repeats the previous bar, so it is only used after a bar holding that single chord, and
    never across a change of meter (the chord is written again there)."""
    cells, prev = [], None
    for t, chords in zip(edges, bars):
        names = [name for _, name, _ in chords]
        if meters.changes_at(t):
            prev = None
        if not names:
            cells.append((t, None))
        elif len(names) == 1 and names == prev:
            cells.append((t, REPEAT_BAR))
        else:
            cells.append((t, " ".join(_marks(chords, meters.beats(t)))))
        prev = names
    return cells


def _rows(cells: list[tuple[float, str | None]], meters: _Meters) -> list[str]:
    """Rows of four bars, with a {time}/{tempo} line where the meter changes. Silent bars before
    the first chord and after the last one are dropped."""
    cells = list(cells)
    while cells and cells[0][1] is None:
        cells.pop(0)
    while cells and cells[-1][1] is None:
        cells.pop()
    rows: list[str] = []
    row: list[str] = []
    for t, cell in cells:
        change = meters.lines(t)
        if row and (change or len(row) == BARS_PER_ROW):
            rows.append("| " + " | ".join(row) + " |")
            row = []
        rows += change
        row.append(NO_CHORD if cell is None else cell)
    if row:
        rows.append("| " + " | ".join(row) + " |")
    return rows


def _bar_rows(downbeats: list[float], start: float, end: float, segs: list[tuple], shape,
              meters: _Meters) -> list[str] | None:
    """Instrumental stretch as bar lines: "| [Am] | % | [C] [G] | [F] |". None when there is no grid."""
    edges = _bar_edges(downbeats, start, end, meters.starts)
    if not edges:
        return None
    return _rows(_cells(edges, _bar_chords(edges, segs, shape, meters, start, end), meters), meters)


def _lyric_word(word: str) -> str:
    """A lyric word that cannot be mistaken for a bar line or a mark."""
    word = word.replace("|", "/")
    if word == DOT:
        return ""
    return f"({word})" if word == REPEAT_BAR or word.upper() == NO_CHORD else word


def _sung_cell(marks: list[tuple[str, float | None]], words: list[tuple[float, str]]) -> str:
    """One bar of a sung row: every word right after the chord it is sung on.
    `marks` are (token, time): chords carry the time they start, dots carry None."""
    starts = [t for _, t in marks if t is not None]
    sung: dict[int, list[str]] = {}
    for t, word in words:
        sung.setdefault(max(0, sum(s <= t + 0.05 for s in starts) - 1), []).append(word)
    if not starts:
        return " ".join([tok for tok, _ in marks] + sung.get(0, []))
    out, k = [], -1
    for tok, t in marks:
        if t is None:
            out.append(tok)
            continue
        k += 1
        glue = "" if tok.startswith("[") else " "          # "[C]word", but "N.C. word"
        out.append((tok + glue + " ".join(sung.get(k, []))).strip())
    return " ".join(out)


def _sung_sections(blocks: list[dict], edges: list[float], segs: list[tuple], shape, meters: _Meters) -> list[str]:
    """Every section as rows of bars, on one bar grid for the whole song. A sung line is one row:
    its words go in the bar where they start (the time each word was heard if the lyrics stage
    could tell, else linear interpolation over the time the line owns), and the bars up to the
    next sung line stay in the row."""
    count = len(edges) - 1
    bars = _bar_chords(edges, segs, shape, meters)

    def bar_at(t: float) -> int:
        return min(count - 1, max(0, bisect.bisect_right(edges, t) - 1))

    def line_near(t: float) -> int:
        i = bar_at(t)
        return i + 1 if t - edges[i] > edges[i + 1] - t else i

    # 1. Where every sung line starts its row. A line that starts late in a bar, or in a bar the
    #    line before is still singing in, is a pickup: its first words are written before the row.
    plans: list[dict] = []
    last_sung = -1
    has_intro = blocks[0]["kind"] == "inst"
    for b in blocks:
        b["plans"] = []
        for t, end, text, *rest in b.get("lines", ()):
            span = max(end - t, 0.5)
            words, floor = [], max(0, last_sung)
            found = list(re.finditer(r"\S+", text))
            heard = rest[0] if rest and rest[0] and len(rest[0]) == len(found) else None
            for k, m in enumerate(found):
                if word := _lyric_word(m.group()):
                    at = heard[k] if heard else t + span * m.start() / len(text)
                    floor = max(floor, bar_at(at))
                    words.append((floor, at, word))
            if not words:
                continue
            first, last = words[0][0], words[-1][0]
            late = (t - edges[first]) / (edges[first + 1] - edges[first]) > PICKUP_FROM and last > first
            before = bool(plans) or (has_intro and any(bars[:first + 1]))   # a pickup needs a bar before it
            row = first + 1 if first <= last_sung or (late and before) else first
            plan = {"row": row, "last": last, "pickup": [w for bar, _, w in words if bar < row],
                    "words": [w for w in words if w[0] >= row]}
            plans.append(plan)
            b["plans"].append(plan)
            last_sung = last
    for plan, nxt in zip(plans, plans[1:] + [{"row": count}]):
        plan["stop"] = nxt["row"]

    # 2. The bars of every instrumental section: from where the line before stops singing (and
    #    not before the time it owns is over) to the first bar of the next sung row.
    cursor, sung_before = 0, None
    for i, b in enumerate(blocks):
        if b["kind"] == "inst":
            stop = next((x["plans"][0]["row"] for x in blocks[i + 1:] if x.get("plans")), count)
            first = cursor
            if sung_before is not None:
                first = min(stop, max(cursor, sung_before["last"] + 1, sung_before["row"], line_near(b["start"])))
                sung_before["stop"] = first
            b["bars"] = (first, stop)
            cursor = stop
        elif b["plans"]:
            sung_before = b["plans"][-1]

    # 3. Write the sections. A chord is written where it changes and at the start of a section.
    out: list[str] = []
    ringing = None
    for b in blocks:
        if b["kind"] == "inst":
            first, stop = b["bars"]
            if rows := _rows(_cells(edges[first:stop], bars[first:stop], meters), meters):
                out += ["", f"{{c: {b['name']}}}"] + rows
                ringing = next((chords[-1][1] for chords in reversed(bars[first:stop]) if chords), None)
            continue
        if not b["plans"]:
            continue
        out += ["", f"{{c: {b['name']}}}"]
        fresh = True
        for plan in b["plans"]:
            row = " ".join(plan["pickup"] + ["|"])
            for j in range(plan["row"], plan["stop"]):
                if change := meters.lines(edges[j]):      # the meter changes here: close the row first,
                    out += [row] * (row != "|") + change  # and say again which chord is ringing
                    row, fresh = "|", True
                chords = bars[j]
                if not chords:
                    marks = [(NO_CHORD, float("-inf"))] if ringing is not None else []
                elif len(chords) == 1 and chords[0][1] == ringing and not fresh:
                    marks = []
                else:
                    times = iter(t for _, _, t in chords)
                    marks = [(tok, next(times) if tok != DOT else None)
                             for tok in _marks(chords, meters.beats(edges[j]))]
                ringing, fresh = chords[-1][1] if chords else None, False
                cell = _sung_cell(marks, [(t, w) for bar, t, w in plan["words"] if bar == j])
                row += f" {cell} |" if cell else " |"
            out.append(row)
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


SAME_LINE = 0.8      # two sung lines this alike (0..1) are the same line ("Oh, please tell me" / "Please tell me")
MIN_REFRAIN = 2      # lines: a shorter repeated run is a refrain line inside a verse, not a section


def _alike(a: str, b: str) -> bool:
    return bool(a) and bool(b) and SequenceMatcher(None, a, b).ratio() >= SAME_LINE


def _split_by_repetition(blocks: list[dict]) -> list[dict]:
    """Cut lyric blocks where the song goes from lines sung once to lines that come back, or the
    other way: synced lyrics often arrive as one long block, and the part that comes back is the
    chorus. Each lyric block gets "repeats": True/False. Parts that follow each other without a
    line sung once in between (a chorus twice, a verse sung again after a chorus) are cut where
    the first line of a part comes round again."""
    texts = [_norm(line[2]) for b in blocks if b["kind"] == "lyrics" for line in b["lines"]]
    comes_back = [any(_alike(a, b) for j, b in enumerate(texts) if j != i) for i, a in enumerate(texts)]
    out: list[dict] = []
    seen = 0
    openers: list[str] = []                          # first lines of the parts that come back
    for block in blocks:
        if block["kind"] != "lyrics":
            out.append(block)
            continue
        n = len(block["lines"])
        flags, norm = comes_back[seen:seen + n], texts[seen:seen + n]
        seen += n
        for i in range(1, n - 1):                    # one odd line between two of the other kind follows them
            if flags[i - 1] == flags[i + 1] != flags[i]:
                flags[i] = flags[i - 1]
        runs: list[list[int]] = []
        for i in range(n):
            if runs and flags[runs[-1][-1]] == flags[i]:
                runs[-1].append(i)
            else:
                runs.append([i])
        merged: list[list[int]] = []                 # a run too short to be a section joins its neighbour
        for run in runs:
            if merged and (len(run) < MIN_REFRAIN or len(merged[-1]) < MIN_REFRAIN):
                merged[-1] += run
            else:
                merged.append(run)
        for run in merged:
            repeats = sum(flags[i] for i in run) * 2 > len(run)
            parts = [[]]
            if repeats:
                openers.append(norm[run[0]])
            for i in run:
                if repeats and len(parts[-1]) >= MIN_REFRAIN and any(_alike(norm[i], o) for o in openers):
                    parts.append([])                 # a first line again: that part starts over
                parts[-1].append(i)
            for part in parts:
                out.append({"kind": "lyrics", "lines": [block["lines"][i] for i in part], "repeats": repeats})
    return out


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
    # when each word was heard, where the lyrics stage could tell (one time per word of the line)
    heard = {(ln["t"] + offset, ln["text"]): [w + offset for w in ln["w"]]
             for ln in lyrics.get("lines", []) if ln.get("t") is not None and ln.get("w")}
    synced = bool(timed)
    if not timed and lyrics.get("lines"):
        # plain lyrics: spread the lines evenly over the part of the song that has chords
        texts = [ln["text"] for ln in lyrics["lines"]]
        start = segs[0][1] if segs else 0.0
        end = segs[-1][2] if segs else duration
        step = (end - start) / max(1, len(texts))
        timed = [(start + i * step, t) for i, t in enumerate(texts)]

    downbeats = analysis.get("downbeats") or []
    meters = _Meters(analysis)
    if analysis.get("meter"):
        out.append(f"{{time: {time_signature(analysis['meter'])}}}")

    if not timed:
        rows = _bar_rows(downbeats, 0.0, duration, segs, shape, meters)
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
        blocks[-1]["lines"].append((t, end, text, heard.get((t, text))))
        prev_end = end
    if duration - prev_end >= 2.0:
        blocks.append({"kind": "inst", "start": prev_end, "end": duration})

    # 2. Name the sections.
    blocks = _split_by_repetition(blocks)
    lyric_blocks = [b for b in blocks if b["kind"] == "lyrics"]
    keys = Counter(_norm(" ".join(line[2] for line in b["lines"])) for b in lyric_blocks)
    chorus_key = None
    if keys:
        candidate, count = max(keys.items(), key=lambda kv: (kv[1], len(kv[0])))
        if count >= 2:
            chorus_key = candidate
    # Among the parts whose lines come back, the one that comes back most often is the chorus
    # (a first verse sung again at the end is still a verse). Two parts are the same part if
    # half the lines of the shorter one are in the other ("Because maybe ..." / "I said maybe ...").
    def same_part(x: dict, y: dict) -> bool:
        short, long = sorted((x, y), key=lambda b: len(b["lines"]))
        theirs = [_norm(line[2]) for line in long["lines"]]
        shared = sum(any(_alike(_norm(line[2]), t) for t in theirs) for line in short["lines"])
        return shared * 2 >= len(short["lines"])

    families: list[list[dict]] = []
    for b in lyric_blocks:
        if b.get("repeats"):
            family = next((f for f in families if any(same_part(b, other) for other in f)), None)
            if family is None:
                families.append(family := [])
            family.append(b)
    families = [f for f in families if len(f) >= 2]
    most = max(map(len, families), default=0)
    refrain = [id(b) for b in next((f for f in reversed(families) if len(f) == most), [])]   # a tie: the later part
    verse = 0
    inst_seen = 0
    n_inst = sum(b["kind"] == "inst" for b in blocks)
    for b in blocks:
        if b["kind"] == "inst":
            inst_seen += 1
            b["name"] = "Intro" if b is blocks[0] else ("Outro" if b is blocks[-1] else "Interlude")
            if n_inst == 1 and b is not blocks[0] and b is not blocks[-1]:
                b["name"] = "Interlude"
        elif id(b) in refrain or (not refrain and _norm(" ".join(line[2] for line in b["lines"])) == chorus_key):
            b["name"] = "Chorus"
        else:
            verse += 1
            b["name"] = f"Verse {verse}"

    # 3. With a bar grid and timed lyrics, every section is rows of bars.
    edges = _bar_edges(downbeats, 0.0, duration, meters.starts) if synced else []
    if len(edges) > 1:
        return "\n".join(out + _sung_sections(blocks, edges, segs, shape, meters)) + "\n"

    # 4. Without one (no beat grid, or lyrics without timing): chords over the lyric lines.
    def active_at(t: float):
        for chord, s, e, _ in segs:
            if s <= t < e:
                return chord
        return None

    for b in blocks:
        if b["kind"] == "inst":
            rows = _bar_rows(downbeats, b["start"], b["end"], segs, shape, meters)
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
        for t, end, text, *_ in b["lines"]:
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


def _mark(word: str) -> str | None:
    """The standalone marks of a bar: ".", "%" and "N.C."."""
    if word in (DOT, REPEAT_BAR):
        return word
    return NO_CHORD if word.upper() == NO_CHORD else None


def bar_row(parts: list[tuple]) -> tuple[list[tuple], list[list[tuple]]]:
    """A parsed line split at its bar lines: (pickup, bars). The pickup (what is written before the
    first "|") and every bar are lists of (kind, value, text): kind is "chord" (value is its name),
    "mark" (value is ".", "%" or "N.C.") or "" (words with nothing before them in the bar); text is
    what follows, as written. A bar emptied in the editor stays as an empty bar."""
    cells: list[list[list]] = [[]]
    for chord, text in parts:
        if chord:
            cells[-1].append(["chord", chord, ""])
        for i, piece in enumerate(text.split("|")):
            if i:
                cells.append([])
            for tok in re.findall(r"\S+|\s+", piece):
                if mark := _mark(tok):
                    cells[-1].append(["mark", mark, ""])
                elif cells[-1]:
                    cells[-1][-1][2] += tok
                elif not tok.isspace():
                    cells[-1].append(["", None, tok])
    if len(cells) > 1 and not cells[-1]:     # nothing after the last "|"
        cells.pop()
    return [tuple(c) for c in cells[0]], [[tuple(c) for c in cell] for cell in cells[1:]]


def line_kind(parts: list[tuple]) -> str:
    """CSS class for a parsed chart line: "bars" (| [Am] | % |), "lyric-bars" (a row of bars with
    words: | [Am]Is it so | [C]perfect |), "chords-only", or "" (words without bar lines)."""
    text = "".join(t for _, t in parts)
    if "|" in text:
        pickup, bars = bar_row(parts)
        return "lyric-bars" if any(t.strip() for cell in [pickup, *bars] for *_, t in cell) else "bars"
    return "chords-only" if not text.strip() else ""


def bar_cells(parts: list[tuple]) -> list[list[tuple[str, str]]]:
    """A "bars" line split into bars; each bar is a list of ("chord", name) or ("mark", "%"/"N.C."/".").
    A bar emptied in the editor stays as an empty bar, so the columns don't shift."""
    pickup, bars = bar_row(parts)
    return [[(kind, value) for kind, value, _ in cell] for cell in ([pickup] if pickup else []) + bars]


def tempo_label(time: str | None, tempo: str | None, meter: str | None = None) -> str:
    """A meter change as a short note for the lyrics view: "6/8 · ♩. = 50", "♩ = 90", "3/4".
    `meter` is the time signature in force when `time` itself does not change."""
    count, unit, beats = meter_of(time or meter)
    note = "♩." if beats != count else {4: "♩", 8: "♪"}.get(unit)
    speed = (f"{note} = {tempo}" if note else f"{tempo} bpm") if tempo else ""
    return " · ".join(x for x in (time, speed) if x)


def parse_chordpro(text: str) -> dict:
    """{"meta": {...}, "sections": [{"name", "lines": [[(chord|None, text), ...], ...], "changes": {...}}]}

    {time} and {tempo} lines in the header set how the song starts and go to meta. The same lines
    further down change the meter from that place on: they are kept in the section's "changes",
    keyed by the index of the line they stand before, as {"time", "tempo", "label"} (time and tempo
    are None when that line was not written; label is the note to show, "6/8 · ♩. = 50")."""
    meta: dict[str, str] = {}
    sections: list[dict] = []
    music = False                   # a music line has been read: the header is over
    now: dict[str, str | None] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if m := _DIRECTIVE.match(line.strip()):
            name, value = m.group(1).lower(), (m.group(2) or "")
            if name in ("c", "comment", "section", "start_of_verse", "start_of_chorus", "sov", "soc"):
                default = {"start_of_chorus": "Chorus", "soc": "Chorus", "start_of_verse": "Verse", "sov": "Verse"}
                sections.append({"name": value or default.get(name, ""), "lines": [], "changes": {}})
            elif name in ("time", "tempo") and sections and (music or name in meta):
                now[name] = value
                at = sections[-1]["changes"].setdefault(len(sections[-1]["lines"]), {"time": None, "tempo": None})
                at[name] = value
                at["label"] = tempo_label(at["time"], at["tempo"], now.get("time"))
            else:
                meta[name] = value
                now[name] = value
            continue
        if not line.strip():
            continue
        if not sections:
            sections.append({"name": "", "lines": [], "changes": {}})
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
        music = True
    return {"meta": meta, "sections": sections}


# --- reading a chart as a score (for the engraved view) ------------------------------------------

def _spread(words: list, lo: int, hi: int) -> list[tuple[int, object]]:
    """Words spread evenly over the beats lo..hi-1, in order."""
    return [(lo + i * (hi - lo) // len(words), w) for i, w in enumerate(words)]


def _words(pieces: list[tuple]) -> list[list[tuple[str, bool]]]:
    """The words written after each piece of a bar as (word, glued): glued means it continues the
    word before it without a space (only a chord mark stands between them)."""
    out, open_word = [], False
    for kind, _, text, *_ in pieces:
        cut = open_word and kind == "chord"
        out.append([(m.group(), cut and m.start() == 0) for m in re.finditer(r"\S+", text)])
        if text or kind:
            open_word = bool(text) and not text[-1].isspace()
    return out


def _join(groups: list[list[str]], k: int, found: list[tuple[str, bool]], last: list[str] | None) -> list[str] | None:
    """Add words to group k. A word cut by a chord mark ("Yester[C]day") is made whole again when both
    halves are sung on the same chord, hyphenated otherwise. Returns the group of the last word."""
    for word, glued in found:
        if glued and last:
            if last is groups[k]:
                last[-1] += word
                continue
            last[-1] += "-"
        groups[k].append(word)
        last = groups[k]
    return last


def _score_bar(raw: dict, before: dict | None) -> dict:
    """One written bar as beats: chords by rule 4 of SCORE_SPEC section 2, words by rules 5 and 6.
    `raw` holds the bar's pieces as (kind, value, text, index), the pickup written before the next
    row (sung on the last beats of this bar) and a lead-in (a pickup that had no bar before it)."""
    count, unit, beats = raw["time"]
    bar = {"beats": [{"chords": [], "words": []} for _ in range(beats)], "guess": raw["guess"],
           "time": (count, unit), "tempo": raw["tempo"]}
    lead: list[list[str]] = [[]]        # a pickup with no bar before it: on the "1" of this one
    last = None
    for piece, found in zip(raw["lead"], _words(raw["lead"])):
        if piece[0] == "chord":
            bar["beats"][0]["chords"].append((piece[1], piece[3]))
        last = _join(lead, 0, found, last)
    bar["beats"][0]["words"] += lead[0]
    pieces = raw["pieces"]
    is_chord = [kind == "chord" or value == NO_CHORD for kind, value, *_ in pieces]
    items = [i for i, (kind, value, *_) in enumerate(pieces) if is_chord[i] or value == DOT]
    chords = [i for i in items if is_chord[i]]
    at = [items.index(i) for i in chords] if len(items) == beats else _default_beats(len(chords), beats)
    for i, beat in zip(chords, at):
        bar["beats"][beat]["chords"].append((pieces[i][1], pieces[i][3]))
    if not chords and before and any(value == REPEAT_BAR for _, value, *_ in pieces):
        for beat, old in enumerate(before["beats"][:beats]):
            bar["beats"][beat]["chords"] = [(name, None) for name, _ in old["chords"]]

    # the pickup of the next row takes the last beats of the bar (at most its second half)
    pickup: list[list[str]] = [[]]
    starts = []                         # the first word of each pickup piece
    last = None
    for found in _words(raw["pickup"]):
        starts.append(len(pickup[0]) - bool(found and found[0][1]))
        last = _join(pickup, 0, found, last)
    room = max(1, beats // 2)
    if len(pickup[0]) < room:
        on = [beats - len(pickup[0]) + i for i in range(len(pickup[0]))]
    else:
        on = [beat for beat, _ in _spread(pickup[0], beats - room, beats)]
    free = min(on, default=beats)       # the bar's own words stay before the pickup when they can

    # the words after a chord are sung while it rings. Words before the first chord share its beats,
    # unless dots put that chord later in the bar: then they are sung on the beats before it.
    sung: list[list[str]] = [[] for _ in chords] or [[]]
    early = [] if chords and at[0] > 0 else sung[0]
    group, last = early, None
    for i, found in enumerate(_words(pieces)):
        if i in chords:
            group = sung[chords.index(i)]
        last = _join([group], 0, found, last)
    spans = [(early, 0, at[0])] if early is not sung[0] else []
    spans += [(found, at[k] if chords else 0, at[k + 1] if k + 1 < len(at) else beats) for k, found in enumerate(sung)]
    for found, lo, hi in spans:
        if found:
            for beat, word in _spread(found, lo, max(lo + 1, min(hi, free) if free > lo else hi)):
                bar["beats"][beat]["words"].append(word)
    for beat, word in zip(on, pickup[0]):
        bar["beats"][beat]["words"].append(word)
    for piece, start in zip(raw["pickup"], starts):    # a chord written in a pickup sits on the word after it
        if piece[0] == "chord":
            bar["beats"][on[start] if 0 <= start < len(on) else beats - 1]["chords"].append((piece[1], piece[3]))
    return bar


def read_score(text: str) -> dict:
    """Chart text as a score: {"meta", "time": (count, unit), "beats", "tempo", "sections", "unbarred"}.

    A section is {"name", "bars"}; a bar is {"beats": [...], "time": (count, unit), "tempo", "guess"}
    with one entry per beat of the meter in force; a beat is {"chords": [(name, index)], "words": [str]}.
    `index` numbers the chord marks exactly as replace_chord() does (None for a chord that is not a
    mark of its own: "N.C.", or one repeated by "%"). The top-level time, beats and tempo are those
    of the first bar. Lines without bar lines carry no bar information: every chord is laid out as
    one bar, flagged "guess"; "unbarred" counts those bars."""
    parsed = parse_chordpro(text)
    time, tempo = meter_of(parsed["meta"].get("time")), parsed["meta"].get("tempo")
    raws: list[dict] = []
    index = 0
    waiting: list[tuple] = []       # a pickup with no bar before it
    for number, section in enumerate(parsed["sections"]):
        for i, parts in enumerate(section["lines"]):
            if change := section["changes"].get(i):
                time = meter_of(change["time"]) if change["time"] else time
                tempo = change["tempo"] or tempo
            barred = "|" in "".join(t for _, t in parts)
            if barred:
                pickup, cells = bar_row(parts)
            else:   # no bar lines: one bar per chord, the words before the first chord in a bar of their own
                pickup, cells = [], [[("chord", chord, t)] if chord else [("", None, t)]
                                     for chord, t in parts if chord or t.strip()]
            numbered = []
            for cell in [pickup, *cells]:
                numbered.append([])
                for kind, value, t in cell:
                    numbered[-1].append((kind, value, t, index if kind == "chord" else None))
                    index += kind == "chord"
            gap = [("", None, " ", None)]     # two pickups in a row stay two words
            if raws:
                raws[-1]["pickup"] += gap * bool(raws[-1]["pickup"]) + numbered[0]
            else:
                waiting += gap * bool(waiting) + numbered[0]
            for cell in numbered[1:]:
                raws.append({"section": number, "pieces": cell, "pickup": [], "lead": waiting, "guess": not barred,
                             "time": time, "tempo": tempo})
                waiting = []
    if waiting:     # nothing but a pickup: give it a bar
        raws.append({"section": 0, "pieces": waiting, "pickup": [], "lead": [], "guess": False,
                     "time": time, "tempo": tempo})

    sections: list[dict] = []
    bar = None
    for raw in raws:
        bar = _score_bar(raw, bar)
        name = parsed["sections"][raw["section"]]["name"]
        if not sections or sections[-1]["id"] != raw["section"]:
            sections.append({"id": raw["section"], "name": name, "bars": []})
        sections[-1]["bars"].append(bar)
    for section in sections:
        del section["id"]
    first = raws[0] if raws else {"time": time, "tempo": tempo}
    return {"meta": parsed["meta"], "time": first["time"][:2], "beats": first["time"][2], "tempo": first["tempo"],
            "sections": sections, "unbarred": sum(raw["guess"] for raw in raws)}
