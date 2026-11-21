"""The engraved score: chart text -> score model (chart.read_score) -> MEI -> Verovio -> SVG.

One staff in the look of a classical edition: treble clef, key and time signature, title, artist
and tempo mark as the header, one rhythm slash per beat, chord symbols above the staff, lyric words
under the beat they start on, section names as rehearsal marks, a double bar at the end of every
section and a final bar line.

The systems are broken here, not by Verovio: its own line breaking underestimates bars that carry
many words and then squeezes them until the words overlap. So the score is laid out twice: once with
every bar alone on a line (which gives the room each bar really needs), then with the breaks chosen
from those widths.
"""

import base64
import hashlib
import re
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from app.music import move_chord_name
from app.pipeline.chart import NO_CHORD, meter_of, read_score

MEI_NS = "http://www.music-encoding.org/ns/mei"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

MIN_WIDTH, MAX_WIDTH = 280, 1600   # CSS pixels the score is laid out for
MAX_BARS = 1000                    # refuse to engrave more (a real song has a few hundred at most)
MARGIN = 20                        # page margin, in Verovio units (the web page has its own padding)
MARGIN_LEFT = 44                   # ... on the left, where the bar numbers of the systems stand
LEAD = 17                          # Verovio units between a bar line and the first slash after it
FILL = 1.0                         # how full a system may be packed (bar widths are measured generously)
LYRIC_SIZE = 4.0                   # words a little smaller than Verovio's default (4.5): more bars on a phone
SECTION_FILL = 0.6                 # a section that ends on a system this full takes the line break
MIN_WORD_SIZE = 50                 # percent: the smallest the words of an overfull bar are set
CACHE_SIZE = 32                    # engravings kept (one per chart text, width, capo and key) ...
CACHE_BYTES = 16_000_000           # ... as long as their SVG stays under this in total

# Major keys by pitch class of the tonic, spelled as app.music.NAMES does: sharps (+) or flats (-).
_SIGNATURE = {0: 0, 1: -5, 2: 2, 3: -3, 4: 4, 5: -1, 6: 6, 7: 1, 8: -4, 9: 3, 10: -2, 11: 5}
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_KEY = re.compile(r"^\s*([A-G])([#b]?)(.*)$")
_NOTE = {2: "\uECA3", 4: "\uECA5", 8: "\uECA7", 16: "\uECA9"}   # SMuFL metronome notes: half, quarter, eighth, 16th
_AUG_DOT = "\uECB7"                                             # ... and the dot of a dotted note
_UNPRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")       # not allowed in XML


def layout_units(width: int) -> int:
    """Width of the page in Verovio units for a screen `width` CSS pixels wide. A phone (400 px)
    gets about two sung bars per system with a staff 26 px high; a desktop gets four or five."""
    return round(380 + 1.8 * min(MAX_WIDTH, max(MIN_WIDTH, width)))


def key_signature(key: str) -> str:
    """MEI key signature ("3s", "2f", "0") of a key name as written on charts ("A", "F#m", "Bb")."""
    m = _KEY.match(key or "")
    if not m:
        return "0"
    tonic = (_PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    if re.match(r"m(?!aj)|min", m.group(3)):
        tonic = (tonic + 3) % 12        # a minor key is written with the signature of its relative major
    n = _SIGNATURE[tonic]
    return "0" if n == 0 else f"{abs(n)}{'s' if n > 0 else 'f'}"


def engraved_chord(name: str) -> str:
    """A chord symbol with real accidentals: "F#m7b5" -> "F♯m7♭5", "Bb/D" -> "B♭/D"."""
    m = re.match(r"^([A-G])([#b]?)(.*?)(?:/([A-G])([#b]?))?$", name)
    if not m:
        return name
    sign = {"#": "♯", "b": "♭", "": ""}
    suffix = re.sub(r"[#b](?=\d)", lambda a: sign[a.group()], m.group(3))
    bass = f"/{m.group(4)}{sign[m.group(5)]}" if m.group(4) else ""
    return m.group(1) + sign[m.group(2)] + suffix + bass


def chart_capo(meta: dict) -> int:
    """The capo a chart is written for (read like the chart page reads it)."""
    m = re.match(r"\s*(\d+)", meta.get("capo", "0"))
    return min(9, int(m.group(1))) if m else 0


def tempo_mark(tempo: str | None, time: tuple[int, int]) -> tuple[str, str] | None:
    """(note, number) of a tempo mark: a quarter note in 4/4, a dotted quarter in 6/8 and 12/8."""
    m = re.match(r"\s*(\d{2,3})", tempo or "")
    if not m:
        return None
    count, unit = time
    compound = meter_of(f"{count}/{unit}")[2] != count
    return _NOTE[4] + _AUG_DOT if compound else _NOTE.get(unit, _NOTE[4]), m.group(1)


def _el(parent, tag: str, attrs: dict | None = None, text: str | None = None):
    el = ET.SubElement(parent, tag, {k: _UNPRINTABLE.sub("", str(v)) for k, v in (attrs or {}).items()})
    if text is not None:
        el.text = _UNPRINTABLE.sub("", text)
    return el


def to_mei(score: dict, *, semis: int = 0, key: str = "", capo: int = 0,
           breaks: frozenset[int] | set[int] = frozenset(), every_bar_alone: bool = False,
           units: int = 0, small: dict[int, int] | None = None) -> str:
    """The score model as MEI. Chords are moved by `semis` semitones; `key` (the key of the written
    shapes) gives the key signature; a system break is written before every bar numbered in `breaks`
    (bars count from 1 through the whole score). `every_bar_alone` puts each bar on its own system
    and leaves the header out (used to measure the bars). `small` gives, for the bars whose words do
    not fit on a line, the size of their words in percent."""
    meta = score["meta"]
    # The namespace is written as a plain xmlns attribute, not with ET.register_namespace: that
    # registry is shared by the whole process, and WeasyPrint (imported for PDFs) changes it.
    root = ET.Element("mei", {"xmlns": MEI_NS, "meiversion": "5.0"})
    head = _el(_el(_el(root, "meiHead"), "fileDesc"), "titleStmt")
    _el(head, "title", text=meta.get("title", ""))
    if meta.get("artist"):
        _el(_el(head, "respStmt"), "persName", {"role": "artist"}, meta["artist"])
    _el(root[0][0], "pubStmt")
    mei_score = _el(_el(_el(_el(root, "music"), "body"), "mdiv"), "score")
    time, tempo = score["time"], score.get("tempo")
    score_def = _el(mei_score, "scoreDef", {"keysig": key_signature(key), "meter.count": time[0], "meter.unit": time[1]})
    if not every_bar_alone:
        _header(score_def, score, capo, units)
    _el(_el(score_def, "staffGrp"), "staffDef", {"n": 1, "lines": 5, "clef.shape": "G", "clef.line": 2})

    n = 0
    total = sum(len(s["bars"]) for s in score["sections"])
    ringing = None
    for section in score["sections"]:
        mei_section = _el(mei_score, "section", {"label": section["name"]} if section["name"] else None)
        for i, bar in enumerate(section["bars"]):
            n += 1
            if n > 1 and (every_bar_alone or n in breaks):
                _el(mei_section, "sb")
            count, unit = bar["time"]
            beats = len(bar["beats"])
            compound = beats != count
            new_meter = (count, unit) != time
            new_tempo = n > 1 and bar["tempo"] and (new_meter or bar["tempo"] != tempo)
            if new_meter:
                _el(mei_section, "scoreDef", {"meter.count": count, "meter.unit": unit})
            time, tempo = (count, unit), bar["tempo"] or tempo
            attrs = {"n": n, XML_ID: f"bar-{n}"}
            if n == total:
                attrs["right"] = "end"
            elif i == len(section["bars"]) - 1:
                attrs["right"] = "dbl"
            measure = _el(mei_section, "measure", attrs)
            layer = _el(_el(measure, "staff", {"n": 1}), "layer", {"n": 1})
            for beat in bar["beats"]:
                note = {"dur": 4 if compound else unit, "pname": "b", "oct": 4,
                        "head.shape": "slash", "stem.visible": "false"}
                if compound:
                    note["dots"] = 1
                el = _el(layer, "note", note)
                if beat["words"]:
                    syl = _el(_el(el, "verse", {"n": 1}), "syl")
                    if small and n in small:
                        syl = _el(syl, "rend", {"fontsize": f"{small[n]}%"})
                    syl.text = " ".join(beat["words"])
            if new_tempo and (mark := tempo_mark(bar["tempo"], (count, unit))):
                # Written as rehearsal text, not as <tempo>: Verovio puts a tempo mark between the
                # staff and the chord symbols, which lifts the chord line of the whole system.
                # Rehearsal text goes above the chords, where an edition prints a tempo.
                el = _el(measure, "reh", {"tstamp": 1, "staff": 1, "type": "tempo"})
                _el(el, "rend", {"glyph.auth": "smufl"}, mark[0]).tail = f" = {mark[1]}"
            if i == 0 and section["name"]:
                _el(measure, "reh", {"tstamp": 1, "staff": 1, "type": "section"}, section["name"])
            step = count / beats                # one beat in time-signature units (3 eighths in 6/8)
            for b, beat in enumerate(bar["beats"]):
                for k, (name, index) in enumerate(beat["chords"]):
                    shown = name if name == NO_CHORD else move_chord_name(name, semis)
                    if index is None and shown == ringing and not (i == 0 and b == 0):
                        continue        # repeated by "%": only printed where the chord really changes
                    ringing = shown
                    attrs = {"tstamp": f"{1 + (b + k / len(beat['chords'])) * step:g}", "staff": 1, "place": "above"}
                    if index is not None:
                        attrs[XML_ID] = f"chord-{index}"
                    _el(_el(measure, "harm", attrs), "rend", {"fontweight": "bold"}, engraved_chord(shown))
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def _header(score_def, score: dict, capo: int, units: int) -> None:
    """Title (centred), capo under it, tempo mark on the left and artist on the right, as in an edition."""
    meta = score["meta"]
    room = (units or 2100) - MARGIN_LEFT - MARGIN
    head = _el(score_def, "pgHead")
    title = meta.get("title", "").strip()
    centre = _el(head, "rend", {"halign": "center", "valign": "top"})
    if title:
        # about 31 units per letter at x-large, 24 at large, 19 at the normal size
        size = "x-large" if len(title) * 31 <= room else "large" if len(title) * 24 <= room else "normal"
        if len(title) * 19 > room:
            title = title[:max(8, room // 19 - 1)].rstrip() + "…"
        _el(centre, "rend", {"type": "title", "fontsize": size}, title)
    if capo:
        if title:
            _el(centre, "lb")
        _el(centre, "rend", {"type": "capo", "fontstyle": "italic"}, f"Capo {capo}")
    if mark := tempo_mark(score.get("tempo"), score["time"]):
        left = _el(head, "rend", {"halign": "left", "valign": "bottom", "type": "tempo", "fontweight": "bold"})
        _el(left, "rend", {"glyph.auth": "smufl"}, mark[0]).tail = f" = {mark[1]}"
    artist = meta.get("artist", "").strip()
    if artist:
        if len(artist) * 19 > room - 220:
            artist = artist[:max(8, (room - 220) // 19 - 1)].rstrip() + "…"
        _el(head, "rend", {"halign": "right", "valign": "bottom", "type": "artist"}, artist)


# --- Verovio --------------------------------------------------------------------------------------

_LOCK = threading.Lock()      # one toolkit, one render at a time (the toolkit is not thread-safe)
_toolkit = None

_OPTIONS = {
    "footer": "none", "header": "encoded", "breaks": "encoded", "adjustPageHeight": True, "pageHeight": 60000,
    "pageMarginLeft": MARGIN_LEFT, "pageMarginRight": MARGIN, "pageMarginTop": MARGIN, "pageMarginBottom": MARGIN,
    "svgViewBox": True, "svgRemoveXlink": True, "svgFormatRaw": True, "scale": 100, "minLastJustification": 0.75,
    # The notes of tempo marks and the accidentals of chord symbols are text in Verovio's music font.
    # By default every SVG embeds that font (60 kB); the page loads it once instead (see music_font).
    "smuflTextFont": "none", "lyricSize": LYRIC_SIZE,
}
FONT_FAMILY = "Leipzig"     # Verovio's default music font, named in the SVG it writes


def music_font() -> bytes:
    """The music font (WOFF2) that the text of the SVG refers to, as shipped with Verovio."""
    import verovio

    css = (Path(verovio.__file__).parent / "data" / f"{FONT_FAMILY}.css").read_text()
    return base64.b64decode(re.search(r"base64,([A-Za-z0-9+/=]+)", css).group(1))


def _render(mei: str, options: dict) -> list[str]:
    """Every page of the MEI as SVG."""
    global _toolkit
    import verovio

    if _toolkit is None:
        verovio.enableLog(verovio.LOG_ERROR)
        _toolkit = verovio.toolkit()
    _toolkit.resetOptions()
    _toolkit.setOptions({**_OPTIONS, **options})
    _toolkit.resetXmlIdSeed(1)                   # same input, same output
    if not _toolkit.loadData(mei):
        raise ValueError("Verovio could not read the score")
    return [_toolkit.renderToSVG(page) for page in range(1, _toolkit.getPageCount() + 1)]


_ANON_ID = re.compile(r' id="[a-z0-9]+"')


def _slim(svg: str) -> str:
    """Drop the ids Verovio gives to every element (a tenth of the file). The ids written here
    (bar-12, chord-7) and those of the glyph definitions all have a hyphen and stay."""
    head, mark, body = svg.partition('<svg class="definition-scale"')
    return head + mark + _ANON_ID.sub("", body)


_BAR = re.compile(r'<g id="bar-(\d+)" class="measure">.*?<path d="M(-?\d+) -?\d+ L(-?\d+) .*?'
                  r'class="notehead">\s*<use [^>]*translate\((-?\d+),', re.S)


def _bar_widths(pages: list[str]) -> tuple[list[float], float]:
    """From a layout with every bar alone on its system: the room each bar needs after a bar line
    (with its time signature where the meter changes), and the room a system needs before its first
    bar (clef and key signature)."""
    seen = {}
    for svg in pages:
        for m in _BAR.finditer(svg):
            n, x1, x2, note = (int(g) for g in m.groups())
            seen[n] = ((x2 - note) / 10 + LEAD, (note - x1) / 10 - LEAD)
    lead = min(before for _, before in seen.values())
    return [seen[n][0] + seen[n][1] - lead for n in sorted(seen)], lead


def choose_breaks(widths: list[float], lead: float, room: float, section_ends: set[int] = frozenset()) -> set[int]:
    """Bars (numbered from 1) that start a new system: as many bars on a system as fit in `room`
    after the `lead` every system begins with; a section that ends on a system already well filled
    takes the break, so that most sections start a line. Then the systems are evened out: one much
    emptier than the system before it takes that system's last bars (three bars and one become two
    and two), except across the start of a section."""
    starts = [1]
    used = lead
    for n, width in enumerate(widths, start=1):
        if n > 1 and (used + width > room * FILL or (n - 1 in section_ends and used >= room * SECTION_FILL)):
            starts.append(n)
            used = lead
        used += width

    def filled(first: int, stop: int) -> float:
        return lead + sum(widths[first - 1:stop - 1])

    for i in range(len(starts) - 1, 0, -1):
        stop = starts[i + 1] if i + 1 < len(starts) else len(widths) + 1
        while starts[i] - 1 > starts[i - 1] and starts[i] - 1 not in section_ends:
            before, here, moved = filled(starts[i - 1], starts[i]), filled(starts[i], stop), widths[starts[i] - 2]
            if here + moved > room * FILL or abs((before - moved) - (here + moved)) >= abs(before - here):
                break
            starts[i] -= 1
    return set(starts[1:])


class TooLong(ValueError):
    """The chart has more bars than we are willing to engrave."""


@dataclass(frozen=True)
class Engraving:
    pages: tuple[str, ...]      # SVG, one per page (a single tall page unless the song is very long)
    bars: int
    unbarred: int               # bars that were guessed from lines without bar lines


_cache: OrderedDict[str, Engraving] = OrderedDict()


def engrave(text: str, *, width: int = 400, capo: int | None = None, key: int = 0) -> Engraving:
    """Engrave chart text for a screen `width` CSS pixels wide, as the shapes to play with `capo`
    (default: the chart's own) after transposing by `key` semitones. Cached."""
    units = layout_units(width)
    digest = hashlib.sha256(f"{units}|{capo}|{key}|{text}".encode()).hexdigest()
    with _LOCK:
        if digest in _cache:
            _cache.move_to_end(digest)
            return _cache[digest]
        result = _engrave(text, units, capo, key)
        _cache[digest] = result
        while len(_cache) > 1 and (len(_cache) > CACHE_SIZE
                                   or sum(len(svg) for e in _cache.values() for svg in e.pages) > CACHE_BYTES):
            _cache.popitem(last=False)
        return result


def _engrave(text: str, units: int, capo: int | None, key: int) -> Engraving:
    score = read_score(text)
    bars = sum(len(s["bars"]) for s in score["sections"])
    if bars == 0:
        return Engraving((), 0, 0)
    if bars > MAX_BARS:
        raise TooLong("This chart is too long to engrave.")
    written_for = chart_capo(score["meta"])
    capo = written_for if capo is None else capo
    shapes = dict(semis=written_for + key - capo, capo=capo,
                  key=move_chord_name(score["meta"]["key"], key - capo) if score["meta"].get("key") else "")

    widths, lead = _bar_widths(_render(to_mei(score, **shapes, every_bar_alone=True),
                                       {"header": "none", "noJustification": True, "pageWidth": 20000}))
    if len(widths) != bars:
        raise ValueError("Verovio did not lay out every bar")
    ends, n = set(), 0
    for section in score["sections"]:
        n += len(section["bars"])
        ends.add(n)
    room = units - MARGIN_LEFT - MARGIN
    small = {}
    for n, width in enumerate(widths, start=1):
        if lead + width > room * FILL:      # too many words for one line: set them smaller
            widest = room * FILL - lead
            small[n] = max(MIN_WORD_SIZE, int(95 * widest / width))
            widths[n - 1] = widest
    breaks = choose_breaks(widths, lead, room, ends)
    pages = _render(to_mei(score, **shapes, breaks=breaks, units=units, small=small), {"pageWidth": units})
    return Engraving(tuple(_slim(svg) for svg in pages), bars, score["unbarred"])
