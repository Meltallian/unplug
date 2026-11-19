"""Parse an Ultimate Guitar style chord sheet (chords over lyrics) into sounding chords."""

import re
from dataclasses import dataclass, field

from app.music import Chord, parse_chord, transpose

_SECTION = re.compile(r"^\[(.+)\]\s*$")
_IGNORE_TOKEN = re.compile(r"^(\(?x\d+\)?|\|+|-+|/|\.+)$", re.I)


@dataclass
class Sheet:
    sections: list[tuple[str, list[Chord | str]]] = field(default_factory=list)

    @property
    def chords(self) -> list[Chord | str]:
        return [c for _, cs in self.sections for c in cs]


def chord_line(line: str) -> list[Chord | str] | None:
    """The chords on a line if every token is a chord (or a repeat marker), else None."""
    if "|-" in line or "-|" in line:  # tablature
        return None
    tokens = [t for t in line.split() if not _IGNORE_TOKEN.match(t)]
    if not tokens:
        return None
    parsed = [parse_chord(t) for t in tokens]
    if any(p is None for p in parsed):
        return None
    return parsed


def parse_sheet(text: str, capo: int = 0) -> Sheet:
    """Chords per section, transposed by the capo so they are the chords that sound."""
    sheet = Sheet()
    current = "Start"
    for raw in text.splitlines():
        line = raw.strip()
        if m := _SECTION.match(line):
            current = m.group(1)
            continue
        chords = chord_line(line)
        if chords:
            if not sheet.sections or sheet.sections[-1][0] != current:
                sheet.sections.append((current, []))
            sheet.sections[-1][1].extend(transpose(c, capo) for c in chords)
    return sheet
