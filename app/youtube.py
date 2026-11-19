import re

_ID = r"([A-Za-z0-9_-]{11})"
_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)" + _ID),
    re.compile(r"(?:youtu\.be/)" + _ID),
    re.compile(r"(?:youtube\.com/(?:shorts|embed|live)/)" + _ID),
]


def parse_youtube_id(text: str) -> str | None:
    text = text.strip()
    for pattern in _PATTERNS:
        if m := pattern.search(text):
            return m.group(1)
    if re.fullmatch(_ID, text):
        return text
    return None
