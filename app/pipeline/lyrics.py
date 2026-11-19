"""Stage 6: lyrics from LRCLIB (synced if possible), with a global timing offset for the video.

Writes lyrics.json:
  {"source": "lrclib"|"pasted"|"none", "synced": bool, "lines": [{"t": float|None, "text": str}],
   "offset": float, "match": {...}}
"""

import json
import re
from pathlib import Path

import httpx
import numpy as np

LRCLIB = "https://lrclib.net/api"
USER_AGENT = "unplug/0.1 (private chord sheet app)"
_NOISE = re.compile(
    r"\s*[\(\[][^)\]]*\b(official|video|audio|lyrics?|visuali[sz]er|4k|hd|hq|remaster(ed)?|upgrade|mv)\b[^)\]]*[\)\]]",
    re.I)
_VARIANTS = ("unplugged", "acoustic", "live", "remix", "demo", "piano", "orchestral")
SAME_LENGTH = 3.0  # seconds; lyrics this close in length to the video need no offset
_LRC_LINE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")


def guess_artist_track(meta: dict) -> tuple[str, str]:
    if meta.get("artist") and meta.get("track"):
        return meta["artist"], meta["track"]
    title = _NOISE.sub("", meta.get("title") or "")
    title = re.sub(r"[\"“”]", "", title)
    title = re.sub(r"\s+\|\s+.*$", "", title)             # "... | Lyric Video"
    parts = re.split(r"\s*[-–—]\s+|\s+[-–—]\s*", title, maxsplit=1)
    if len(parts) == 2:
        artist, track = parts
    else:
        artist, track = meta.get("uploader") or "", title
    track = re.sub(r"\s+(feat\.?|ft\.?|featuring)\s.*$", "", track, flags=re.I)
    track = re.sub(r"\s*[\(\[]?(feat\.?|ft\.?)[^)\]]*[\)\]]", "", track, flags=re.I)
    return artist.strip(), re.sub(r"\s+", " ", track).strip()


def parse_lrc(text: str) -> list[dict]:
    lines = []
    for raw in text.splitlines():
        m = _LRC_LINE.match(raw.strip())
        if m:
            lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "text": m.group(3).strip()})
    return lines


def plain_lines(text: str) -> list[dict]:
    return [{"t": None, "text": line.strip()} for line in text.strip().splitlines()]


def _variants(s: str) -> set[str]:
    s = s.lower()
    return {v for v in _VARIANTS if v in s}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def choose_match(results: list[dict], artist: str, track: str, duration: float | None) -> dict | None:
    """Best LRCLIB result: right artist and title, same variant (unplugged/acoustic/...), synced, close duration."""
    want_variants = _variants(track)
    base_track = _norm(re.sub(r"[\(\[].*?[\)\]]", "", track))

    def score(r: dict) -> float | None:
        if artist and _norm(artist) not in _norm(r.get("artistName", "")) \
                and _norm(r.get("artistName", "")) not in _norm(artist):
            return None
        if base_track and base_track not in _norm(r.get("trackName", "")):
            return None
        s = 0.0
        s += 30 if r.get("syncedLyrics") else (0 if r.get("plainLyrics") else -100)
        s -= 15 * len(want_variants ^ _variants(r.get("trackName", "") + " " + (r.get("albumName") or "")))
        if duration and r.get("duration"):
            s -= min(abs(duration - r["duration"]), 60) / 2
        return s

    scored = [(score(r), r) for r in results]
    scored = [(s, r) for s, r in scored if s is not None and s > -50]
    return max(scored, key=lambda x: x[0])[1] if scored else None


def search_lrclib(artist: str, track: str, duration: float | None, client: httpx.Client) -> dict | None:
    queries = [{"q": f"{artist} {track}"}, {"track_name": track, "artist_name": artist}]
    base = re.sub(r"\s*[\(\[].*?[\)\]]", "", track).strip()
    if base != track:
        queries.append({"track_name": base, "artist_name": artist})
    for params in queries:
        r = client.get(f"{LRCLIB}/search", params=params)
        r.raise_for_status()
        if match := choose_match(r.json(), artist, track, duration):
            return match
    return None


def vocal_activity(wav: Path, hop_s: float = 0.1) -> tuple[np.ndarray, float]:
    """Rough per-frame activity in the vocal band (300-3400 Hz), standardised. Returns (activity, hop seconds)."""
    import librosa

    y, sr = librosa.load(wav, sr=None, mono=True)
    hop = int(sr * hop_s)
    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = spec[(freqs >= 300) & (freqs <= 3400)].sum(axis=0)
    act = np.log1p(band)
    return (act - act.mean()) / (act.std() + 1e-9), hop_s


def estimate_offset(lines: list[dict], activity: np.ndarray, hop_s: float,
                    video_duration: float, lyrics_duration: float | None) -> float:
    """Shift (s) to add to LRC times so sung lines land on high vocal-band activity and long gaps on low."""
    times = [ln["t"] for ln in lines if ln["text"]]
    if not times:
        return 0.0
    diff = (video_duration - lyrics_duration) if lyrics_duration else 0.0
    if abs(diff) <= SAME_LENGTH:
        return 0.0  # same length: almost surely the same audio and timing
    lo, hi = min(0.0, diff) - 3.0, max(0.0, diff) + 3.0

    def mean_act(a: float, b: float) -> float | None:
        i, j = int(a / hop_s), int(b / hop_s)
        if j <= i or i < 0 or j > len(activity):
            return None
        return float(activity[i:j].mean())

    def score(offset: float) -> float:
        sung, gaps = [], []
        for t, nxt in zip(times, times[1:] + [None]):
            if (m := mean_act(t + offset, t + offset + 3.0)) is not None:
                sung.append(m)
            if nxt is not None and nxt - t > 8.0 and (m := mean_act(t + offset + 5.0, nxt + offset - 0.5)) is not None:
                gaps.append(m)
        if not sung:
            return -1e9
        return float(np.mean(sung)) - (float(np.mean(gaps)) if gaps else 0.0)

    candidates = np.arange(lo, hi + 1e-9, 0.25)
    best = max(candidates, key=score)
    return round(float(best), 2)


def fetch_lyrics(song_dir: Path, wav: Path, meta: dict) -> dict:
    out = song_dir / "cache" / "lyrics.json"
    if out.exists():
        try:
            cached = json.loads(out.read_text())
            if cached.get("source") != "none":
                return cached
        except json.JSONDecodeError:
            pass
    artist, track = guess_artist_track(meta)
    result = {"source": "none", "synced": False, "lines": [], "offset": 0.0, "query": [artist, track]}
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20) as client:
            match = search_lrclib(artist, track, meta.get("duration"), client)
    except httpx.HTTPError as exc:
        result["error"] = f"LRCLIB: {type(exc).__name__}"
        match = None
    if match:
        synced = bool(match.get("syncedLyrics"))
        lines = parse_lrc(match["syncedLyrics"]) if synced else plain_lines(match.get("plainLyrics") or "")
        offset = 0.0
        if synced:
            act, hop_s = vocal_activity(wav)
            offset = estimate_offset(lines, act, hop_s, meta.get("duration") or 0, match.get("duration"))
        result = {"source": "lrclib", "synced": synced, "lines": lines, "offset": offset,
                  "query": [artist, track],
                  "match": {k: match.get(k) for k in ("id", "artistName", "trackName", "albumName", "duration")}}
    save_lyrics(out, result)
    return result


def save_lyrics(out: Path, data: dict) -> None:
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(out)
