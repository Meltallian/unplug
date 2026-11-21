"""Stage 6: lyrics from LRCLIB (synced if possible), placed on the video's timeline.

LRCLIB times belong to the recording the lyrics were timed on, usually the album track, and are
not always right for it either. A video often adds an intro, an outro or a scene. So the times
are checked by listening (words.py: are the lines heard where their times say?) and, when they
are not, the lines are placed again:

1. Reference recording: the album track is looked up on YouTube by its exact length, and the two
   recordings are matched by their chords, window by window. Exact to a fraction of a second,
   and it follows a scene inserted in the middle. Only possible when the lengths differ.
2. By ear: the lines are looked for in what the model hears sung, each line on its own.
3. Vocal activity (last resort): one shift for the whole song, the one that puts sung lines on
   activity in the vocal band, among those that keep every line on music.

Then every word gets its own time: the chord goes over the word sung when it changes.

Writes lyrics.json:
  {"source": "lrclib"|"pasted"|"none", "synced": bool, "offset": float, "match": {...},
   "lines": [{"t": float|None, "text": str, "lrc_t": float, "w": [float, ...]}],
   "aligned": "same"|"reference"|"heard"|"activity", "heard": share of lines heard in place,
   "words": "heard"|"syllables"}
Line times ("t") and word times ("w", one per word of the line) are video times; "offset" is 0
for lyrics placed here (it remains for older files) and "lrc_t" keeps the LRCLIB time.
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
OFFSET_VERSION = 5  # bump when the placement changes: cached lyrics are placed again
REFERENCE_TOL = 2.0  # seconds; the reference recording must be this close in length to the lyrics' own
WINDOW_S, HOP_S = 30.0, 10.0   # the recordings are matched in windows of this length, this far apart
MIN_MATCH = 0.6     # mean chord agreement (0..1) a window needs to count as found in the video
MIN_FOUND = 0.5     # share of windows that must be found for the reference to be trusted
LINE_WINDOW = 2.0  # seconds of music a sung line needs under its start
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
            text = m.group(3).strip().replace("[", "(").replace("]", ")")  # never read as chords
            lines.append({"t": int(m.group(1)) * 60 + float(m.group(2)), "text": text})
    return lines


def plain_lines(text: str) -> list[dict]:
    """Plain lyrics; section headers like [Chorus] and stray brackets are dropped."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if re.fullmatch(r"\[[^\]]*\]|\([^)]*\)", line):
            continue
        lines.append({"t": None, "text": line.replace("[", "(").replace("]", ")")})
    return lines


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
                    video_duration: float, lyrics_duration: float | None,
                    music: list[tuple[float, float]] | None = None) -> float:
    """Shift (s) to add to LRC times so sung lines land on high vocal-band activity and long gaps on low.

    `music`: the stretches of the video where chords were found. A line is sung over music, so
    shifts that push lines into silence (a video intro or outro without the band) are ruled out
    first; among the others the vocal-band activity decides. Without this, loud guitars pass
    for singing and a video with a long outro gets its lyrics tens of seconds late."""
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

    def off_the_music(offset: float) -> float:
        """Share of sung lines that would start where there is no music."""
        missed = 0
        for t in times:
            a, b = t + offset, t + offset + LINE_WINDOW
            covered = sum(max(0.0, min(b, e) - max(a, s)) for s, e in music)
            missed += covered < LINE_WINDOW / 2
        return missed / len(times)

    candidates = np.arange(lo, hi + 1e-9, 0.25)
    if music:
        missed = np.array([off_the_music(c) for c in candidates])
        candidates = candidates[missed <= missed.min() + 0.03]
    best = max(candidates, key=score)
    return round(float(best), 2)


def find_reference(artist: str, track: str, lyrics_duration: float, exclude_id: str) -> str | None:
    """YouTube id of the recording the lyrics were timed on: an upload of exactly its length."""
    import yt_dlp

    search = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "noplaylist": True})
    for query in (f"{artist} {track} topic", f"{artist} {track}"):       # "topic": the label's audio upload
        found = search.extract_info(f"ytsearch8:{query}", download=False).get("entries") or []
        for entry in found:
            if entry.get("id") != exclude_id and entry.get("duration") \
                    and abs(entry["duration"] - lyrics_duration) <= REFERENCE_TOL:
                return entry["id"]
    return None


def match_recordings(reference: np.ndarray, video: np.ndarray, step: float, lo: float, hi: float) -> list[tuple[float, float]] | None:
    """Where each part of the reference recording is in the video.

    Both are (frames × chords) probabilities, `step` seconds per frame. Returns [(reference time,
    shift)]: a window of the reference centred on that time is found `shift` seconds later in the
    video (lo <= shift <= hi). None if too few windows are found (not the same recording)."""
    def unit(m: np.ndarray) -> np.ndarray:
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

    # Only chords can say where a passage is: frames the model hears as "no chord" (the last
    # column) match any other such frame, at any lag. They are left out, and a window that is
    # mostly such frames counts as not found.
    chords_ref = reference[:, :-1] * (np.argmax(reference, axis=1) != reference.shape[1] - 1)[:, None]
    chords_vid = video[:, :-1] * (np.argmax(video, axis=1) != video.shape[1] - 1)[:, None]
    voiced = np.any(chords_ref > 0, axis=1)
    ref, vid = unit(chords_ref), unit(chords_vid)
    lags = np.arange(int(np.floor(lo / step)), int(np.ceil(hi / step)) + 1)
    window, hop = int(WINDOW_S / step), int(HOP_S / step)
    out, found = [], 0
    for start in range(0, max(1, len(ref) - window + 1), hop):
        part = ref[start:start + window]
        best_lag, best = None, -1.0
        for lag in lags:
            a = start + lag
            if a < 0 or a + len(part) > len(vid):
                continue
            agreement = float(np.mean(np.sum(part * vid[a:a + len(part)], axis=1)))
            if agreement > best:
                best_lag, best = lag, agreement
        centre = (start + len(part) / 2) * step
        music = float(np.mean(voiced[start:start + window]))
        ok = best_lag is not None and music >= 0.5 and best >= MIN_MATCH * music
        found += ok
        out.append((centre, best_lag * step if ok else None))
    if not out or found < MIN_FOUND * len(out):
        return None
    known = [(t, shift) for t, shift in out if shift is not None]      # windows not found take their neighbour's shift
    filled = [(t, shift if shift is not None else min(known, key=lambda k: abs(k[0] - t))[1]) for t, shift in out]
    shifts = [s for _, s in filled]
    smooth = [float(np.median(shifts[max(0, i - 1):i + 2])) for i in range(len(shifts))]   # one odd window is noise
    return [(t, round(s, 2)) for (t, _), s in zip(filled, smooth)]


def shift_at(shifts: list[tuple[float, float]], t: float) -> float:
    """The shift of the window nearest to reference time t."""
    return min(shifts, key=lambda k: abs(k[0] - t))[1]


def reference_shifts(song_dir: Path, wav: Path, meta: dict, match: dict, cookies_file: Path | None) -> list[tuple[float, float]] | None:
    """match_recordings() between the recording the lyrics were timed on and this video; None
    when that recording is not found or does not match. Its audio is not kept."""
    import shutil

    import librosa

    from app.pipeline import btc, chords
    from app.pipeline.audio import decode
    from app.pipeline.download import download

    artist, track = match.get("artistName") or "", match.get("trackName") or ""
    duration, video_duration = match.get("duration"), meta.get("duration") or 0
    if not (artist and track and duration):
        return None
    ref_id = find_reference(artist, track, duration, meta.get("youtube_id", ""))
    if not ref_id:
        return None
    ref_dir = song_dir / "cache" / "reference"
    try:
        download(ref_id, ref_dir, cookies_file)
        y, sr = librosa.load(decode(ref_dir), sr=None, mono=True)
        reference = chords.plain_probabilities(btc.chord_probabilities(y, sr))
    finally:
        shutil.rmtree(ref_dir, ignore_errors=True)
    video = chords.plain_probabilities(chords.features(wav)["probs"])
    diff = video_duration - duration
    return match_recordings(reference, video, btc.FRAME_S, min(0.0, diff) - 3.0, max(0.0, diff) + 3.0)


def music_stretches(analysis: dict | None) -> list[tuple[float, float]] | None:
    """Where the chord stage found chords: [(start, end)], neighbours joined."""
    if not analysis:
        return None
    out: list[list[float]] = []
    for seg in analysis.get("segments", []):
        if seg.get("chord") is None:
            continue
        if out and seg["start"] - out[-1][1] < 0.5:
            out[-1][1] = seg["end"]
        else:
            out.append([seg["start"], seg["end"]])
    return [(a, b) for a, b in out] or None


FITS = 0.35   # share of lines that must be heard where their times say for the times to be believed
              # (on test songs: 65 % of lines in the right place are heard, 4 % of lines 17 s away)


def place(lines: list[dict], song_dir: Path, wav: Path, meta: dict, match: dict, analysis: dict | None,
          cookies_file: Path | None = None) -> dict:
    """Put synced lines and their words on the video's timeline.

    Returns {"lines": [{"t", "lrc_t", "text", "w"}], "offset": 0.0, "aligned": how the lines were
    placed, "words": how the words were, "heard": share of lines heard where they were put}.
    The LRCLIB times are tried as they are first; if the lines are not heard there, the lyrics'
    own recording is looked up and matched to the video ("reference"), or the lines are looked
    for in what is sung ("heard"); vocal activity is the last resort. Never raises."""
    from app.pipeline import words

    lines = [{"t": ln.get("lrc_t", ln["t"]), "text": ln["text"]} for ln in lines]       # back to LRCLIB times
    video_duration, duration = meta.get("duration") or 0, match.get("duration")
    differ = bool(duration) and abs(video_duration - duration) > SAME_LENGTH

    try:
        import librosa

        y, _ = librosa.load(wav, sr=16000, mono=True)
        first = next((ln["t"] for ln in lines if ln["text"]), 0.0)
        language = words.language_of(y, first)
    except Exception:      # no audio, no model: the old way, unchecked
        y = language = None

    def listen(shifts: list[float]) -> tuple[list[dict], float]:
        """The lines moved by `shifts` with their word times, and the share of sung lines heard there."""
        moved = [{"t": round(ln["t"] + s, 2), "lrc_t": ln["t"], "text": ln["text"]} for ln, s in zip(lines, shifts)]
        sung = [ln for ln in moved if ln["text"]]
        times, share = None, 0.0
        if y is not None:
            try:
                report: list[dict] = []
                times = words.align(sung, y, 16000, language, report=report)
                share = float(np.mean([r["heard"] for r in report])) if report else 0.0
            except Exception:
                times = None
        for ln, ts in zip(sung, times if times is not None else words.by_syllables(sung)):
            ln["w"] = ts
        return moved, share

    def done(how: str, placed: tuple[list[dict], float]) -> dict:
        return {"lines": placed[0], "offset": 0.0, "aligned": how, "heard": round(placed[1], 2),
                "words": "heard" if placed[1] > 0 else "syllables"}

    as_timed = listen([0.0] * len(lines))
    if as_timed[1] >= FITS or (y is None and not differ):
        return done("same", as_timed)

    if differ:
        try:
            shifts = reference_shifts(song_dir, wav, meta, match, cookies_file)
        except Exception:      # lyrics are optional: no network, no such upload, a failed download ...
            shifts = None
        if shifts:
            return done("reference", listen([shift_at(shifts, ln["t"]) for ln in lines]))

    if y is not None:
        try:
            found = words.find_lines(lines, words.transcribe(y, language))
        except Exception:
            found = None
        if found:
            by_ear = listen(found[0])
            if by_ear[1] > as_timed[1]:
                return done("heard", by_ear)

    if differ:
        try:
            act, hop_s = vocal_activity(wav)
            offset = estimate_offset(lines, act, hop_s, video_duration, duration, music_stretches(analysis))
            return done("activity", listen([offset] * len(lines)))
        except Exception:
            pass
    return done("same", as_timed)


def fetch_lyrics(song_dir: Path, wav: Path, meta: dict, analysis: dict | None = None,
                 cookies_file: Path | None = None) -> dict:
    """Lyrics for a song (cached). `analysis`: the chord stage's result, used to place the lyrics."""
    out = song_dir / "cache" / "lyrics.json"
    if out.exists():
        try:
            cached = json.loads(out.read_text())
            if cached.get("source") == "lrclib" and cached.get("synced") and cached.get("match") \
                    and cached.get("offset_version") != OFFSET_VERSION:
                cached.update(place(cached["lines"], song_dir, wav, meta, cached["match"], analysis, cookies_file))
                cached["offset_version"] = OFFSET_VERSION
                save_lyrics(out, cached)
            if cached.get("source") != "none":
                return cached
        except json.JSONDecodeError:
            pass
    artist, track = guess_artist_track(meta)
    result = {"source": "none", "synced": False, "lines": [], "offset": 0.0, "query": [artist, track]}
    try:
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20) as client:
            match = search_lrclib(artist, track, meta.get("duration"), client)
    except Exception as exc:  # lyrics are optional: never fail the song over them
        result["error"] = f"LRCLIB: {type(exc).__name__}"
        match = None
    if match:
        synced = bool(match.get("syncedLyrics"))
        kept = {k: match.get(k) for k in ("id", "artistName", "trackName", "albumName", "duration")}
        result = {"source": "lrclib", "synced": synced, "offset": 0.0, "offset_version": OFFSET_VERSION,
                  "query": [artist, track], "match": kept}
        if synced:
            result.update(place(parse_lrc(match["syncedLyrics"]), song_dir, wav, meta, kept, analysis, cookies_file))
        else:
            result["lines"] = plain_lines(match.get("plainLyrics") or "")
    save_lyrics(out, result)
    return result


def save_lyrics(out: Path, data: dict) -> None:
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(out)
