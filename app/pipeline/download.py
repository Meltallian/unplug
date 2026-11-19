"""Stage 1: fetch the audio from YouTube and keep it (compressed) in the song folder."""

import json
from pathlib import Path

import yt_dlp


def download(youtube_id: str, song_dir: Path) -> dict:
    """Download best audio to song_dir/audio.<ext> and write meta.json. Cached."""
    song_dir.mkdir(parents=True, exist_ok=True)
    meta_path = song_dir / "meta.json"
    existing = list(song_dir.glob("audio.*"))
    if meta_path.exists() and existing:
        return json.loads(meta_path.read_text())

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(song_dir / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    cookies = song_dir.parent.parent / "youtube-cookies.txt"
    if cookies.exists():
        opts["cookiefile"] = str(cookies)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=True)

    audio = next(song_dir.glob("audio.*"))
    meta = {
        "youtube_id": youtube_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "artist": info.get("artist"),
        "track": info.get("track"),
        "duration": info.get("duration"),
        "audio_file": audio.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta
