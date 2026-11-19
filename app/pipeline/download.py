"""Stage 1: fetch the audio from YouTube and keep it (compressed) in the song folder."""

import json
from pathlib import Path

import yt_dlp


def download(youtube_id: str, song_dir: Path, cookies_file: Path | None = None) -> dict:
    """Download best audio to song_dir/audio.<ext> and write meta.json. Cached."""
    song_dir.mkdir(parents=True, exist_ok=True)
    meta_path = song_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if (song_dir / meta["audio_file"]).exists():
            return meta

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(song_dir / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_file and cookies_file.exists():
        opts["cookiefile"] = str(cookies_file)

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=True)

    audio = Path(info["requested_downloads"][0]["filepath"])
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
