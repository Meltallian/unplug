"""Stage 1: fetch the audio from YouTube and keep it (compressed) in the song folder."""

import json
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import yt_dlp

RETRIES = 2        # extra attempts after a failed download
RETRY_WAIT = 3.0   # seconds between attempts


@contextmanager
def _writable_copy(cookies_file: Path | None):
    """yt-dlp writes the cookie jar back on close, but secrets are mounted read-only:
    hand it a private copy (one per download, so parallel workers never share a file)."""
    if not (cookies_file and cookies_file.exists()):
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="unplug-cookies-") as tmp:
        copy = Path(tmp) / "cookies.txt"
        shutil.copyfile(cookies_file, copy)
        yield copy


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

    with _writable_copy(cookies_file) as cookies:
        if cookies:
            opts["cookiefile"] = str(cookies)
        for attempt in range(RETRIES + 1):
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={youtube_id}", download=True)
                break
            except yt_dlp.utils.DownloadError:      # YouTube answers 403 now and then; the next try usually passes
                if attempt == RETRIES:
                    raise
                time.sleep(RETRY_WAIT)

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
