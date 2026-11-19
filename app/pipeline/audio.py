"""Stage 2: decode the stored audio to a mono wav for analysis (temporary, recreated on demand)."""

import json
import subprocess
from pathlib import Path

SAMPLE_RATE = 22050


def decode(song_dir: Path) -> Path:
    wav = song_dir / "cache" / "audio.wav"
    if wav.exists():
        return wav
    meta = json.loads((song_dir / "meta.json").read_text())
    wav.parent.mkdir(exist_ok=True)
    tmp = wav.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(song_dir / meta["audio_file"]),
         "-ac", "1", "-ar", str(SAMPLE_RATE), str(tmp)],
        check=True,
    )
    tmp.rename(wav)
    return wav
