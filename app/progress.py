"""Progress shown while a song is processed: current stage, time spent, typical time, time left.

Typical times come from the last songs that finished (median), scaled to the song's length for
the stages whose cost grows with it.
"""

from datetime import datetime, timezone
from statistics import median

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Song, SongStatus

STAGES = ["download", "chords", "lyrics", "chart"]
LABELS = {"starting": "Starting", "download": "Downloading the audio", "chords": "Detecting chords",
          "lyrics": "Fetching lyrics", "chart": "Building the chart"}
PER_MINUTE = {"chords", "lyrics"}   # cost grows with the song's length
DEFAULTS = {"download": 8.0, "chords": 9.0, "lyrics": 1.5, "chart": 0.5}   # seconds (per audio minute if scaled)
# Faster than this means the stage was served from cache (e.g. a reprocess): not a useful estimate.
MIN_REAL = {"download": 0.5, "chords": 1.0, "lyrics": 0.1, "chart": 0.0}
TYPICAL_LENGTH = 240.0              # seconds of audio assumed before the length is known


def typical(session: Session) -> dict[str, float]:
    """Median seconds per stage (per audio minute for scaled stages) over recent finished songs."""
    rows = session.execute(
        select(Song.timings, Song.duration).where(Song.status == SongStatus.done, Song.timings.isnot(None))
        .order_by(Song.updated_at.desc()).limit(30)).all()
    out = dict(DEFAULTS)
    for stage in STAGES:
        values = []
        for timings, duration in rows:
            if not timings or stage not in timings or (stage in PER_MINUTE and not duration):
                continue
            value = timings[stage] / (duration / 60) if stage in PER_MINUTE else timings[stage]
            if value >= MIN_REAL[stage]:
                values.append(value)
        if values:
            out[stage] = median(values)
    return out


def expected(stage: str, typ: dict[str, float], duration: float | None) -> float:
    minutes = (duration or TYPICAL_LENGTH) / 60
    return typ.get(stage, 0.0) * (minutes if stage in PER_MINUTE else 1)


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def progress(session: Session, song: Song) -> dict | None:
    """Live numbers for the status line of a song being processed (None if not processing)."""
    if song.status != SongStatus.processing:
        return None
    now = datetime.now(timezone.utc)
    typ = typical(session)
    stage = song.stage or "starting"
    elapsed = (now - _aware(song.stage_started_at)).total_seconds() if song.stage_started_at else 0.0
    exp = expected(stage, typ, song.duration) if stage in STAGES else 0.0
    later = STAGES[STAGES.index(stage) + 1:] if stage in STAGES else STAGES
    left = max(exp - elapsed, 0.0) + sum(expected(s, typ, song.duration) for s in later)
    return {"label": LABELS.get(stage, stage), "elapsed": elapsed, "expected": exp, "left": left,
            "step": STAGES.index(stage) + 1 if stage in STAGES else 0, "steps": len(STAGES)}


def queue_wait(session: Session, position: int) -> float:
    """Rough wait before a queued song starts: songs ahead × a typical song ÷ 2 parallel workers."""
    typ = typical(session)
    per_song = sum(expected(s, typ, None) for s in STAGES)
    return per_song * max(position - 1, 0) / 2


def fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min {seconds % 60:02d} s"
