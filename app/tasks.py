import json
from datetime import datetime, timezone

from celery import Celery
from celery.signals import worker_ready
from sqlalchemy import select, update

from app.config import get_settings
from app.db import SessionLocal
from app.models import Chart, Song, SongStatus
from app.pipeline.audio import decode
from app.pipeline.chart import build_chordpro
from app.pipeline.chords import analyze, load_segments
from app.pipeline.download import download
from app.pipeline.lyrics import fetch_lyrics, guess_artist_track

settings = get_settings()
celery = Celery("unplug", broker=settings.redis_url, backend=None)
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True)


def recover_interrupted() -> list[int]:
    """Songs left 'processing' by a worker that was stopped mid-job go back in the queue.

    Runs when the worker starts. There is a single worker container, so at that moment
    nothing can really be processing: any such row is an orphan.
    """
    with SessionLocal() as s:
        stuck = s.scalars(select(Song).where(Song.status == SongStatus.processing)).all()
        for song in stuck:
            song.status, song.stage, song.queued_at = SongStatus.queued, None, datetime.now(timezone.utc)
        s.commit()
        ids = [song.id for song in stuck]
    for song_id in ids:
        process_song.delay(song_id)
    return ids


@worker_ready.connect
def _on_worker_ready(**_) -> None:
    recover_interrupted()


def _set(song_id: int, **fields) -> None:
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        for k, v in fields.items():
            setattr(song, k, v)
        s.commit()


def _stage(song_id: int, stage: str | None) -> None:
    """Move to the next stage, recording how long the previous one took (shown as estimates)."""
    now = datetime.now(timezone.utc)
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        if song.stage and song.stage_started_at and song.stage != "starting":
            started = song.stage_started_at
            if started.tzinfo is None:          # SQLite returns naive datetimes
                started = started.replace(tzinfo=timezone.utc)
            song.timings = {**(song.timings or {}), song.stage: round((now - started).total_seconds(), 1)}
        song.stage, song.stage_started_at = stage, now if stage else None
        s.commit()


def build_draft(song_id: int) -> None:
    """(Re)build the shared draft chart from cached stages. Users' edited copies are left alone."""
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        song_dir = settings.songs_dir / song.youtube_id
        meta = json.loads((song_dir / "meta.json").read_text())
        chords_path = song_dir / "cache" / "chords.json"
        if not chords_path.exists():
            return
        analysis = json.loads(chords_path.read_text())
        lyrics_path = song_dir / "cache" / "lyrics.json"
        lyrics = json.loads(lyrics_path.read_text()) if lyrics_path.exists() else {"lines": []}
        text = build_chordpro(title=song.title or meta["title"], artist=song.artist or "",
                              analysis=analysis, segments=load_segments(analysis),
                              lyrics=lyrics, duration=meta.get("duration") or 0)
        draft = s.scalar(select(Chart).where(Chart.song_id == song_id, Chart.user_id.is_(None)))
        if draft:
            draft.chordpro = text
        else:
            s.add(Chart(song_id=song_id, user_id=None, chordpro=text))
        s.commit()


@celery.task(name="process_song")
def process_song(song_id: int) -> None:
    # Atomically claim the job: a duplicate or outdated message finds nothing to claim and exits.
    with SessionLocal() as s:
        claimed = s.execute(
            update(Song).where(Song.id == song_id, Song.status == SongStatus.queued)
            .values(status=SongStatus.processing, stage="starting", timings={},
                    started_at=datetime.now(timezone.utc), stage_started_at=datetime.now(timezone.utc))
        ).rowcount
        s.commit()
        if not claimed:
            return
        youtube_id = s.get(Song, song_id).youtube_id
    song_dir = settings.songs_dir / youtube_id
    try:
        _stage(song_id, "download")
        meta = download(youtube_id, song_dir, settings.cookies_file)
        artist, track = guess_artist_track(meta)
        _set(song_id, title=track or meta["title"], duration=meta["duration"], artist=artist or meta.get("uploader"))
        _stage(song_id, "chords")
        wav = decode(song_dir)
        analyze(wav, song_dir / "cache" / "chords.json")
        _stage(song_id, "lyrics")
        lyrics = fetch_lyrics(song_dir, wav, meta)
        if (match := lyrics.get("match")) and " - " not in match["trackName"]:
            _set(song_id, title=match["trackName"], artist=match["artistName"])
        _stage(song_id, "chart")
        build_draft(song_id)
        _stage(song_id, None)
        _set(song_id, status=SongStatus.done)
    except Exception as exc:  # report any failure to the user
        _set(song_id, status=SongStatus.failed, error=f"{type(exc).__name__}: {exc}"[:2000])
        raise


@celery.task(name="rebuild_chart")
def rebuild_chart(song_id: int) -> None:
    build_draft(song_id)
