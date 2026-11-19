from celery import Celery

from app.config import get_settings
from sqlalchemy import update

from app.db import SessionLocal
from app.models import Song, SongStatus
from app.pipeline.download import download

settings = get_settings()
celery = Celery("unplug", broker=settings.redis_url, backend=None)
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True)


def _set(song_id: int, **fields) -> None:
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        for k, v in fields.items():
            setattr(song, k, v)
        s.commit()


@celery.task(name="process_song")
def process_song(song_id: int) -> None:
    # Atomically claim the job: a duplicate or outdated message finds nothing to claim and exits.
    with SessionLocal() as s:
        claimed = s.execute(
            update(Song).where(Song.id == song_id, Song.status == SongStatus.queued)
            .values(status=SongStatus.processing, stage="starting")
        ).rowcount
        s.commit()
        if not claimed:
            return
        youtube_id = s.get(Song, song_id).youtube_id
    song_dir = settings.songs_dir / youtube_id
    try:
        _set(song_id, stage="download", error=None)
        meta = download(youtube_id, song_dir, settings.cookies_file)
        _set(song_id, title=meta["title"], duration=meta["duration"],
             artist=meta.get("artist") or meta.get("uploader"))
        # Later stages (tuning, beats, chords, lyrics, alignment, simplify) plug in here.
        _set(song_id, status=SongStatus.done, stage=None)
    except Exception as exc:  # report any failure to the user
        _set(song_id, status=SongStatus.failed, error=f"{type(exc).__name__}: {exc}"[:2000])
        raise
