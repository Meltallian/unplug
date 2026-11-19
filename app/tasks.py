from celery import Celery

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Song, SongStatus
from app.pipeline.download import download

settings = get_settings()
celery = Celery("unplug", broker=settings.redis_url, backend=None)
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, broker_connection_retry_on_startup=True)


@celery.on_after_configure.connect
def _setup(**_):
    init_db()


def _set(song_id: int, **fields) -> None:
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        for k, v in fields.items():
            setattr(song, k, v)
        s.commit()


@celery.task(name="process_song")
def process_song(song_id: int) -> None:
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        if song is None:
            return
        youtube_id = song.youtube_id
    song_dir = settings.songs_dir / youtube_id
    try:
        _set(song_id, status=SongStatus.processing, stage="download", error=None)
        meta = download(youtube_id, song_dir)
        _set(song_id, title=meta["title"], duration=meta["duration"],
             artist=meta.get("artist") or meta.get("uploader"))
        # Later stages (tuning, beats, chords, lyrics, alignment, simplify) plug in here.
        _set(song_id, status=SongStatus.done, stage=None)
    except Exception as exc:  # report any failure to the user
        _set(song_id, status=SongStatus.failed, error=f"{type(exc).__name__}: {exc}"[:2000])
        raise
