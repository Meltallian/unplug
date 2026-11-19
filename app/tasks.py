import json

from celery import Celery
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


def _set(song_id: int, **fields) -> None:
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        for k, v in fields.items():
            setattr(song, k, v)
        s.commit()


def build_draft(song_id: int) -> None:
    """(Re)build the shared draft chart from cached stages. Users' edited copies are left alone."""
    with SessionLocal() as s:
        song = s.get(Song, song_id)
        song_dir = settings.songs_dir / song.youtube_id
        meta = json.loads((song_dir / "meta.json").read_text())
        analysis = json.loads((song_dir / "cache" / "chords.json").read_text())
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
        artist, track = guess_artist_track(meta)
        _set(song_id, title=track or meta["title"], duration=meta["duration"], artist=artist or meta.get("uploader"))
        _set(song_id, stage="chords")
        wav = decode(song_dir)
        analyze(wav, song_dir / "cache" / "chords.json")
        _set(song_id, stage="lyrics")
        lyrics = fetch_lyrics(song_dir, wav, meta)
        if (match := lyrics.get("match")) and " - " not in match["trackName"]:
            _set(song_id, title=match["trackName"], artist=match["artistName"])
        _set(song_id, stage="chart")
        build_draft(song_id)
        _set(song_id, status=SongStatus.done, stage=None)
    except Exception as exc:  # report any failure to the user
        _set(song_id, status=SongStatus.failed, error=f"{type(exc).__name__}: {exc}"[:2000])
        raise


@celery.task(name="rebuild_chart")
def rebuild_chart(song_id: int) -> None:
    build_draft(song_id)
