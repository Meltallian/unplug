from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user
from app.config import Settings, get_settings
from app.db import SessionLocal, get_session
from app.models import Chart, LibraryEntry, Song, SongStatus
from app.pipeline.chart import parse_chordpro
from app.youtube import parse_youtube_id

HERE = Path(__file__).parent


app = FastAPI(title="unplug", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

PENDING = (SongStatus.queued, SongStatus.processing)
# A processing song with no stage update for this long is considered stuck and may be reprocessed.
# (Queued songs are never "stuck": the worker's atomic claim makes a duplicate message harmless.)
STALE_AFTER = timedelta(minutes=30)


@app.exception_handler(HTTPException)
async def _friendly_errors(request: Request, exc: HTTPException):
    if exc.status_code in (400, 404, 409, 429) and "text/html" in request.headers.get("accept", ""):
        return templates.TemplateResponse(request, "error.html", {"message": exc.detail, "user": None},
                                          status_code=exc.status_code)
    return await http_exception_handler(request, exc)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _send_task(song_id: int) -> None:
    from app.tasks import process_song

    process_song.delay(song_id)


def _enqueue(song_id: int) -> None:
    """Send the job; if the broker is unreachable, mark the song failed so it can be retried."""
    try:
        _send_task(song_id)
    except Exception as exc:
        with SessionLocal() as s:
            s.execute(update(Song).where(Song.id == song_id).values(
                status=SongStatus.failed, error=f"Could not queue job: {type(exc).__name__}"))
            s.commit()


def _queue_position(session: Session, song: Song) -> int | None:
    if song.status != SongStatus.queued:
        return None
    ahead = session.scalar(
        select(func.count()).select_from(Song)
        .where(Song.status == SongStatus.queued, Song.queued_at < song.queued_at)
    )
    return ahead + 1


@app.get("/", response_class=HTMLResponse)
def library(request: Request, q: str = "", user: CurrentUser = Depends(current_user),
            session: Session = Depends(get_session)):
    stmt = (select(LibraryEntry).join(Song).where(LibraryEntry.user_id == user.id)
            .order_by(LibraryEntry.created_at.desc()))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Song.title.ilike(like) | Song.artist.ilike(like))
    entries = session.scalars(stmt).all()
    return templates.TemplateResponse(request, "library.html",
                                      {"entries": entries, "q": q, "user": user, "error": None})


@app.post("/songs")
def add_song(request: Request, link: str = Form(...), user: CurrentUser = Depends(current_user),
             session: Session = Depends(get_session), settings: Settings = Depends(get_settings)):
    youtube_id = parse_youtube_id(link)
    if not youtube_id:
        raise HTTPException(400, "That doesn't look like a YouTube link.")
    song = session.scalar(select(Song).where(Song.youtube_id == youtube_id))
    new = song is None
    if new:
        pending = session.scalar(
            select(func.count()).select_from(Song)
            .where(Song.submitted_by_id == user.id, Song.status.in_(PENDING)))
        if pending >= settings.max_queued_per_user:
            raise HTTPException(429, f"You already have {pending} songs waiting. Try again when one finishes.")
        try:
            song = Song(youtube_id=youtube_id, submitted_by_id=user.id)
            session.add(song)
            session.commit()
        except IntegrityError:  # someone added the same video at the same moment
            session.rollback()
            new = False
            song = session.scalar(select(Song).where(Song.youtube_id == youtube_id))
    try:
        session.add(LibraryEntry(user_id=user.id, song_id=song.id))
        session.commit()
    except IntegrityError:  # already in this user's library
        session.rollback()
    if new:
        _enqueue(song.id)
    return RedirectResponse(f"/songs/{song.id}", status_code=303)


def _song_for_user(session: Session, song_id: int, user: CurrentUser) -> Song:
    song = session.get(Song, song_id)
    in_lib = song and session.scalar(select(LibraryEntry).where(
        LibraryEntry.user_id == user.id, LibraryEntry.song_id == song_id))
    if not song or not (in_lib or user.is_admin):
        raise HTTPException(404)
    return song


def _song_files(song: Song):
    return get_settings().songs_dir / song.youtube_id


def _lyrics_info(song: Song) -> dict:
    import json

    path = _song_files(song) / "cache" / "lyrics.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"source": "none", "lines": []}


def _chart_for(session: Session, song: Song, user: CurrentUser) -> tuple[Chart | None, bool]:
    """The user's own edited copy if any, else the shared draft. Returns (chart, is_own_copy)."""
    own = session.scalar(select(Chart).where(Chart.song_id == song.id, Chart.user_id == user.id))
    if own:
        return own, True
    return session.scalar(select(Chart).where(Chart.song_id == song.id, Chart.user_id.is_(None))), False


@app.get("/songs/{song_id}", response_class=HTMLResponse)
def song_page(request: Request, song_id: int, user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    chart, own = _chart_for(session, song, user)
    if song.status != SongStatus.done or chart is None:
        return templates.TemplateResponse(request, "song.html",
                                          {"song": song, "pos": _queue_position(session, song), "user": user})
    lyrics = _lyrics_info(song)
    return templates.TemplateResponse(request, "chart.html", {
        "song": song, "user": user, "chart": parse_chordpro(chart.chordpro), "own": own,
        "lyrics_source": lyrics.get("source"), "lyrics_synced": lyrics.get("synced"),
        "lyrics_match": lyrics.get("match"),
    })


@app.get("/songs/{song_id}/edit", response_class=HTMLResponse)
def edit_page(request: Request, song_id: int, user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    chart, own = _chart_for(session, song, user)
    if chart is None:
        raise HTTPException(404, "This song has no chart yet.")
    return templates.TemplateResponse(request, "edit.html",
                                      {"song": song, "user": user, "text": chart.chordpro, "own": own})


@app.post("/songs/{song_id}/edit")
def save_edit(song_id: int, chordpro: str = Form(...), user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    """Save the user's own copy. The shared draft stays untouched (it is the 'before' for evaluation)."""
    song = _song_for_user(session, song_id, user)
    if len(chordpro) > 100_000:
        raise HTTPException(400, "That chart is too long.")
    own = session.scalar(select(Chart).where(Chart.song_id == song.id, Chart.user_id == user.id))
    if own:
        own.chordpro = chordpro
    else:
        session.add(Chart(song_id=song.id, user_id=user.id, chordpro=chordpro))
    session.commit()
    return RedirectResponse(f"/songs/{song_id}", status_code=303)


@app.post("/songs/{song_id}/reset")
def reset_edit(song_id: int, user: CurrentUser = Depends(current_user),
               session: Session = Depends(get_session)):
    _song_for_user(session, song_id, user)
    own = session.scalar(select(Chart).where(Chart.song_id == song_id, Chart.user_id == user.id))
    if own:
        session.delete(own)
        session.commit()
    return RedirectResponse(f"/songs/{song_id}", status_code=303)


@app.post("/songs/{song_id}/lyrics")
def paste_lyrics(song_id: int, lyrics: str = Form(...), user: CurrentUser = Depends(current_user),
                 session: Session = Depends(get_session)):
    """Lyrics pasted by a user (when LRCLIB had none). Shared by everyone with this song."""
    from app.pipeline.lyrics import plain_lines, save_lyrics
    from app.tasks import build_draft

    song = _song_for_user(session, song_id, user)
    if song.status != SongStatus.done:
        raise HTTPException(400, "Wait until the song has finished processing.")
    if _lyrics_info(song).get("source") not in ("none", "pasted"):
        raise HTTPException(409, "This song already has lyrics from LRCLIB.")
    if len(lyrics) > 50_000 or not lyrics.strip():
        raise HTTPException(400, "Paste the song's lyrics (one line per sung line).")
    save_lyrics(_song_files(song) / "cache" / "lyrics.json",
                {"source": "pasted", "synced": False, "lines": plain_lines(lyrics), "offset": 0.0})
    build_draft(song.id)
    return RedirectResponse(f"/songs/{song_id}", status_code=303)


@app.get("/songs/{song_id}/status", response_class=HTMLResponse)
def song_status(request: Request, song_id: int, user: CurrentUser = Depends(current_user),
                session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    return templates.TemplateResponse(request, "_status.html",
                                      {"song": song, "pos": _queue_position(session, song)})


@app.post("/songs/{song_id}/reprocess")
def reprocess(song_id: int, user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    _song_for_user(session, song_id, user)
    stale = datetime.now(timezone.utc) - STALE_AFTER
    # Atomic claim: only one click/user wins, so the job is never queued twice.
    claimed = session.execute(
        update(Song)
        .where(Song.id == song_id,
               Song.status.not_in(PENDING)
               | ((Song.status == SongStatus.processing) & (Song.updated_at < stale)))
        .values(status=SongStatus.queued, error=None, stage=None,
                queued_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc))
    ).rowcount
    session.commit()
    if claimed:
        _enqueue(song_id)
    return RedirectResponse(f"/songs/{song_id}", status_code=303)


@app.post("/songs/{song_id}/remove")
def remove(song_id: int, user: CurrentUser = Depends(current_user),
           session: Session = Depends(get_session)):
    """Remove from the user's library. The shared song and its audio stay (admin can purge later)."""
    entry = session.scalar(select(LibraryEntry).where(LibraryEntry.user_id == user.id,
                                                      LibraryEntry.song_id == song_id))
    if entry:
        session.delete(entry)
        session.commit()
    return RedirectResponse("/", status_code=303)
