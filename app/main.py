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
from app.models import LibraryEntry, Song, SongStatus
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
    if exc.status_code in (400, 404, 429) and "text/html" in request.headers.get("accept", ""):
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


@app.get("/songs/{song_id}", response_class=HTMLResponse)
def song_page(request: Request, song_id: int, user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    return templates.TemplateResponse(request, "song.html",
                                      {"song": song, "pos": _queue_position(session, song), "user": user,
                                       "analysis": _analysis(song)})


def _analysis(song: Song) -> dict | None:
    """Detected key, capo and chord shapes for the interim song page (until the chart view exists)."""
    import json

    from app.music import chord_name, transpose

    path = get_settings().songs_dir / song.youtube_id / "cache" / "chords.json"
    if song.status != SongStatus.done or not path.exists():
        return None
    data = json.loads(path.read_text())
    capo = data["capo"]
    shapes = [chord_name(transpose(tuple(s["chord"]), -capo)) if s["chord"] else "N.C." for s in data["segments"]]
    timeline = [{"t": f"{int(s['start'] // 60)}:{int(s['start'] % 60):02d}", "shape": sh}
                for s, sh in zip(data["segments"], shapes)]
    return {"key": data["key"]["name"], "capo": capo, "tempo": round(data["tempo"]),
            "shapes": sorted(set(shapes) - {"N.C."}), "timeline": timeline}


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
