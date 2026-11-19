from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import CurrentUser, current_user
from app.config import Settings, get_settings
from app.db import get_session, init_db
from app.models import LibraryEntry, Song, SongStatus
from app.youtube import parse_youtube_id

HERE = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app):
    init_db()
    yield


app = FastAPI(title="unplug", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")

PENDING = (SongStatus.queued, SongStatus.processing)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def _enqueue(song_id: int) -> None:
    from app.tasks import process_song

    process_song.delay(song_id)


def _queue_position(session: Session, song: Song) -> int | None:
    if song.status != SongStatus.queued:
        return None
    ahead = session.scalar(
        select(func.count()).select_from(Song)
        .where(Song.status.in_(PENDING), Song.created_at < song.created_at)
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
        song = Song(youtube_id=youtube_id, submitted_by_id=user.id)
        session.add(song)
        session.flush()
    if not session.scalar(select(LibraryEntry).where(LibraryEntry.user_id == user.id,
                                                     LibraryEntry.song_id == song.id)):
        session.add(LibraryEntry(user_id=user.id, song_id=song.id))
    session.commit()
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
                                      {"song": song, "pos": _queue_position(session, song), "user": user})


@app.get("/songs/{song_id}/status", response_class=HTMLResponse)
def song_status(request: Request, song_id: int, user: CurrentUser = Depends(current_user),
                session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    return templates.TemplateResponse(request, "_status.html",
                                      {"song": song, "pos": _queue_position(session, song)})


@app.post("/songs/{song_id}/reprocess")
def reprocess(song_id: int, user: CurrentUser = Depends(current_user),
              session: Session = Depends(get_session)):
    song = _song_for_user(session, song_id, user)
    if song.status not in PENDING:
        song.status, song.error, song.stage = SongStatus.queued, None, None
        session.commit()
        _enqueue(song.id)
    return RedirectResponse(f"/songs/{song.id}", status_code=303)


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
