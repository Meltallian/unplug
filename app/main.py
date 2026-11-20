import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app import progress
from app.auth import CurrentUser, current_user
from app.config import Settings, get_settings
from app.db import SessionLocal, get_session
from app.models import Chart, LibraryEntry, Song, SongStatus
from app.pipeline.chart import bar_cells, bar_row, line_kind, parse_chordpro
from app.youtube import parse_youtube_id

HERE = Path(__file__).parent


app = FastAPI(title="unplug", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["line_kind"] = line_kind
templates.env.filters["bar_cells"] = bar_cells
templates.env.filters["bar_row"] = bar_row

PENDING = (SongStatus.queued, SongStatus.processing)
# A processing song with no stage update for this long is considered stuck and may be reprocessed.
# (Queued songs are never "stuck": the worker's atomic claim makes a duplicate message harmless.)
STALE_AFTER = timedelta(minutes=30)


@app.exception_handler(HTTPException)
async def _friendly_errors(request: Request, exc: HTTPException):
    if exc.status_code in (400, 404, 409, 413, 429, 503) and "text/html" in request.headers.get("accept", ""):
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


def _status_info(session: Session, song: Song) -> dict:
    pos = _queue_position(session, song)
    return {"pos": pos, "wait": progress.queue_wait(session, pos) if pos else None,
            "prog": progress.progress(session, song), "fmt": progress.fmt}


def _queue_position(session: Session, song: Song) -> int | None:
    if song.status != SongStatus.queued:
        return None
    ahead = session.scalar(
        select(func.count()).select_from(Song)
        .where(Song.status == SongStatus.queued, Song.queued_at < song.queued_at)
    )
    return ahead + 1


@app.get("/", response_class=HTMLResponse)
def library(request: Request, q: str = "", all: int = 0, user: CurrentUser = Depends(current_user),
            session: Session = Depends(get_session)):
    show_all = bool(all) and user.is_admin
    if show_all:   # admin: every song on the server, also those in nobody's library
        stmt = select(Song).order_by(Song.created_at.desc())
    else:
        stmt = (select(Song).join(LibraryEntry).where(LibraryEntry.user_id == user.id)
                .order_by(LibraryEntry.created_at.desc()))
    if q:
        like = f"%{q}%"
        stmt = stmt.where(Song.title.ilike(like) | Song.artist.ilike(like))
    songs = session.scalars(stmt).all()
    return templates.TemplateResponse(request, "library.html",
                                      {"songs": songs, "q": q, "user": user, "show_all": show_all})


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
        return templates.TemplateResponse(request, "song.html", {"song": song, "user": user,
                                                                 **_status_info(session, song)})
    lyrics = _lyrics_info(song)
    return templates.TemplateResponse(request, "chart.html", {
        "song": song, "user": user, "chart": parse_chordpro(chart.chordpro), "own": own,
        "lyrics_source": lyrics.get("source"), "lyrics_synced": lyrics.get("synced"),
        "lyrics_match": lyrics.get("match"),
    })


@app.get("/songs/{song_id}/score", response_class=HTMLResponse)
def song_score(song_id: int, width: int = 400, capo: int | None = None, key: int = 0,
               user: CurrentUser = Depends(current_user), session: Session = Depends(get_session)):
    """The engraved score of the chart as an HTML fragment (inline SVG), laid out for a screen
    `width` CSS pixels wide, with the capo / transposition currently chosen on screen."""
    from html import escape

    from app.pipeline.score import TooLong, engrave

    song = _song_for_user(session, song_id, user)
    chart, own = _chart_for(session, song, user)
    if chart is None:
        raise HTTPException(404, "This song has no chart yet.")
    if not ((capo is None or 0 <= capo <= 9) and -6 <= key <= 6):
        raise HTTPException(400, "Capo must be 0–9 and transposition −6…+6.")
    try:
        engraving = engrave(chart.chordpro, width=width, capo=capo, key=key)
    except TooLong as exc:
        raise HTTPException(413, str(exc))
    if not engraving.pages:
        return HTMLResponse('<p class="muted">This chart has no chords to engrave yet.</p>')
    note = ""
    if engraving.unbarred:
        fix = ("Add | bar lines in Edit, or go back to the automatic chart." if own
               else "Reprocess the song to get real bars.")
        note = ('<p class="muted small score-note">Lines without bar lines are drawn with one bar '
                f"per chord. {escape(fix)}</p>")
    return HTMLResponse(note + "".join(f'<div class="score-page">{svg}</div>' for svg in engraving.pages))


@app.get("/score-font.woff2")
def score_font():
    """The music font used by the text of the engraved score (tempo notes, sharps and flats)."""
    from fastapi.responses import Response

    from app.pipeline.score import music_font

    return Response(music_font(), media_type="font/woff2", headers={"Cache-Control": "public, max-age=604800"})


# PDF rendering is CPU and memory heavy: one at a time, and only for charts of sensible size.
_PDF_LOCK = threading.Lock()
PDF_MAX_CHARS = 30_000


def _no_fetch(url: str, *args, **kwargs):
    raise ValueError(f"PDF rendering never fetches resources ({url[:40]})")


@app.get("/songs/{song_id}/pdf")
def chart_pdf(request: Request, song_id: int, capo: int | None = None, key: int = 0,
              user: CurrentUser = Depends(current_user), session: Session = Depends(get_session)):
    """The chart as a PDF, with the capo / transposition currently chosen on screen."""
    from fastapi.responses import Response
    from weasyprint import HTML

    from app.music import diagram_svg, move_chord_name, shape_key

    song = _song_for_user(session, song_id, user)
    chart, _own = _chart_for(session, song, user)
    if chart is None:
        raise HTTPException(404, "This song has no chart yet.")
    if len(chart.chordpro) > PDF_MAX_CHARS:
        raise HTTPException(413, "This chart is too long to print.")
    parsed = parse_chordpro(chart.chordpro)
    meta = parsed["meta"]
    m = re.match(r"\s*(\d+)", meta.get("capo", "0"))
    chart_capo = min(9, int(m.group(1))) if m else 0   # same reading as the page (parseInt)
    capo = chart_capo if capo is None else capo
    if not (0 <= capo <= 9 and -6 <= key <= 6):
        raise HTTPException(400, "Capo must be 0–9 and transposition −6…+6.")
    semis = chart_capo + key - capo

    used: list[str] = []
    sections = []
    for section in parsed["sections"]:
        lines = []
        for line in section["lines"]:
            segs = []
            for chord, text in line:
                name = move_chord_name(chord, semis) if chord else None
                if name and (k := shape_key(name)) and k not in used:
                    used.append(k)
                segs.append((name, text))
            lines.append(segs)
        sections.append({"name": section["name"], "lines": lines, "changes": section["changes"]})
    html = templates.get_template("pdf.html").render(
        title=meta.get("title") or song.title, artist=meta.get("artist") or song.artist or "",
        key=move_chord_name(meta["key"], key) if meta.get("key") else "", capo=capo,
        tempo=meta.get("tempo", ""), time=meta.get("time", ""), sections=sections,
        diagrams=[(k, diagram_svg(k)) for k in used])
    if not _PDF_LOCK.acquire(timeout=30):
        raise HTTPException(503, "Busy printing another chart; try again in a moment.")
    try:
        pdf = HTML(string=html, url_fetcher=_no_fetch).write_pdf()
    finally:
        _PDF_LOCK.release()
    title = f"{meta.get('title') or song.title or 'chart'} - chords"
    ascii_name = re.sub(r"[^A-Za-z0-9 ._-]", "", title).strip() or "chart"
    disposition = f'attachment; filename="{ascii_name}.pdf"; filename*=UTF-8\'\'{quote(title + ".pdf")}'
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": disposition})


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


@app.post("/songs/{song_id}/chord")
def change_chord(song_id: int, index: int = Form(...), chord: str = Form(""), expected: str = Form(...),
                 user: CurrentUser = Depends(current_user), session: Session = Depends(get_session)):
    """Tap-to-edit: change or remove one chord. Works on the user's own copy (created on first edit)."""
    from fastapi.responses import Response

    from app.pipeline.chart import CHORD_SYMBOL, replace_chord

    song = _song_for_user(session, song_id, user)
    chart, own = _chart_for(session, song, user)
    if chart is None:
        raise HTTPException(404)
    chord = chord.strip()
    if chord and not CHORD_SYMBOL.fullmatch(chord):
        raise HTTPException(400, "That doesn't look like a chord.")
    try:
        text = replace_chord(chart.chordpro, index, chord or None, expected=expected.strip())
    except (IndexError, ValueError):
        raise HTTPException(409, "The chart changed; reload the page.")
    if own:
        chart.chordpro = text
    else:
        session.add(Chart(song_id=song.id, user_id=user.id, chordpro=text))
    try:
        session.commit()
    except IntegrityError:  # two first edits at once: the other one won
        session.rollback()
        raise HTTPException(409, "The chart changed; reload the page.")
    return Response(status_code=204)


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
    return templates.TemplateResponse(request, "_status.html", {"song": song, **_status_info(session, song)})


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
    """Remove from the user's library, with the user's own edited copy. The shared song stays."""
    session.execute(delete(LibraryEntry).where(LibraryEntry.user_id == user.id, LibraryEntry.song_id == song_id))
    session.execute(delete(Chart).where(Chart.user_id == user.id, Chart.song_id == song_id))
    session.commit()
    return RedirectResponse("/", status_code=303)


def _fixture_ids() -> set[str]:
    """YouTube ids used by evaluation fixtures: their audio is kept even when a song is deleted."""
    import json

    from app.youtube import parse_youtube_id

    ids = set()
    for src in (get_settings().data_dir / "fixtures").glob("*/source.json"):
        try:
            ids.add(parse_youtube_id(json.loads(src.read_text())["youtube_url"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return ids


@app.post("/songs/{song_id}/delete")
def delete_song(song_id: int, user: CurrentUser = Depends(current_user),
                session: Session = Depends(get_session)):
    """Admin only: delete a song for everyone (library entries, all chart copies, audio)."""
    import shutil

    if not user.is_admin:
        raise HTTPException(403, "Only the admin can delete a song for everyone.")
    song = session.get(Song, song_id)
    if song is None:
        raise HTTPException(404)
    if song.status in PENDING and song.updated_at > datetime.now(timezone.utc) - STALE_AFTER:
        raise HTTPException(409, "This song is being processed; try again when it has finished.")
    youtube_id = song.youtube_id
    keep_audio = youtube_id in _fixture_ids()   # decided before anything is deleted
    session.execute(delete(LibraryEntry).where(LibraryEntry.song_id == song_id))
    session.execute(delete(Chart).where(Chart.song_id == song_id))
    session.delete(song)
    session.commit()
    if not keep_audio:
        shutil.rmtree(get_settings().songs_dir / youtube_id, ignore_errors=True)
    return RedirectResponse("/", status_code=303)
