import enum
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


class SongStatus(str, enum.Enum):
    queued = "queued"
    processing = "processing"
    done = "done"
    failed = "failed"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Song(Base):
    """One processed YouTube video, shared by every user who added it."""

    __tablename__ = "songs"
    id: Mapped[int] = mapped_column(primary_key=True)
    youtube_id: Mapped[str] = mapped_column(String(16), unique=True)
    title: Mapped[str | None] = mapped_column(String(300))
    artist: Mapped[str | None] = mapped_column(String(300))
    duration: Mapped[float | None]
    status: Mapped[SongStatus] = mapped_column(Enum(SongStatus), default=SongStatus.queued)
    stage: Mapped[str | None] = mapped_column(String(50))
    error: Mapped[str | None] = mapped_column(Text)
    submitted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class LibraryEntry(Base):
    __tablename__ = "library_entries"
    __table_args__ = (UniqueConstraint("user_id", "song_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    song: Mapped[Song] = relationship()


class Chart(Base):
    """ChordPro text. user_id NULL = the shared automatic draft; otherwise a user's edited copy."""

    __tablename__ = "charts"
    __table_args__ = (UniqueConstraint("song_id", "user_id"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    song_id: Mapped[int] = mapped_column(ForeignKey("songs.id", ondelete="CASCADE"))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    chordpro: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
