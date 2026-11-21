def test_task_claims_job_only_once(client, monkeypatch):
    from app import tasks
    from app.db import SessionLocal
    from app.models import Song

    calls = []
    monkeypatch.setattr(tasks, "download", lambda yid, d, c=None: calls.append(yid) or
                        {"title": "T", "duration": 1, "artist": "A", "uploader": "U"})
    monkeypatch.setattr(tasks, "decode", lambda d: d / "audio.wav")
    monkeypatch.setattr(tasks, "analyze", lambda wav, out: {})
    monkeypatch.setattr(tasks, "fetch_lyrics", lambda d, wav, meta, analysis=None, cookies=None: {"lines": []})
    monkeypatch.setattr(tasks, "build_draft", lambda song_id: None)
    client.post("/songs", data={"link": "_K6KyS3vxkc"})
    with SessionLocal() as s:
        song_id = s.query(Song).one().id
    tasks.process_song(song_id)
    tasks.process_song(song_id)  # duplicate message: already done, must not run again
    assert calls == ["_K6KyS3vxkc"]
    with SessionLocal() as s:
        assert s.get(Song, song_id).status.value == "done"


def test_tests_do_not_use_real_data():
    from app.config import get_settings
    assert "unplug-test-" in str(get_settings().data_dir)


def test_interrupted_songs_are_requeued(client, monkeypatch):
    from app import tasks
    from app.db import SessionLocal
    from app.models import Song, SongStatus
    sent = []
    monkeypatch.setattr(tasks.process_song, "delay", sent.append)
    client.post("/songs", data={"link": "_K6KyS3vxkc"})
    with SessionLocal() as s:
        song = s.query(Song).one()
        song.status, song.stage = SongStatus.processing, "chords"
        s.commit()
        sid = song.id
    assert tasks.recover_interrupted() == [sid] and sent == [sid]
    with SessionLocal() as s:
        assert s.get(Song, sid).status == SongStatus.queued
