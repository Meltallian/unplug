def test_task_claims_job_only_once(client, monkeypatch):
    from app import tasks
    from app.db import SessionLocal
    from app.models import Song

    calls = []
    monkeypatch.setattr(tasks, "download", lambda yid, d, c=None: calls.append(yid) or
                        {"title": "T", "duration": 1, "artist": "A", "uploader": "U"})
    monkeypatch.setattr(tasks, "decode", lambda d: d / "audio.wav")
    monkeypatch.setattr(tasks, "analyze", lambda wav, out: {})
    monkeypatch.setattr(tasks, "fetch_lyrics", lambda d, wav, meta: {"lines": []})
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
