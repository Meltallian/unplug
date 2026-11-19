def test_task_claims_job_only_once(client, monkeypatch):
    from app import tasks
    from app.db import SessionLocal
    from app.models import Song

    calls = []
    monkeypatch.setattr(tasks, "download", lambda yid, d, c=None: calls.append(yid) or
                        {"title": "T", "duration": 1, "artist": "A", "uploader": "U"})
    client.post("/songs", data={"link": "_K6KyS3vxkc"})
    with SessionLocal() as s:
        song_id = s.query(Song).one().id
    tasks.process_song(song_id)
    tasks.process_song(song_id)  # duplicate message: already done, must not run again
    assert calls == ["_K6KyS3vxkc"]
    with SessionLocal() as s:
        assert s.get(Song, song_id).status.value == "done"
