def test_add_song_enqueues_once_and_shares(client):
    r = client.post("/songs", data={"link": "https://youtu.be/_K6KyS3vxkc"}, follow_redirects=False)
    assert r.status_code == 303
    assert len(client.enqueued) == 1
    # adding the same video again does not reprocess it
    client.post("/songs", data={"link": "https://www.youtube.com/watch?v=_K6KyS3vxkc"})
    assert len(client.enqueued) == 1
    page = client.get("/")
    assert page.status_code == 200
    assert "queued" in page.text


def test_rejects_bad_link(client):
    assert client.post("/songs", data={"link": "nope"}).status_code == 400


def test_queue_limit(client, monkeypatch):
    for i in range(5):
        client.post("/songs", data={"link": f"https://youtu.be/aaaaaaaaaa{i}"})
    r = client.post("/songs", data={"link": "https://youtu.be/bbbbbbbbbbb"})
    assert r.status_code == 429


def test_status_and_remove(client):
    r = client.post("/songs", data={"link": "_K6KyS3vxkc"}, follow_redirects=False)
    url = r.headers["location"]
    assert "position 1" in client.get(url + "/status").text
    client.post(url + "/remove")
    assert client.get(url).status_code == 404


def test_requires_access_token_without_dev_user(client, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "dev_user_email", "")
    assert client.get("/").status_code == 401


def _song(url_or_resp):
    from app.db import SessionLocal
    from app.models import Song
    with SessionLocal() as s:
        return s.query(Song).filter_by(youtube_id=url_or_resp).one()


def test_enqueue_failure_marks_song_failed(client, monkeypatch):
    from app import main

    def boom(_):
        raise ConnectionError("redis down")
    monkeypatch.setattr(main, "_send_task", boom)
    client.post("/songs", data={"link": "_K6KyS3vxkc"})
    song = _song("_K6KyS3vxkc")
    assert song.status.value == "failed" and "Could not queue" in song.error


def test_reprocess_only_once_while_pending(client):
    r = client.post("/songs", data={"link": "_K6KyS3vxkc"}, follow_redirects=False)
    url = r.headers["location"]
    client.post(url + "/reprocess")          # still queued: ignored
    assert len(client.enqueued) == 1
    from app.db import SessionLocal
    from app.models import Song, SongStatus
    with SessionLocal() as s:
        s.query(Song).update({"status": SongStatus.failed}); s.commit()
    client.post(url + "/reprocess")
    client.post(url + "/reprocess")          # second click while re-queued: ignored
    assert len(client.enqueued) == 2


def test_reprocessed_song_goes_to_back_of_queue(client):
    from app.db import SessionLocal
    from app.models import Song, SongStatus
    first = client.post("/songs", data={"link": "aaaaaaaaaaa"}, follow_redirects=False).headers["location"]
    client.post("/songs", data={"link": "bbbbbbbbbbb"})
    with SessionLocal() as s:
        s.query(Song).filter_by(youtube_id="aaaaaaaaaaa").update({"status": SongStatus.failed}); s.commit()
    client.post(first + "/reprocess")
    assert "position 2" in client.get(first + "/status").text


def test_html_error_page(client):
    r = client.post("/songs", data={"link": "nope"}, headers={"accept": "text/html"})
    assert r.status_code == 400 and "YouTube link" in r.text and "<html" in r.text


def test_dev_user_refused_in_production():
    import pytest
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(dev_user_email="x@example.com", cf_access_aud="abc")
