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


def _done_song_with_draft(client, tmp_path, monkeypatch):
    import json

    from app.config import get_settings
    from app.db import SessionLocal
    from app.models import Chart, Song, SongStatus
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path)
    url = client.post("/songs", data={"link": "_K6KyS3vxkc"}, follow_redirects=False).headers["location"]
    song_dir = tmp_path / "songs" / "_K6KyS3vxkc" / "cache"
    song_dir.mkdir(parents=True)
    (song_dir.parent / "meta.json").write_text(json.dumps({"title": "T", "duration": 10, "audio_file": "a.webm"}))
    (song_dir / "chords.json").write_text(json.dumps({"capo": 0, "key": {"name": "C"}, "tempo": 90,
        "segments": [{"chord": [0, "maj"], "start": 0, "end": 10, "beats": 8, "name": "C"}]}))
    with SessionLocal() as s:
        song = s.query(Song).one()
        song.status = SongStatus.done
        s.add(Chart(song_id=song.id, user_id=None, chordpro="{title: T}\n{key: C}\n{c: Verse}\n[C]hello\n"))
        s.commit()
    return url


def test_chart_view_edit_and_reset(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    page = client.get(url)
    assert page.status_code == 200 and 'data-c="C"' in page.text and "No lyrics found" in page.text
    client.post(url + "/edit", data={"chordpro": "{c: Verse}\n[G]hello\n"})
    assert 'data-c="G"' in client.get(url).text
    client.post(url + "/reset")
    assert 'data-c="C"' in client.get(url).text


def test_paste_lyrics_rebuilds_draft(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    client.post(url + "/lyrics", data={"lyrics": "first line\nsecond line"})
    page = client.get(url).text
    assert "first line" in page and "No lyrics found" not in page


def test_paste_refused_when_lrclib_lyrics_exist(client, tmp_path, monkeypatch):
    import json
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    (tmp_path / "songs" / "_K6KyS3vxkc" / "cache" / "lyrics.json").write_text(
        json.dumps({"source": "lrclib", "synced": True, "lines": [{"t": 1.0, "text": "real"}]}))
    assert client.post(url + "/lyrics", data={"lyrics": "vandal"}).status_code == 409


def test_tap_edit_changes_one_chord_in_own_copy(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    page = client.get(url).text
    assert 'data-i="0"' in page
    assert client.post(url + "/chord", data={"index": 0, "chord": "Am"}).status_code == 204
    assert 'data-c="Am"' in client.get(url).text
    assert client.post(url + "/chord", data={"index": 0, "chord": "<script>"}).status_code == 400
    assert client.post(url + "/chord", data={"index": 5, "chord": "C"}).status_code == 409
    client.post(url + "/reset")
    assert 'data-c="C"' in client.get(url).text      # the shared draft was never touched
