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
    assert client.post(url + "/chord", data={"index": 0, "chord": "Am", "expected": "C"}).status_code == 204
    assert 'data-c="Am"' in client.get(url).text
    assert client.post(url + "/chord", data={"index": 0, "chord": "<script>", "expected": "Am"}).status_code == 400
    assert client.post(url + "/chord", data={"index": 5, "chord": "C", "expected": "C"}).status_code == 409
    # stale tab: thinks chord 0 is still C
    assert client.post(url + "/chord", data={"index": 0, "chord": "G", "expected": "C"}).status_code == 409
    client.post(url + "/reset")
    assert 'data-c="C"' in client.get(url).text      # the shared draft was never touched


def test_pdf_download(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    r = client.get(url + "/pdf?capo=2&key=0")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:4] == b"%PDF"
    assert client.get(url + "/pdf?capo=12").status_code == 400


def test_move_chord_name():
    from app.music import move_chord_name, shape_key
    assert move_chord_name("Fmaj7", 2) == "Gmaj7" and move_chord_name("C/G", -1) == "B/F#"
    assert move_chord_name("N.C.", 3) == "N.C." and shape_key("Bbm7") == "Bbm"


def test_pdf_limits(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models import Chart
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    with SessionLocal() as s:
        c = s.query(Chart).one()
        c.chordpro = "{capo: 12}\n" + "[C]la\n" * 8000
        s.commit()
    assert client.get(url + "/pdf").status_code == 413


def test_remove_deletes_own_copy_but_keeps_song(client, tmp_path, monkeypatch):
    from app.db import SessionLocal
    from app.models import Chart, Song
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    client.post(url + "/chord", data={"index": 0, "chord": "G", "expected": "C"})
    client.post(url + "/remove")
    with SessionLocal() as s:
        assert s.query(Song).count() == 1
        assert s.query(Chart).filter(Chart.user_id.isnot(None)).count() == 0
    assert "No songs yet" in client.get("/").text


def test_delete_for_everyone_is_admin_only(client, tmp_path, monkeypatch):
    from app.config import get_settings
    from app.db import SessionLocal
    from app.models import Song
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    assert client.post(url + "/delete").status_code == 403
    monkeypatch.setattr(get_settings(), "admin_emails", "tester@example.com")
    song_dir = tmp_path / "songs" / "_K6KyS3vxkc"
    assert song_dir.exists()
    assert "All songs (admin)" in client.get("/").text
    client.post(url + "/delete")
    with SessionLocal() as s:
        assert s.query(Song).count() == 0
    assert not song_dir.exists()


def test_delete_keeps_fixture_audio(client, tmp_path, monkeypatch):
    import json
    from app.config import get_settings
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    monkeypatch.setattr(get_settings(), "admin_emails", "tester@example.com")
    (tmp_path / "fixtures" / "x").mkdir(parents=True)
    (tmp_path / "fixtures" / "x" / "source.json").write_text(json.dumps({"youtube_url": "https://youtu.be/_K6KyS3vxkc"}))
    client.post(url + "/delete")
    assert (tmp_path / "songs" / "_K6KyS3vxkc").exists()


def test_progress_shows_stage_and_estimate(client):
    from datetime import datetime, timedelta, timezone
    from app.db import SessionLocal
    from app.models import Song, SongStatus
    url = client.post("/songs", data={"link": "_K6KyS3vxkc"}, follow_redirects=False).headers["location"]
    with SessionLocal() as s:
        song = s.query(Song).one()
        song.status, song.stage, song.duration = SongStatus.processing, "chords", 240
        song.stage_started_at = datetime.now(timezone.utc) - timedelta(seconds=12)
        s.commit()
    text = client.get(url + "/status").text
    assert "Detecting chords" in text and "step 2 of 4" in text and "12 s so far" in text and "left in total" in text


def test_stage_timings_recorded():
    from app import tasks
    from app.db import SessionLocal, init_db
    from app.models import Song, SongStatus
    init_db()
    with SessionLocal() as s:
        song = Song(youtube_id="ttttttttttt", status=SongStatus.processing, stage="starting"); s.add(song); s.commit()
        sid = song.id
    tasks._stage(sid, "download"); tasks._stage(sid, "chords"); tasks._stage(sid, None)
    with SessionLocal() as s:
        song = s.get(Song, sid)
        assert set(song.timings) == {"download", "chords"} and song.stage is None
        s.delete(song); s.commit()


# --- score view -----------------------------------------------------------------------------------

BARRED = ("{title: T}\n{key: C}\n{capo: 0}\n{tempo: 90}\n{time: 4/4}\n"
          "{c: Intro}\n| [C] | % |\n"
          "{c: Verse}\nAnd | [C]hello there | [G]you . . [Am] | |\n"
          "{c: Chorus}\n{time: 6/8}\n{tempo: 60}\n| [F]sing |\n")


def _set_draft(text):
    from app.db import SessionLocal
    from app.models import Chart
    with SessionLocal() as s:
        s.query(Chart).filter(Chart.user_id.is_(None)).one().chordpro = text
        s.commit()


def test_song_page_has_score_and_lyrics_views(client, tmp_path, monkeypatch):
    import re
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    _set_draft(BARRED)
    page = client.get(url).text
    assert 'id="score"' in page and 'data-view="score"' in page and 'data-view="lyrics"' in page
    # sung rows: bars as cells with the pickup in front, never a raw "|" among the words
    assert 'class="line lbars"' in page and 'class="cell pickup"' in page and 'class="line bars"' in page
    assert all("|" not in text for text in re.findall(r'<span class="tx">([^<]*)</span>', page))
    assert ">hello there <" in page and ">And <" in page
    assert '<div class="meter">6/8 · ♩. = 60</div>' in page
    # every chord mark is numbered in text order, as POST /chord counts them
    assert [int(i) for i in re.findall(r'data-i="(\d+)"', page)] == [0, 1, 2, 3, 4]
    assert re.findall(r'data-c="([^"]+)" data-i', page) == ["C", "C", "G", "Am", "F"]


def test_score_fragment_matches_the_page(client, tmp_path, monkeypatch):
    import re
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    _set_draft(BARRED)
    r = client.get(url + "/score?width=400")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.text.startswith('<div class="score-page"><svg') and "score-note" not in r.text
    page = client.get(url).text
    assert sorted(int(i) for i in re.findall(r'id="chord-(\d+)"', r.text)) == [int(i) for i in re.findall(r'data-i="(\d+)"', page)]
    # capo and key as on screen; a silly width is clamped, not refused
    assert ">D<" in client.get(url + "/score?width=400&key=2").text
    assert ">A<" in client.get(url + "/score?width=400&capo=3").text and "Capo 3" in client.get(url + "/score?capo=3").text
    assert client.get(url + "/score?width=1").status_code == 200 and client.get(url + "/score?width=100000").status_code == 200
    assert client.get(url + "/score?capo=12").status_code == 400 and client.get(url + "/score?key=7").status_code == 400
    assert client.get("/songs/999/score").status_code == 404


def test_score_follows_a_tapped_chord_change(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)
    _set_draft(BARRED)
    assert client.post(url + "/chord", data={"index": 3, "chord": "Dm7", "expected": "Am"}).status_code == 204
    svg = client.get(url + "/score?width=400").text
    assert ">Dm7<" in svg and ">Am<" not in svg
    assert 'data-c="Dm7" data-i="3"' in client.get(url).text


def test_score_of_a_chart_without_bars(client, tmp_path, monkeypatch):
    url = _done_song_with_draft(client, tmp_path, monkeypatch)       # the old draft: "[C]hello", no bar lines
    r = client.get(url + "/score")
    assert r.status_code == 200 and "<svg" in r.text and 'id="chord-0"' in r.text
    assert "score-note" in r.text and "Reprocess" in r.text
    client.post(url + "/edit", data={"chordpro": "{c: Verse}\n[G]hello\n"})                 # own copy, still no bars
    assert "Add | bar lines in Edit" in client.get(url + "/score").text
    client.post(url + "/edit", data={"chordpro": "{title: nothing to play}\n"})
    assert "no chords to engrave" in client.get(url + "/score").text


def test_score_needs_a_chart(client):
    url = client.post("/songs", data={"link": "_K6KyS3vxkc"}, follow_redirects=False).headers["location"]
    assert client.get(url + "/score").status_code == 404


def test_score_font_is_served(client):
    r = client.get("/score-font.woff2")
    assert r.status_code == 200 and r.headers["content-type"] == "font/woff2" and r.content[:4] == b"wOF2"
    assert "max-age" in r.headers["cache-control"]


def test_pdf_template_shows_bar_rows():
    # the HTML that goes to WeasyPrint (the PDF itself needs libpango, see test_pdf_download)
    import re
    from app import main
    from app.pipeline.chart import parse_chordpro
    html = main.templates.get_template("pdf.html").render(
        title="T", artist="", key="C", capo=0, tempo="90", time="4/4", diagrams=[],
        sections=parse_chordpro(BARRED + "[Am]an old [F]line\n")["sections"])
    assert 'class="line lbars"' in html and 'class="cell pickup"' in html and 'class="line bars"' in html
    assert '<span class="meter">6/8 · ♩. = 60</span>' in html
    assert all("|" not in text for text in re.findall(r'<span class="tx">([^<]*)</span>', html))
    assert ">hello there <" in html and '<b class="ch rep">·</b>' in html and ">an old <" in html
