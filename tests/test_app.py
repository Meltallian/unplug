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
