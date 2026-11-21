import pytest

from app.youtube import parse_youtube_id


@pytest.mark.parametrize("link", [
    "https://www.youtube.com/watch?v=_K6KyS3vxkc",
    "https://youtube.com/watch?feature=share&v=_K6KyS3vxkc",
    "https://youtu.be/_K6KyS3vxkc?si=abc",
    "https://www.youtube.com/shorts/_K6KyS3vxkc",
    "https://m.youtube.com/watch?v=_K6KyS3vxkc&t=30s",
    "_K6KyS3vxkc",
])
def test_parses(link):
    assert parse_youtube_id(link) == "_K6KyS3vxkc"


@pytest.mark.parametrize("bad", ["", "hello", "https://vimeo.com/123456", "https://youtube.com/watch?v=short"])
def test_rejects(bad):
    assert parse_youtube_id(bad) is None


def test_download_is_tried_again_after_a_refusal(tmp_path, monkeypatch):
    import yt_dlp

    from app.pipeline import download as dl

    calls = []

    class Flaky:
        def __init__(self, opts): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, url, download):
            calls.append(url)
            if len(calls) < 3:
                raise yt_dlp.utils.DownloadError("HTTP Error 403: Forbidden")
            (tmp_path / "audio.webm").write_bytes(b"x")
            return {"title": "T", "duration": 3, "requested_downloads": [{"filepath": str(tmp_path / "audio.webm")}]}

    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", Flaky)
    monkeypatch.setattr(dl, "RETRY_WAIT", 0.0)
    assert dl.download("abcdefghijk", tmp_path)["title"] == "T" and len(calls) == 3
    calls.clear()
    (tmp_path / "meta.json").unlink()
    monkeypatch.setattr(dl, "RETRIES", 1)
    import pytest
    with pytest.raises(yt_dlp.utils.DownloadError):
        dl.download("abcdefghijk", tmp_path)
