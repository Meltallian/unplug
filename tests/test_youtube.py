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
