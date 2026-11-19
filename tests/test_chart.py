import numpy as np

from app.pipeline.chart import build_chordpro, parse_chordpro
from app.pipeline.lyrics import choose_match, estimate_offset, guess_artist_track, parse_lrc

ANALYSIS = {"capo": 0, "key": {"name": "C"}, "tempo": 90.0}
C, AM, F, G = (0, "maj"), (9, "min"), (5, "maj"), (7, "maj")


def test_guess_artist_track():
    assert guess_artist_track({"title": "Paleface Swiss - River Of Sorrows (unplugged) feat.  Ron Deris"}) == \
        ("Paleface Swiss", "River Of Sorrows (unplugged)")
    assert guess_artist_track({"title": "Green Day - Boulevard Of Broken Dreams [Official Music Video] [4K Upgrade]"}) == \
        ("Green Day", "Boulevard Of Broken Dreams")
    assert guess_artist_track({"title": 'Architects - "Doomsday"'}) == ("Architects", "Doomsday")
    assert guess_artist_track({"title": "Woodkid – To The Wilder (from DEATH STRANDING 2: ON THE BEACH) | Lyric Video"})[0] == "Woodkid"


def test_choose_match_prefers_same_variant_and_synced():
    results = [
        {"artistName": "Paleface Swiss", "trackName": "River Of Sorrows", "duration": 243, "syncedLyrics": "x"},
        {"artistName": "Paleface Swiss", "trackName": "River Of Sorrows (Unplugged)", "duration": 267, "syncedLyrics": "x"},
        {"artistName": "Someone Else", "trackName": "River Of Sorrows (Unplugged)", "duration": 284, "syncedLyrics": "x"},
    ]
    m = choose_match(results, "Paleface Swiss", "River Of Sorrows (unplugged)", 284)
    assert m["duration"] == 267


def test_parse_lrc():
    assert parse_lrc("[00:12.50] Hello\n[01:02.00]\n") == [{"t": 12.5, "text": "Hello"}, {"t": 62.0, "text": ""}]


def test_estimate_offset_finds_shift():
    hop = 0.1
    act = -np.ones(600)                       # 60 s of "no voice"
    for start in (20, 30, 40):                # voice actually starts at 20, 30, 40 s
        act[int(start / hop):int((start + 3) / hop)] = 1.0
    lines = [{"t": t, "text": "la"} for t in (10, 20, 30)]   # lyrics file says 10, 20, 30
    assert abs(estimate_offset(lines, act, hop, 60, 50) - 10) < 0.5


def test_chart_places_chords_and_names_chorus():
    segs = [(AM, 0, 4, 4), (C, 4, 8, 4), (F, 8, 10, 4), (G, 10, 12, 4), (AM, 12, 16, 4), (C, 16, 20, 4),
            (F, 20, 24, 4), (G, 24, 28, 4)]
    lyrics = {"offset": 0.0, "lines": [
        {"t": 4.0, "text": "Verse line one"}, {"t": 8.0, "text": "Hook line here"}, {"t": 10.0, "text": "hook again"},
        {"t": 12.0, "text": ""},
        {"t": 16.0, "text": "Hook line here"}, {"t": 18.0, "text": "hook again"},
    ]}
    text = build_chordpro(title="T", artist="A", analysis=ANALYSIS, segments=segs, lyrics=lyrics, duration=28)
    assert "{c: Intro}" in text and "[Am]" in text
    assert "[C]Verse line one" in text
    parsed = parse_chordpro(text)
    names = [s["name"] for s in parsed["sections"]]
    assert names[0] == "Intro"
    assert parsed["meta"]["key"] == "C"


def test_chart_uses_capo_shapes():
    analysis = {**ANALYSIS, "capo": 2}
    segs = [((4, "min"), 0, 8, 8)]   # sounding Em → shape Dm with capo 2
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics={"lines": []}, duration=8)
    assert "[Dm]" in text and "{capo: 2}" in text


def test_short_chords_are_merged():
    segs = [(C, 0, 4, 4), (G, 4, 4.5, 1), (C, 4.5, 8, 4)]
    text = build_chordpro(title="T", artist="", analysis=ANALYSIS, segments=segs, lyrics={"lines": []}, duration=8)
    assert "[G]" not in text


def test_parse_chordpro_inline():
    p = parse_chordpro("{c: Verse}\n[Am]Is it so [C]perfect now\nno chords here\n[F]  [G]\n")
    line = p["sections"][0]["lines"][0]
    assert line == [("Am", "Is it so "), ("C", "perfect now")]
    assert p["sections"][0]["lines"][1] == [(None, "no chords here")]
    assert p["sections"][0]["lines"][2] == [("F", "  "), ("G", "")]


def test_guess_more_title_shapes():
    assert guess_artist_track({"title": "The Devil Wears Prada- Chemical (Official Music Video)"}) == \
        ("The Devil Wears Prada", "Chemical")
    assert guess_artist_track({"title": "Woodkid – To The Wilder (from DEATH STRANDING 2: ON THE BEACH) | Lyric Video"}) == \
        ("Woodkid", "To The Wilder (from DEATH STRANDING 2: ON THE BEACH)")


def test_no_offset_when_same_length():
    act = np.zeros(600)
    assert estimate_offset([{"t": 10, "text": "la"}], act, 0.1, 60, 59) == 0.0


def test_carried_chord_not_repeated():
    segs = [(AM, 0, 20, 16)]
    lyrics = {"lines": [{"t": 1.0, "text": "one"}, {"t": 4.0, "text": "two"}]}
    text = build_chordpro(title="T", artist="", analysis=ANALYSIS, segments=segs, lyrics=lyrics, duration=20)
    assert "[Am]one" in text and "\ntwo" in text


def test_plain_lines_drop_section_headers():
    from app.pipeline.lyrics import plain_lines
    assert [x["text"] for x in plain_lines("[Verse 1]\nhello\n(x2)\n[Chorus]\nworld [yeah]")] == ["hello", "world (yeah)"]


def test_lyrics_failure_does_not_fail(tmp_path, monkeypatch):
    import json
    from app.pipeline import lyrics
    monkeypatch.setattr(lyrics, "search_lrclib", lambda *a: (_ for _ in ()).throw(ValueError("html page")))
    (tmp_path / "cache").mkdir()
    out = lyrics.fetch_lyrics(tmp_path, tmp_path / "a.wav", {"title": "A - B", "duration": 10})
    assert out["source"] == "none" and "error" in out
