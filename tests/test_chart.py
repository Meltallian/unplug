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


def test_replace_chord():
    from app.pipeline.chart import replace_chord
    text = "{title: T}\n{c: Verse}\n[Am]Is it so [C]perfect\n[F]  [G]\n"
    assert replace_chord(text, 1, "Em") == "{title: T}\n{c: Verse}\n[Am]Is it so [Em]perfect\n[F]  [G]\n"
    assert replace_chord(text, 3, None) == "{title: T}\n{c: Verse}\n[Am]Is it so [C]perfect\n[F]  \n"
    import pytest
    with pytest.raises(IndexError):
        replace_chord(text, 9, "C")


def test_replace_chord_skips_empty_brackets_and_checks_expected():
    import pytest
    from app.pipeline.chart import parse_chordpro, replace_chord
    text = "[ ]la [Am]la [C]la\n"
    # the page numbers only real chords: Am = 0, C = 1
    chords = [c for line in parse_chordpro(text)["sections"][0]["lines"] for c, _ in line if c]
    assert chords == ["Am", "C"]
    assert replace_chord(text, 1, "G", expected="C") == "[ ]la [Am]la [G]la\n"
    with pytest.raises(ValueError):
        replace_chord(text, 1, "G", expected="Am")


BARS = {**ANALYSIS, "meter": 4, "downbeats": [float(t) for t in range(0, 40, 2)]}   # a bar every 2 s


def test_instrumental_is_written_as_bars_with_repeat_signs():
    from app.pipeline.chart import line_kind
    # Am for two bars, then C and G sharing one bar, then F; lyrics start at 10 s
    segs = [(AM, 0, 4, 8), (C, 4, 5, 2), (G, 5, 6, 2), (F, 6, 10, 8), (C, 10, 14, 8)]
    lyrics = {"offset": 0.0, "lines": [{"t": 10.0, "text": "First sung line"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=14)
    assert "| [Am] | % | [C] [G] | [F] |\n| % |" in text
    assert "{time: 4/4}" in text
    parsed = parse_chordpro(text)
    assert parsed["meta"]["time"] == "4/4"
    intro = parsed["sections"][0]
    assert intro["name"] == "Intro" and [line_kind(l) for l in intro["lines"]] == ["bars", "bars"]
    assert line_kind(parsed["sections"][1]["lines"][0]) == "lyric-bars"     # sung lines are rows of bars too


def test_no_lyrics_chart_is_bars():
    segs = [(AM, 0, 4, 8), (F, 4, 8, 8)]
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics={"lines": []}, duration=8)
    assert "| [Am] | % | [F] | % |" in text


def test_line_kind():
    from app.pipeline.chart import line_kind
    assert line_kind([(None, "| "), ("Am", " | % |")]) == "bars"
    assert line_kind([("Am", " "), ("F", "")]) == "chords-only"
    assert line_kind([("Am", "Hello | world")]) == "lyric-bars"     # any line with "|" is a row of bars
    assert line_kind([("Am", "Hello world")]) == ""


def test_bar_cells():
    from app.pipeline.chart import bar_cells
    line = parse_chordpro("{c: Intro}\n| [Am] | % | [C] [G] | [F] |\n")["sections"][0]["lines"][0]
    assert bar_cells(line) == [[("chord", "Am")], [("mark", "%")], [("chord", "C"), ("chord", "G")], [("chord", "F")]]


def test_bars_continue_where_the_beat_grid_stops_short():
    # grid detected only from 4 s to 10 s; chords from 0 s to 20 s, no lyrics
    analysis = {**ANALYSIS, "meter": 4, "downbeats": [4.0, 6.0, 8.0, 10.0]}
    segs = [(AM, 0, 4, 8), (C, 4, 10, 12), (G, 10, 14, 8), (F, 14, 20, 12)]
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics={"lines": []}, duration=20)
    rows = [l for l in text.splitlines() if l.startswith("|")]
    assert rows == ["| [Am] | % | [C] | % |", "| % | [G] | % | [F] |", "| % | % |"]


def test_outro_before_the_first_downbeat_is_kept():
    analysis = {**ANALYSIS, "meter": 4, "downbeats": [195.0, 197.0, 199.0]}
    segs = [(C, 180, 189, 8), (AM, 189, 196, 8)]
    lyrics = {"offset": 0.0, "lines": [{"t": 185.0, "text": "last line"}]}
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics=lyrics, duration=197)
    assert "{c: Outro}" in text and "| [Am] |" in text


def test_repeat_sign_only_after_a_single_chord_bar():
    segs = [(C, 0, 2, 4), (C, 2, 3, 2), (G, 3, 6, 6), (F, 6, 8, 4)]
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics={"lines": []}, duration=8)
    assert "| [C] | [C] [G] | [G] | [F] |" in text


def test_silent_bar_inside_is_no_chord():
    segs = [(C, 0, 2, 4), ("N", 2, 4, 4), (G, 4, 6, 4)]
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics={"lines": []}, duration=6)
    assert "| [C] | N.C. | [G] |" in text


def test_emptied_bar_keeps_its_column():
    from app.pipeline.chart import bar_cells, line_kind
    line = parse_chordpro("{c: Intro}\n| [Am] |  | [F] |\n")["sections"][0]["lines"][0]
    assert line_kind(line) == "bars"
    assert bar_cells(line) == [[("chord", "Am")], [], [("chord", "F")]]
    assert line_kind(parse_chordpro("| [C] | N.C. | % |")["sections"][0]["lines"][0]) == "bars"


def test_gap_in_the_beat_grid_is_filled_with_bars():
    from app.pipeline.chart import _bar_edges
    edges = _bar_edges([0.0, 2.0, 4.0, 12.0, 14.0], 0.0, 16.0)
    assert edges == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]


def test_chord_ringing_before_the_first_bar_line_is_kept():
    analysis = {**ANALYSIS, "meter": 4, "downbeats": [10.6, 12.6, 14.6]}
    segs = [(F, 5, 10.4, 8), (C, 10.4, 16, 8)]
    lyrics = {"offset": 0.0, "lines": [{"t": 0.5, "text": "a line"}, {"t": 1.0, "text": "end"}]}
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics=lyrics, duration=16)
    # C starts on the last beat of the bar F rings in: dots keep it there ("[F] [C]" would read as beat 3)
    assert "| [F] . . [C] | [C] | % | % |" in text


# --- bars everywhere (SCORE_SPEC section 2) -----------------------------------------------------

def _rows(text):
    return [l for l in text.splitlines() if "|" in l]


def test_sung_lines_are_rows_of_bars():
    # a bar every 2 s; Am 0-4, C 4-8, F 8-12, G 12-18. The first line owns 8 s (4-12): "aa" at 4 s,
    # "bb" at 8.8 s; the second starts at 14 s and owns 4 s: "cc" at 14, "dd" at 15.5, "ee" at 17.
    segs = [(AM, 0, 4, 8), (C, 4, 8, 8), (F, 8, 12, 8), (G, 12, 18, 12)]
    lyrics = {"offset": 0.0, "lines": [{"t": 4.0, "text": "aa bb"}, {"t": 14.0, "text": "cc dd ee"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=18)
    assert text.split("\n\n", 1)[1] == (
        "{c: Intro}\n"
        "| [Am] | % |\n"
        "\n"
        "{c: Verse 1}\n"
        "| [C]aa | | [F]bb | | [G] |\n"     # a chord only where it changes; bars without words stay in the row
        "| cc dd | ee |\n")


def test_line_starting_late_in_a_bar_is_a_pickup():
    segs = [(AM, 0, 4, 8), (C, 4, 12, 16)]
    lyrics = {"offset": 0.0, "lines": [{"t": 3.5, "text": "And so it goes on"}]}     # 0.5 s before the bar line
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=12)
    assert _rows(text) == ["| [Am] | % |", "And | [C]so | it | goes | on |"]
    # ... but not when there is no bar before it to sing the pickup in
    lyrics = {"offset": 0.0, "lines": [{"t": 1.5, "text": "And so it goes on"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=[(C, 2, 12, 16)], lyrics=lyrics, duration=12)
    assert _rows(text) == ["| And | [C]so | it | goes | on |", "| [C] |"]       # the last bar is the outro


def test_line_starting_in_a_bar_still_sung_is_a_pickup():
    segs = [(C, 0, 12, 24)]
    lyrics = {"offset": 0.0, "lines": [{"t": 0.0, "text": "one two three"}, {"t": 4.5, "text": "four five six seven"},
                                       {"t": 5.0, "text": "yeah"}, {"t": 5.5, "text": "eight nine"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=12)
    rows = _rows(text)
    assert rows[0] == "| [C]one two | three |"
    assert rows[1] == "| four five six seven |"       # 0.5 s for four words: all in the bar they start in
    assert rows[2] == "yeah |"                         # sung in that same bar: a pickup with no bar of its own
    assert rows[3] == "eight | | nine | |"             # "eight" too; "nine" comes two bars later


def test_chord_off_its_default_beat_is_written_with_dots():
    # bars of 2 s in 4/4: G comes in on beat 4 of the first bar, F on beat 2 of the third
    segs = [(C, 0, 1.5, 3), (G, 1.5, 4.5, 6), (F, 4.5, 8, 7)]
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics={"lines": []}, duration=8)
    assert "| [C] . . [G] | [G] | [G] [F] . . | [F] |" in text
    # three beats in a bar: the default place of a second chord is beat 3, so beat 2 needs a dot
    waltz = {**ANALYSIS, "meter": 3, "downbeats": [0.0, 3.0, 6.0, 9.0]}
    segs = [(C, 0, 1, 1), (G, 1, 5, 4), (F, 5, 9, 4)]
    text = build_chordpro(title="T", artist="", analysis=waltz, segments=segs, lyrics={"lines": []}, duration=9)
    assert "{time: 3/4}" in text and "| [C] [G] . | [G] [F] | [F] |" in text


def test_words_follow_the_chord_they_are_sung_on():
    segs = [(C, 0, 3.5, 7), (G, 3.5, 8, 9)]          # G on beat 4 of the second bar
    lyrics = {"offset": 0.0, "lines": [{"t": 2.0, "text": "aa bb cc dd"}]}       # words at 2, 3.6, 5.3 and 6.9 s
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=8)
    assert _rows(text) == ["| [C]aa . . [G]bb | cc | dd |"]


def test_silence_under_sung_bars_is_no_chord():
    segs = [(C, 0, 2, 4), ("N", 2, 6, 8), (G, 6, 10, 8)]
    lyrics = {"offset": 0.0, "lines": [{"t": 0.0, "text": "la la la la la"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=10)
    assert _rows(text) == ["| [C]la la | N.C. la | la | [G]la |", "| [G] |"]       # the second row is the outro


def test_words_that_look_like_marks_are_made_harmless():
    segs = [(C, 0, 8, 16)]
    lyrics = {"offset": 0.0, "lines": [{"t": 0.0, "text": "100 % sure | or . not N.C."}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=8)
    words = " ".join(_rows(text)).replace("[C]", "").split()
    assert [w for w in words if w != "|"] == ["100", "(%)", "sure", "/", "or", "not", "(N.C.)"]


def test_plain_lyrics_keep_chords_over_lines():
    # no timing: the words cannot be put in bars, so the lines stay as before (and the intro stays bars)
    segs = [(AM, 0, 8, 16), (C, 8, 16, 16)]
    lyrics = {"lines": [{"t": None, "text": "first line"}, {"t": None, "text": "second line"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=16)
    assert "[Am]first line\n[C]second line" in text


def test_meter_change_between_sections():
    # chorus in 6/8 (bars of 3 s), verse in 4/4 (bars of 2 s) from 6 s, chorus again from 14 s
    analysis = {**ANALYSIS, "tempo": 50.0, "meter": 6, "downbeats": [0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0, 17.0],
                "meters": [{"start": 0.0, "meter": 6, "tempo": 50.0}, {"start": 6.0, "meter": 4, "tempo": 75.0},
                           {"start": 14.0, "meter": 6, "tempo": 50.0}]}
    segs = [(AM, 0, 6, 12), (C, 6, 9, 6), (G, 9, 14, 10), (F, 14, 20, 12)]
    lyrics = {"offset": 0.0, "lines": [{"t": 6.0, "text": "one two three four"}, {"t": 13.5, "text": "back again now"}]}
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics=lyrics, duration=20)
    assert text.split("\n\n", 1)[1] == (
        "{c: Intro}\n"
        "| [Am] | % |\n"
        "\n"
        "{c: Verse 1}\n"
        "{time: 4/4}\n"
        "{tempo: 75}\n"
        "| [C]one two | [C] [G]three | four | |\n"
        "back |\n"                     # the pickup is sung in the last 4/4 bar
        "{time: 6/8}\n"
        "{tempo: 50}\n"
        "| [F]again | now |\n")
    parsed = parse_chordpro(text)
    assert parsed["meta"]["time"] == "6/8" and parsed["meta"]["tempo"] == "50"       # the header: how the song starts
    verse = parsed["sections"][1]
    assert verse["changes"] == {0: {"time": "4/4", "tempo": "75", "label": "4/4 · ♩ = 75"},
                                2: {"time": "6/8", "tempo": "50", "label": "6/8 · ♩. = 50"}}


def test_meter_change_inside_an_instrumental_row():
    analysis = {**ANALYSIS, "tempo": 120.0, "meter": 4, "downbeats": [0.0, 2.0, 4.0, 6.0, 9.0, 12.0],
                "meters": [{"start": 0.0, "meter": 4, "tempo": 120.0}, {"start": 6.0, "meter": 6, "tempo": 40.0}]}
    segs = [(AM, 0, 6, 12), (C, 6, 15, 18)]
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=segs, lyrics={"lines": []}, duration=15)
    assert text.endswith("{c: Chords}\n| [Am] | % | % |\n{time: 6/8}\n{tempo: 40}\n| [C] | % | % |\n")
    # the chord that keeps ringing is written again where the meter changes
    text = build_chordpro(title="T", artist="", analysis=analysis, segments=[(AM, 0, 15, 30)], lyrics={"lines": []}, duration=15)
    assert text.endswith("| [Am] | % | % |\n{time: 6/8}\n{tempo: 40}\n| [Am] | % | % |\n")


def test_replace_chord_counts_the_marks_of_bar_rows():
    from app.pipeline.chart import replace_chord
    text = "{c: Verse}\n{time: 3/4}\nBy now | [Em7]should've | [G] . [D]gotta | % |\n| N.C. | [A7sus4] |\n"
    assert replace_chord(text, 2, "Dsus4", expected="D").splitlines()[2] == "By now | [Em7]should've | [G] . [Dsus4]gotta | % |"
    assert replace_chord(text, 3, None, expected="A7sus4").splitlines()[3] == "| N.C. |  |"


def test_meter_of():
    from app.pipeline.chart import meter_of
    assert meter_of("4/4") == (4, 4, 4) and meter_of("3/4") == (3, 4, 3) and meter_of(" 2/4 ") == (2, 4, 2)
    assert meter_of("6/8") == (6, 8, 2) and meter_of("12/8") == (12, 8, 4) and meter_of("9/8") == (9, 8, 3)
    assert meter_of(None) == (4, 4, 4) and meter_of("waltz") == (4, 4, 4) and meter_of("0/4") == (4, 4, 4)


def test_bar_row_splits_a_sung_row():
    from app.pipeline.chart import bar_row, line_kind
    line = parse_chordpro("By now you | [Em7]should've somehow | [C] . . [D]do | N.C. hush | | % |\n")["sections"][0]["lines"][0]
    pickup, bars = bar_row(line)
    assert line_kind(line) == "lyric-bars"
    assert pickup == [("", None, "By now you ")]
    assert bars == [[("chord", "Em7", "should've somehow ")],
                    [("chord", "C", " "), ("mark", ".", " "), ("mark", ".", " "), ("chord", "D", "do ")],
                    [("mark", "N.C.", " hush ")], [], [("mark", "%", " ")]]
    # dots do not make an instrumental row a sung one
    dots = parse_chordpro("| [C] . . [D] | % |\n")["sections"][0]["lines"][0]
    assert line_kind(dots) == "bars"
    from app.pipeline.chart import bar_cells
    assert bar_cells(dots) == [[("chord", "C"), ("mark", "."), ("mark", "."), ("chord", "D")], [("mark", "%")]]


def test_later_time_and_tempo_do_not_overwrite_the_header():
    p = parse_chordpro("{title: T}\n{tempo: 50}\n{time: 6/8}\n{c: A}\n| [C] |\n{c: B}\n{tempo: 75}\n{time: 4/4}\n| [G] |\n"
                       "{tempo: 80}\n| [D] |\n{time: 3/2}\n")
    assert p["meta"] == {"title": "T", "tempo": "50", "time": "6/8"}
    assert p["sections"][0]["changes"] == {}
    assert p["sections"][1]["changes"] == {0: {"time": "4/4", "tempo": "75", "label": "4/4 · ♩ = 75"},
                                           1: {"time": None, "tempo": "80", "label": "♩ = 80"},
                                           2: {"time": "3/2", "tempo": None, "label": "3/2"}}
    # a chart whose first section gives the tempo (hand-written): still the header
    p = parse_chordpro("{c: Verse}\n{tempo: 120}\n[C]hello\n")
    assert p["meta"] == {"tempo": "120"} and p["sections"][0]["changes"] == {}


# --- reading a chart as a score -------------------------------------------------------------------

def _beats(bar):
    """A bar as text: one item per beat, "Chord:words"."""
    return ["+".join(name for name, _ in beat["chords"]) + (":" + " ".join(beat["words"]) if beat["words"] else "")
            for beat in bar["beats"]]


def _bars(text):
    from app.pipeline.chart import read_score
    return [_beats(bar) for section in read_score(text)["sections"] for bar in section["bars"]]


def test_score_places_chords_on_beats():
    # one chord: the whole bar; two: the second on beat 3; as many chords and dots as beats: one per beat
    assert _bars("| [C] | [C] [G] | [C] . . [G] | [C] [D] [Em] | [C] [D] [Em] [F] | |\n") == [
        ["C", "", "", ""], ["C", "", "G", ""], ["C", "", "", "G"], ["C", "D", "Em", ""], ["C", "D", "Em", "F"], ["", "", "", ""]]
    assert _bars("{time: 3/4}\n| [C] [G] | [C] [G] . | [C] |\n") == [["C", "", "G"], ["C", "G", ""], ["C", "", ""]]
    assert _bars("{time: 6/8}\n| [C] [G] | [C] | [C] [D] [Em] |\n") == [["C", "G"], ["C", ""], ["C+D", "Em"]]
    assert _bars("{time: 12/8}\n| [C] [G] |\n") == [["C", "", "G", ""]]
    # dots that do not add up to the bar are not beats
    assert _bars("| [C] . [G] |\n") == [["C", "", "G", ""]]


def test_score_repeat_and_no_chord():
    from app.pipeline.chart import read_score
    score = read_score("| [C] [G] | % | N.C. | [N.C.] |\n")
    bars = score["sections"][0]["bars"]
    assert [_beats(b) for b in bars] == [["C", "", "G", ""], ["C", "", "G", ""], ["N.C.", "", "", ""], ["N.C.", "", "", ""]]
    # only real chord marks are numbered, like replace_chord counts them ("[N.C.]" is a mark, "N.C." is not)
    assert [c for b in bars for beat in b["beats"] for c in beat["chords"]] == [
        ("C", 0), ("G", 1), ("C", None), ("G", None), ("N.C.", None), ("N.C.", 2)]


def test_score_spreads_words_over_the_beats_of_their_chord():
    assert _bars("| [Em7]Today is gonna be the | [G]day [D]that they're | so [C]what |\n") == [
        ["Em7:Today is", ":gonna", ":be", ":the"],              # five words on the four beats of Em7
        ["G:day", "", "D:that", ":they're"],                    # one word on G's two beats, two on D's
        ["C:so", "", ":what", ""]]                              # words before the first chord share its beats
    assert _bars("| [C]one . . [D]two |\n") == [["C:one", "", "", "D:two"]]
    assert _bars("| just words here |\n") == [[":just", ":words", ":here", ""]]


def test_score_pickup_is_sung_at_the_end_of_the_bar_before():
    text = ("{c: Intro}\n| [G] | [D] |\n{c: Verse}\nBy now you | [Em]should've |\n"
            "And | [C]so on and on |\nYeah yeah |\n")
    assert _bars(text) == [["G", "", "", ""],
                           ["D", "", ":By now", ":you"],                     # the last bar of the section above
                           ["Em:should've", "", "", ":And"],
                           ["C:so on", ":and on", ":Yeah", ":yeah"]]         # the bar's own words make room for the pickup
    # no bar before it: on the "1" of the first bar
    assert _bars("Hey | [C]you |\n") == [["C:Hey you", "", "", ""]]
    assert _bars("Hey |\n") == [[":Hey", "", "", ""]]


def test_score_numbers_chords_like_replace_chord():
    from app.pipeline.chart import read_score, replace_chord
    text = "{title: [X]}\n{c: Verse}\nBy [D]now | [Em7]should've [ ] | [G] . . [C]do | % |\n[Am]old [F]line\n"
    score = read_score(text)
    marks = sorted((i, name) for s in score["sections"] for b in s["bars"] for beat in b["beats"]
                   for name, i in beat["chords"] if i is not None)
    assert marks == [(0, "D"), (1, "Em7"), (2, "G"), (3, "C"), (4, "Am"), (5, "F")]
    for i, name in marks:
        assert f"[{name}x]" in replace_chord(text, i, name + "x", expected=name)


def test_score_of_a_chart_without_bars_is_one_bar_per_chord():
    from app.pipeline.chart import read_score
    score = read_score("{c: Verse}\nIs it [Am]so perfect [C]now\nno chords here\n[F]  [G]\n")
    bars = score["sections"][0]["bars"]
    assert [_beats(b) for b in bars] == [[":Is", "", ":it", ""], ["Am:so", "", ":perfect", ""], ["C:now", "", "", ""],
                                         [":no", ":chords", ":here", ""], ["F", "", "", ""], ["G", "", "", ""]]
    assert all(b["guess"] for b in bars) and score["unbarred"] == 6
    assert read_score("| [C] |\n")["unbarred"] == 0
    assert read_score("{title: T}\n")["sections"] == []


def test_score_word_cut_by_a_chord_is_hyphenated():
    assert _bars("| Yester[C]day [G]all |\n") == [["C:Yesterday", "", "G:all", ""]]
    assert _bars("| [Am]Yester[C]day |\n") == [["Am:Yester-", "", "C:day", ""]]
    # only a chord mark cuts a word: two pickups written without a space after them stay two words
    assert _bars("| [C] |\nhey|\nyou | [G] |\n") == [["C", "", ":hey", ":you"], ["G", "", "", ""]]


def test_score_words_before_a_chord_that_dots_put_later():
    assert _bars("| so . . . [C]what |\n") == [[":so", "", "", "C:what"]]           # three beats of the chord before, C on 4
    assert _bars("| one two . . [C]three . |\n") == [[":one", ":two", "C:three", ""]]
    # a chord written in a pickup sits on the word that follows it
    assert _bars("| [G] |\nAnd [D]now | [C]go |\n") == [["G", "", ":And", "D:now"], ["C:go", "", "", ""]]


def test_score_follows_meter_changes():
    from app.pipeline.chart import read_score
    text = ("{time: 6/8}\n{tempo: 50}\n{c: Chorus}\n| [D]me get out | [C] |\n{c: Verse}\n{time: 4/4}\n{tempo: 75}\n"
            "| [C]It's hard to take | % |\nHelp |\n{time: 6/8}\n{tempo: 50}\n| [D]me |\n")
    score = read_score(text)
    assert (score["time"], score["beats"], score["tempo"]) == ((6, 8), 2, "50")
    bars = [b for s in score["sections"] for b in s["bars"]]
    assert [(b["time"], b["tempo"], len(b["beats"])) for b in bars] == [
        ((6, 8), "50", 2), ((6, 8), "50", 2), ((4, 4), "75", 4), ((4, 4), "75", 4), ((6, 8), "50", 2)]
    assert _beats(bars[3]) == ["C", "", "", ":Help"]         # the pickup belongs to the last 4/4 bar; % repeats C


def test_built_chart_reads_back_bar_for_bar():
    from app.pipeline.chart import read_score
    segs = [(AM, 0, 4, 8), (C, 4, 8, 8), (F, 8, 12, 8), (G, 12, 16, 8)]
    lyrics = {"offset": 0.0, "lines": [{"t": 3.5, "text": "one two three four five"}, {"t": 10.0, "text": "six seven"}]}
    text = build_chordpro(title="T", artist="", analysis=BARS, segments=segs, lyrics=lyrics, duration=16)
    score = read_score(text)
    assert [s["name"] for s in score["sections"]] == ["Intro", "Verse 1"]
    bars = [b for s in score["sections"] for b in s["bars"]]
    assert len(bars) == 8 and score["unbarred"] == 0          # 16 s of 2 s bars, none lost or doubled
    assert _beats(bars[1]) == ["Am", "", "", ":one"]          # "%" repeats Am; the pickup on the last beat
    assert _beats(bars[2])[0] == "C:two"


def test_one_long_block_is_cut_into_verses_and_choruses_by_the_lines_that_come_back():
    from app.pipeline.chart import _split_by_repetition
    chorus = ["Help me get out of it", "I can't keep my head up", "Please tell me, how do you feel", "I'm drowning in this river"]
    verse = ["It's hard to take", "For you, I'm a pawn", "Tried to make amends"]
    texts = chorus + verse + ["Oh, please tell me, how do you feel" if x.startswith("Please") else x for x in chorus] + chorus
    block = {"kind": "lyrics", "lines": [(float(i), float(i + 1), text) for i, text in enumerate(texts)]}
    parts = _split_by_repetition([{"kind": "inst", "start": 0.0, "end": 1.0}, block])
    assert [(p["kind"], p.get("repeats"), len(p.get("lines", []))) for p in parts] == \
        [("inst", None, 0), ("lyrics", True, 4), ("lyrics", False, 3), ("lyrics", True, 4), ("lyrics", True, 4)]


def test_a_refrain_line_inside_a_verse_does_not_cut_it():
    from app.pipeline.chart import _split_by_repetition
    texts = ["Line one", "Line two", "Hallelujah", "Line four", "Line five", "Line six", "Hallelujah", "Line eight"]
    block = {"kind": "lyrics", "lines": [(float(i), float(i + 1), text) for i, text in enumerate(texts)]}
    parts = _split_by_repetition([block])
    assert [(p["repeats"], len(p["lines"])) for p in parts] == [(False, 8)]
