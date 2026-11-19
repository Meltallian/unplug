import pytest

from app.eval.sheet import chord_line, parse_sheet
from app.music import choose_capo, diatonic, key_from_chords, parse_chord, transpose


@pytest.mark.parametrize("token,expected", [
    ("Am", (9, "min")), ("Bb", (10, "maj")), ("F#m", (6, "min")), ("Fmaj7", (5, "maj")),
    ("G*", (7, "maj")), ("C/G", (0, "maj")), ("Dsus4", (2, "maj")), ("E5", (4, "maj")),
    ("Am7", (9, "min")), ("N.C.", "N"),
])
def test_parse_chord(token, expected):
    assert parse_chord(token) == expected


@pytest.mark.parametrize("token", ["I", "And", "Ah-ah,", "Em,", "then", "Am|"])
def test_not_chords(token):
    assert parse_chord(token) is None


def test_chord_line_detection():
    assert chord_line("Am C G Am F") is not None
    assert chord_line("Am G Dm F x1") is not None
    assert chord_line("  I walk a lonely road ") is None
    assert chord_line("E|------3-----|") is None


def test_capo_transposes_sheet_to_sounding_chords():
    sheet = parse_sheet("[Verse]\nC   Am\nlyrics\nEm  D\n", capo=1)
    assert sheet.chords == [(1, "maj"), (10, "min"), (5, "min"), (3, "maj")]


def test_key_from_chords():
    assert key_from_chords([(c, 1.0) for c in [(0, "maj"), (5, "maj"), (7, "maj"), (9, "min")]]) in [(0, "major"), (9, "minor")]


def test_minor_key_allows_major_five():
    assert (4, "maj") in diatonic((9, "minor"))


def test_capo_none_for_open_song():
    c_major = [((0, "maj"), 1), ((9, "min"), 1), ((5, "maj"), 1), ((7, "maj"), 1)]
    assert choose_capo(c_major) == 0


def test_capo_for_barre_heavy_song():
    bb = [((10, "maj"), 1), ((7, "min"), 1), ((2, "min"), 1), ((5, "maj"), 1)]   # Bb Gm Dm F
    capo = choose_capo(bb)
    assert capo > 0
    assert transpose((10, "maj"), -capo) in [(7, "maj"), (9, "maj"), (2, "maj"), (0, "maj"), (4, "maj")]


def test_corrupt_chord_cache_is_recomputed(tmp_path, monkeypatch):
    from app.pipeline import chords
    out = tmp_path / "chords.json"
    out.write_text('{"version": "baseline-2", "seg')          # truncated by a crash
    monkeypatch.setattr(chords.librosa, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recomputing")))
    import pytest
    with pytest.raises(RuntimeError, match="recomputing"):
        chords.analyze(tmp_path / "x.wav", out)
