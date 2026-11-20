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


def _beat_probs(*chords_per_beat):
    """Model output for a row of beats: each beat is a dict {(root, quality) or "N": probability}."""
    import numpy as np
    from app.pipeline import btc
    probs = np.full((len(chords_per_beat), btc.N_CHORDS), 1e-4)
    for i, beat in enumerate(chords_per_beat):
        for chord, value in beat.items():
            probs[i, btc.N_INDEX if chord == "N" else btc.chord_index(*chord)] = value
    return probs


def _decode(probs, pos=None, meter=4, **params):
    import numpy as np
    from app.pipeline.chords import Params, decode_beats
    n = len(probs)
    pos = np.arange(n) % meter if pos is None else pos
    segments, _key = decode_beats(probs, np.ones(n), np.arange(n + 1.0), pos, meter, Params(**params))
    return [(s["chord"], s["start"], s["end"]) for s in segments]


def test_richer_chords_count_towards_their_plain_chord():
    # the model hesitates between G and G7 on every beat, and sometimes prefers Em: it is G all along
    g = {(7, "maj"): 0.3, (7, "7"): 0.3, (4, "min"): 0.35}
    assert _decode(_beat_probs(*[g] * 8)) == [((7, "maj"), 0.0, 8.0)]


def test_seventh_is_written_only_when_the_model_is_sure():
    sure = _beat_probs(*[{(4, "min7"): 0.8, (4, "min"): 0.1}] * 4)
    unsure = _beat_probs(*[{(4, "min7"): 0.4, (4, "min"): 0.5}] * 4)
    assert _decode(sure)[0][0] == (4, "min7")
    assert _decode(unsure)[0][0] == (4, "min")
    assert _decode(sure, quality_share=1.01)[0][0] == (4, "min")     # switched off


def test_silent_beats_are_no_chord():
    import numpy as np
    from app.pipeline.chords import decode_beats
    probs = _beat_probs(*[{(0, "maj"): 0.9}] * 8)
    rms = np.array([1, 1, 1, 1, 0.001, 0.001, 0.001, 0.001])
    segments, _ = decode_beats(probs, rms, np.arange(9.0), np.arange(8) % 4, 4)
    assert [s["chord"] for s in segments] == [(0, "maj"), "N"]


def test_span_means_weighs_cut_frames_by_their_part_inside():
    import numpy as np
    from app.pipeline.chords import span_means
    frames = np.array([[1.0], [3.0], [5.0], [7.0]])            # one value per second
    means = span_means(frames, 1.0, np.array([0.0, 1.5, 4.0, 9.0]))
    assert np.allclose(means[:, 0], [(1 + 1.5) / 1.5, (1.5 + 5 + 7) / 2.5, 0.0])


def test_corrupt_features_cache_is_recomputed(tmp_path, monkeypatch):
    import pytest
    from app.pipeline import chords
    (tmp_path / f"features-{chords.FEATURES_VERSION}.npz").write_bytes(b"PK\x03\x04 truncated")
    monkeypatch.setattr(chords.librosa, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("recomputing")))
    with pytest.raises(RuntimeError, match="recomputing"):
        chords.features(tmp_path / "audio.wav")


def test_chord_names_of_richer_chords():
    from app.music import chord_name, plain, shape_cost
    assert [chord_name((4, q)) for q in ("maj", "min", "7", "min7", "maj7", "sus4", "hdim7")] == \
        ["E", "Em", "E7", "Em7", "Emaj7", "Esus4", "Em7b5"]
    assert plain((4, "min7")) == (4, "min") and plain((2, "sus4")) == (2, "maj")
    assert shape_cost((4, "min7")) == 0 and shape_cost((5, "maj7")) < shape_cost((5, "maj"))
