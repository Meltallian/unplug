import numpy as np

from app.pipeline.rhythm import fit_grid


def _beats(n, step=0.5):
    return np.arange(n) * step


def test_grid_is_regular_four_four_despite_noisy_downbeats():
    beats = _beats(64)
    downs = beats[::4].copy()
    downs = np.delete(downs, [3, 7, 8])            # the model missed some bar lines
    downs = np.sort(np.append(downs, beats[22]))   # and saw one in the wrong place
    meter, pos = fit_grid(beats, downs)
    assert meter == 4
    assert list(pos[:12]) == [0, 1, 2, 3] * 3
    assert all(pos[i] == i % 4 for i in range(64))


def test_grid_follows_the_bar_when_the_song_starts_on_a_pickup():
    beats = _beats(33)
    meter, pos = fit_grid(beats, beats[1::4])      # first beat is a pickup
    assert meter == 4 and pos[1] == 0 and pos[5] == 0 and pos[0] == 3


def test_three_four_is_detected():
    beats = _beats(60)
    meter, pos = fit_grid(beats, beats[::3])
    assert meter == 3 and list(pos[:6]) == [0, 1, 2, 0, 1, 2]


def test_no_downbeats_still_gives_a_grid():
    meter, pos = fit_grid(_beats(8), np.array([]))
    assert meter == 4 and len(pos) == 8


def test_chord_change_snaps_to_the_bar_line():
    from app.pipeline.chords import decode
    c, am, mix = np.zeros(12), np.zeros(12), np.zeros(12)
    c[[0, 4, 7]] = 1.0
    am[[9, 0, 4]] = 1.0
    mix[[0, 4, 7, 9]] = 1.0                         # ambiguous beat: C and Am equally likely
    cols = [c] * 4 + [mix] + [am] * 7               # bar lines at beats 0, 4, 8
    chroma = np.array(cols).T
    n = chroma.shape[1]
    feats = {"chroma": chroma, "bass": np.ones((12, n)), "rms": np.ones(n), "times": np.arange(n + 1.0),
             "pos": np.arange(n) % 4, "meter": np.array(4), "tuning": np.array(0.0), "tempo": np.array(120.0)}
    result = decode(feats)
    names = [(s["name"], s["start"], s["on_bar"]) for s in result["segments"]]
    assert names == [("C", 0.0, True), ("Am", 4.0, True)]
    assert result["downbeats"] == [0.0, 4.0, 8.0] and result["meter"] == 4
