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
    from app.pipeline import btc
    from app.pipeline.chords import decode_beats
    n = 12                                          # bar lines at beats 0, 4, 8
    probs = np.full((n, btc.N_CHORDS), 1e-4)
    c, am = btc.chord_index(0, "maj"), btc.chord_index(9, "min")
    probs[:4, c] = 0.9
    probs[4, [c, am]] = 0.45                        # ambiguous beat: C and Am equally likely
    probs[5:, am] = 0.9
    segments, _ = decode_beats(probs, np.ones(n), np.arange(n + 1.0), np.arange(n) % 4, 4)
    assert [(s["chord"], s["start"], s["on_bar"]) for s in segments] == [((0, "maj"), 0.0, True), ((9, "min"), 4.0, True)]


def test_late_chord_change_is_pulled_back_to_the_bar_line():
    from app.pipeline import btc
    from app.pipeline.chords import decode_beats
    n = 12
    probs = np.full((n, btc.N_CHORDS), 1e-4)
    c, am = btc.chord_index(0, "maj"), btc.chord_index(9, "min")
    probs[:5, c] = 0.8                              # the model hears C one beat into the second bar
    probs[:5, am] = 0.1
    probs[5:, am] = 0.9
    segments, _ = decode_beats(probs, np.ones(n), np.arange(n + 1.0), np.arange(n) % 4, 4)
    assert [(s["chord"], s["start"]) for s in segments] == [((0, "maj"), 0.0), ((9, "min"), 4.0)]


def test_random_downbeats_stay_four_four():
    rng = np.random.default_rng(1)
    beats = _beats(400)
    for _ in range(20):
        downs = np.sort(rng.choice(beats, 40, replace=False))
        assert fit_grid(beats, downs)[0] == 4


def test_six_eight_is_detected():
    beats = _beats(120, 0.3)
    meter, pos = fit_grid(beats, beats[::6])
    assert meter == 6 and list(pos[:7]) == [0, 1, 2, 3, 4, 5, 0]
