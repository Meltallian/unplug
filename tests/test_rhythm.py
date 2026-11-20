import numpy as np

from app.pipeline.rhythm import bar_grid, fit_grid, notated_tempo, regular_beats, tempo


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


def test_tempo_is_finer_than_the_tracker_raster():
    true = 60 / 87.0                                         # 0.6897 s between beats
    beats = np.round(np.arange(200) * true / 0.02) * 0.02    # the tracker answers in 20 ms steps
    assert abs(tempo(beats) - 87.0) < 0.3
    assert abs(60 / np.median(np.diff(beats)) - 87.0) > 1.0  # the median alone would be off


def test_holes_in_the_beat_track_are_filled_and_the_grid_reaches_both_ends():
    full = 2.0 + np.arange(100) * 0.5                        # beats from 2 s to 51.5 s
    seen = np.delete(full, [20, 21, 22, 60])                 # the tracker lost a few
    beats = regular_beats(seen, duration=60.0)
    assert np.allclose(np.diff(beats), 0.5)
    assert beats[0] <= 0.5 and beats[-1] > 59.0
    assert all(np.min(np.abs(beats - t)) < 1e-9 for t in full)


def test_a_long_or_uneven_hole_is_left_alone():
    a = np.arange(40) * 0.5
    b = a[-1] + 0.5 * 30.3 + np.arange(40) * 0.5            # a break that is no whole number of beats
    beats = regular_beats(np.concatenate([a, b]), duration=float(b[-1]) + 0.2)
    assert np.max(np.diff(beats)) > 10


def test_fast_beats_in_threes_are_six_eight_and_the_chords_choose_the_bars():
    beats = np.arange(240) * 0.4                              # 150 beats a minute
    downs = beats[::3]                                        # the tracker hears a "1" every three beats
    novelty = np.zeros(len(beats))
    novelty[3::6] = 1.0                                       # the chord changes every six, from beat 3 on
    meter, pos = bar_grid(beats, downs, novelty)
    assert meter == 6
    assert list(pos[3:15]) == [0, 1, 2, 3, 4, 5] * 2
    assert abs(notated_tempo(beats, meter) - 50.0) < 0.01     # dotted quarters


def test_slow_beats_in_threes_stay_a_waltz_and_six_slow_beats_become_one():
    beats = np.arange(120) * 0.65                             # 92 beats a minute
    assert bar_grid(beats, beats[::3])[0] == 3
    meter, pos = bar_grid(beats, beats[::6])
    assert meter == 3 and list(pos[:6]) == [0, 1, 2, 0, 1, 2]
    assert abs(notated_tempo(beats, 3) - 60 / 0.65) < 0.01
