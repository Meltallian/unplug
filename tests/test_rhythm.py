import numpy as np

from app.pipeline.rhythm import bar_grid, fit_grid, regular_beats, tempo


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
    probs[:4, c] = 0.9
    probs[4, c], probs[4, am] = 0.6, 0.3            # the model still leans to C one beat into the second bar
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
    grid = bar_grid(beats, downs, novelty)
    assert grid.main_meter == 6 and set(grid.size) == {6}
    assert list(grid.pos[3:15]) == [0, 1, 2, 3, 4, 5] * 2
    assert grid.sections() == [{"start": 0.0, "meter": 6, "tempo": 50.0}]   # dotted quarters


def test_slow_beats_in_threes_stay_a_waltz_and_six_slow_beats_become_one():
    beats = np.arange(120) * 0.65                             # 92 beats a minute
    assert bar_grid(beats, beats[::3]).main_meter == 3
    grid = bar_grid(beats, beats[::6])
    assert grid.main_meter == 3 and list(grid.pos[:6]) == [0, 1, 2, 0, 1, 2]
    assert abs(grid.sections()[0]["tempo"] - 60 / 0.65) < 0.01


def test_chorus_in_six_eight_and_verse_in_four_four_each_get_their_bars():
    # 150 eighth notes a minute throughout: 16 bars of 6/8, then 12 bars of 4/4 (8 eighths), then 6/8 again
    beats = np.arange(16 * 6 + 12 * 8 + 16 * 6) * 0.4
    a, b = 16 * 6, 16 * 6 + 12 * 8
    downs = np.concatenate([beats[:a:3], beats[a:b:8], beats[b::3]])
    novelty = np.zeros(len(beats))
    novelty[:a:6] = novelty[b::6] = 1.0                       # the chords change with the bars
    grid = bar_grid(beats, downs, novelty)
    assert list(grid.pos[:12]) == [0, 1, 2, 3, 4, 5] * 2
    assert list(grid.pos[a:a + 16]) == [0, 1, 2, 3, 4, 5, 6, 7] * 2 and set(grid.size[a:b]) == {8}
    assert list(grid.pos[b:b + 6]) == [0, 1, 2, 3, 4, 5]
    assert grid.sections() == [{"start": 0.0, "meter": 6, "tempo": 50.0},
                               {"start": round(float(beats[a]), 3), "meter": 4, "tempo": 75.0},
                               {"start": round(float(beats[b]), 3), "meter": 6, "tempo": 50.0}]


def test_a_fast_song_in_four_keeps_its_bars_and_its_tempo():
    beats = np.arange(400) * 0.28                             # 214 beats a minute, bars of four
    grid = bar_grid(beats, beats[::4])
    assert set(grid.size) == {4} and grid.main_meter == 4
    assert abs(grid.sections()[0]["tempo"] - 60 / 0.28) < 0.01 and len(grid.sections()) == 1


def test_a_section_tracked_at_half_speed_is_put_on_the_pulse_of_the_song():
    fast = np.arange(120) * 0.4
    slow = fast[-1] + 0.8 + np.arange(40) * 0.8               # the tracker counts this part in quarters
    back = slow[-1] + 0.4 + np.arange(120) * 0.4
    beats = regular_beats(np.concatenate([fast, slow, back]), duration=float(back[-1]) + 0.3)
    assert np.allclose(np.diff(beats), 0.4)


def test_a_short_run_tracked_at_double_speed_loses_the_extra_beats():
    a = np.arange(100) * 0.5
    double = a[-1] + 0.25 + np.arange(16) * 0.25              # 4 s counted twice as fast
    b = double[-1] + 0.5 + np.arange(100) * 0.5
    beats = regular_beats(np.concatenate([a, double, b]), duration=float(b[-1]) + 0.3)
    assert np.allclose(np.diff(beats), 0.5)


def test_a_part_tracked_in_triplets_is_put_on_the_beat():
    slow = np.arange(100) * 0.84                             # most of the song: 71 beats a minute
    triplets = -0.28 * np.arange(1, 91)[::-1]                # the first 25 s counted three times as fast
    beats = regular_beats(np.concatenate([triplets, slow]) + 26.0, duration=110.0)
    assert np.allclose(np.diff(beats), 0.84)


def test_pulse_is_what_most_of_the_time_is_counted_in():
    from app.pipeline.rhythm import _typical
    assert _typical(np.array([0.28] * 90 + [0.84] * 100)) == 0.84     # 25 s of triplets, 84 s of beats
    assert _typical(np.array([0.4] * 380 + [0.8] * 100)) == 0.4      # 152 s of eighths, 80 s of quarters
