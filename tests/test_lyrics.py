import numpy as np

from app.pipeline.lyrics import estimate_offset, music_stretches


def _activity(levels: list[tuple[float, float, float]], seconds: float = 90.0, hop: float = 0.1) -> np.ndarray:
    act = np.full(int(seconds / hop), -1.0)
    for a, b, level in levels:
        act[int(a / hop):int(b / hop)] = level
    return act


LINES = [{"t": float(t), "text": "la la"} for t in (10, 20, 30, 40)]


def test_loud_outro_no_longer_pulls_the_lyrics_late():
    # a loud band from 5 to 52 s with the voice barely above it (lines sung at 12, 22, 32, 42 s),
    # then a video outro with three loud bursts spaced like sung lines
    act = _activity([(5, 52, 0.5), (12, 15, 0.8), (22, 25, 0.8), (32, 35, 0.8), (42, 45, 0.8),
                     (58, 61, 1.0), (68, 71, 1.0), (78, 81, 1.0)])
    late = estimate_offset(LINES, act, 0.1, video_duration=90, lyrics_duration=40)
    assert late > 40                                            # activity alone: fooled by the outro
    music = [(5.0, 52.0)]                                       # chords were only found up to 52 s
    assert abs(estimate_offset(LINES, act, 0.1, 90, 40, music) - 2.0) <= 0.5


def test_music_stretches_join_neighbouring_chords_and_skip_no_chord():
    analysis = {"segments": [{"chord": None, "start": 0.0, "end": 5.0}, {"chord": [0, "maj"], "start": 5.0, "end": 9.0},
                             {"chord": [7, "maj"], "start": 9.0, "end": 12.0}, {"chord": None, "start": 12.0, "end": 20.0},
                             {"chord": [0, "maj"], "start": 20.0, "end": 24.0}]}
    assert music_stretches(analysis) == [(5.0, 12.0), (20.0, 24.0)]
    assert music_stretches(None) is None and music_stretches({"segments": []}) is None
