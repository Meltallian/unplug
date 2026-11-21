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


def _recording(seconds: int, seed: int = 0, step: float = 0.1) -> np.ndarray:
    """A made-up recording: a new chord (one of 24) every 2 s, as (frames × 25) probabilities."""
    rng = np.random.default_rng(seed)
    chords = rng.integers(0, 24, size=seconds // 2 + 1)
    frames = np.full((int(seconds / step), 25), 0.01)
    for i in range(len(frames)):
        frames[i, chords[int(i * step / 2)]] = 0.9
    return frames


def test_reference_recording_is_found_in_a_video_with_an_intro_and_an_inserted_scene():
    from app.pipeline.lyrics import match_recordings, shift_at
    step = 0.1
    album = _recording(200)
    cut = int(120 / step)                                       # the video: 9 s of intro, the album up to 120 s,
    video = np.vstack([np.full((90, 25), 0.04), album[:cut],    # 4 s of something else, then the rest
                       np.full((40, 25), 0.04), album[cut:], np.full((100, 25), 0.04)])
    shifts = match_recordings(album, video, step, lo=-3.0, hi=26.0)
    assert shifts is not None
    assert abs(shift_at(shifts, 30.0) - 9.0) < 0.2 and abs(shift_at(shifts, 100.0) - 9.0) < 0.2
    assert abs(shift_at(shifts, 150.0) - 13.0) < 0.2 and abs(shift_at(shifts, 190.0) - 13.0) < 0.2


def test_another_recording_is_not_taken_for_the_reference():
    from app.pipeline.lyrics import match_recordings
    assert match_recordings(_recording(200, seed=1), _recording(220, seed=2), 0.1, lo=-3.0, hi=23.0) is None


def _listening(monkeypatch, heard_share: dict):
    """words.align replaced: a line is "heard" if its video time is one of heard_share's keys' truths."""
    from app.pipeline import lyrics, words
    import librosa
    monkeypatch.setattr(librosa, "load", lambda *a, **k: (np.zeros(16000, dtype=np.float32), 16000))
    monkeypatch.setattr(words, "language_of", lambda y, at=0.0: "en")

    def align(lines, y, sr, language=None, report=None):
        for i, ln in enumerate(lines):
            report.append({"line": i, "heard": ln["t"] in heard_share, "sure": 0.5, "late": 0.0})
        return [[ln["t"]] * len(ln["text"].split()) for ln in lines]
    monkeypatch.setattr(words, "align", align)


META, MATCH = {"duration": 220, "youtube_id": "v"}, {"duration": 200.0, "artistName": "A", "trackName": "B"}
TWO = [{"t": 10.0, "text": "one"}, {"t": 150.0, "text": "two words"}]


def test_times_that_are_heard_in_place_are_kept(tmp_path, monkeypatch):
    from app.pipeline import lyrics
    _listening(monkeypatch, {10.0, 150.0})
    monkeypatch.setattr(lyrics, "reference_shifts", lambda *a: 1 / 0)
    placed = lyrics.place(TWO, tmp_path, tmp_path / "a.wav", META, MATCH, None)
    assert placed["aligned"] == "same" and placed["heard"] == 1.0 and placed["words"] == "heard"
    assert placed["lines"] == [{"t": 10.0, "lrc_t": 10.0, "text": "one", "w": [10.0]},
                               {"t": 150.0, "lrc_t": 150.0, "text": "two words", "w": [150.0, 150.0]}]


def test_lines_not_heard_in_place_are_placed_by_the_reference_recording(tmp_path, monkeypatch):
    from app.pipeline import lyrics
    _listening(monkeypatch, {19.0, 163.0})
    monkeypatch.setattr(lyrics, "reference_shifts", lambda *a: [(15.0, 9.0), (145.0, 13.0)])
    placed = lyrics.place(TWO, tmp_path, tmp_path / "a.wav", META, MATCH, None)
    assert placed["aligned"] == "reference" and [ln["t"] for ln in placed["lines"]] == [19.0, 163.0]
    # placing again starts from the LRCLIB times, not from the placed ones
    again = lyrics.place(placed["lines"], tmp_path, tmp_path / "a.wav", META, MATCH, None)
    assert [ln["t"] for ln in again["lines"]] == [19.0, 163.0]


def test_same_length_but_wrong_times_are_fixed_by_ear(tmp_path, monkeypatch):
    from app.pipeline import lyrics, words
    _listening(monkeypatch, {34.0, 171.0})
    monkeypatch.setattr(lyrics, "reference_shifts", lambda *a: 1 / 0)          # same length: never asked
    monkeypatch.setattr(words, "transcribe", lambda y, language=None: [])
    monkeypatch.setattr(words, "find_lines", lambda lines, heard: ([24.0, 21.0], 1.0))
    placed = lyrics.place(TWO, tmp_path, tmp_path / "a.wav", {"duration": 201}, MATCH, None)
    assert placed["aligned"] == "heard" and [ln["t"] for ln in placed["lines"]] == [34.0, 171.0]


def test_without_a_model_the_words_still_get_times(tmp_path, monkeypatch):
    from app.pipeline import lyrics
    import librosa

    def broken(*a, **k):
        raise RuntimeError("no audio")
    monkeypatch.setattr(librosa, "load", broken)
    monkeypatch.setattr(lyrics, "reference_shifts", broken)
    monkeypatch.setattr(lyrics, "vocal_activity", lambda wav: (np.zeros(2200), 0.1))
    placed = lyrics.place(TWO, tmp_path, tmp_path / "a.wav", META, MATCH, None)
    assert placed["aligned"] == "activity" and placed["words"] == "syllables"
    assert all(len(ln["w"]) == len(ln["text"].split()) for ln in placed["lines"])
    same = lyrics.place(TWO, tmp_path, tmp_path / "a.wav", {"duration": 201}, MATCH, None)
    assert same["aligned"] == "same" and [ln["t"] for ln in same["lines"]] == [10.0, 150.0]


def test_passages_heard_as_no_chord_do_not_place_the_reference():
    from app.pipeline.lyrics import match_recordings, shift_at
    silent_album, silent_video = np.full((2000, 25), 0.01), np.full((2200, 25), 0.01)
    silent_album[:, 24] = silent_video[:, 24] = 1.0
    assert match_recordings(silent_album, silent_video, 0.1, lo=-3.0, hi=23.0) is None
    # a song with 60 s of "no chord" in the middle, 20 s later in the video: the middle follows its neighbours
    album = _recording(200)
    album[700:1300] = 0.01
    album[700:1300, 24] = 1.0
    intro = np.full((200, 25), 0.01)
    intro[:, 24] = 1.0
    shifts = match_recordings(album, np.vstack([intro, album]), 0.1, lo=-3.0, hi=23.0)
    assert shifts is not None and all(abs(shift - 20.0) < 0.2 for _, shift in shifts)
