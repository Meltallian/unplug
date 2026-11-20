from types import SimpleNamespace

import numpy as np

from app.pipeline import words


def test_syllables():
    assert [words.syllables(w) for w in ("the", "take", "river", "yesterday", "tomorrow", "drowning", "c'est", "bohème", "I'm")] == \
        [1, 1, 2, 3, 3, 2, 1, 2, 1]


def test_a_line_lasts_as_long_as_its_syllables_need_not_until_the_next_line():
    lines = [{"t": 0.0, "text": "la la la la"}, {"t": 2.0, "text": "la la la la"},      # 0.5 s a syllable
             {"t": 4.0, "text": "la la la la"}, {"t": 6.0, "text": "la la"},            # ... then a long wait
             {"t": 14.0, "text": "la la la la"}]
    times = words.by_syllables(lines)
    assert times[0] == [0.0, 0.5, 1.0, 1.5]                       # a dense line fills its gap
    assert times[3][1] < 6.0 + 1.0                                # the second word does not wait 4 s
    assert [len(t) for t in times] == [4, 4, 4, 2, 4] and words.by_syllables([{"t": 1.0, "text": ""}]) == [[]]


def _fake_whisper(monkeypatch, heard_at):
    """Whisper replaced by a stand-in that "hears" the words where heard_at(word index) says."""
    import whisper.timing

    def find_alignment(net, tokenizer, tokens, mel, num_frames):
        pieces = tokenizer.decode(tokens).split(" ")[1:]
        return [SimpleNamespace(word=" " + p, start=heard_at(i), probability=0.8) for i, p in enumerate(pieces)]

    monkeypatch.setattr(whisper.timing, "find_alignment", find_alignment)
    monkeypatch.setattr(words, "_whisper", lambda name: SimpleNamespace(is_multilingual=False, dims=SimpleNamespace(n_mels=80)))


def test_words_take_the_times_the_model_hears(monkeypatch):
    lines = [{"t": 10.0, "text": "Help me get out"}, {"t": 15.0, "text": "I can't keep up"}]
    # relative to the window start (9.6 s): line one at 10.0, 10.2, 11.5, 12.0; line two at 15.1 ...
    heard = [0.4, 0.6, 1.9, 2.4, 5.5, 5.8, 6.3, 7.0]
    _fake_whisper(monkeypatch, lambda i: heard[i])
    times = words.align(lines, np.zeros(16000 * 25, dtype=np.float32), 16000)
    assert times == [[10.0, 10.2, 11.5, 12.0], [15.1, 15.4, 15.9, 16.6]]


def test_a_line_heard_far_from_its_own_time_keeps_the_guess(monkeypatch):
    lines = [{"t": 10.0, "text": "Help me get out"}, {"t": 15.0, "text": "I can't keep up"}]
    heard = [0.4, 0.6, 1.9, 2.4, 9.0, 9.2, 9.4, 9.6]             # line two "heard" 3.6 s late: not believed
    _fake_whisper(monkeypatch, lambda i: heard[i])
    times = words.align(lines, np.zeros(16000 * 25, dtype=np.float32), 16000)
    assert times[0] == [10.0, 10.2, 11.5, 12.0] and times[1] == words.by_syllables(lines)[1]


def test_lines_are_found_in_what_is_heard_each_with_its_own_shift():
    lines = [{"t": 10.0, "text": "Today is gonna be the day"}, {"t": 16.0, "text": "that they're gonna throw it back to you"},
             {"t": 22.0, "text": "la la"}, {"t": 30.0, "text": "by now you should've somehow"}]
    said = "today he's gonna be the day that there gonna throw it back to you la la by now you shut up somehow".split()
    at = [34.0, 34.4, 34.8, 35.2, 35.6, 36.0,  39.5, 39.9, 40.3, 40.7, 41.1, 41.5, 41.9, 42.3,  45.0, 45.5,  51.0, 51.3, 51.6, 51.9, 52.2, 52.5]
    shifts, share = words.find_lines(lines, list(zip(at, said)))
    assert abs(shifts[0] - 24.0) < 0.1 and abs(shifts[1] - 23.5) < 0.1 and abs(shifts[3] - 21.0) < 0.5
    assert shifts[2] in (shifts[1], shifts[3]) and share == 0.75          # the short line follows a neighbour
    assert words.find_lines(lines, [(1.0, "something"), (2.0, "else"), (3.0, "entirely")]) is None


def test_a_chorus_is_not_found_in_another_chorus_before_the_lines_above_it():
    chorus = "help me get out of it"
    lines = [{"t": 10.0, "text": "it is hard to take and hard to break"}, {"t": 20.0, "text": chorus},
             {"t": 60.0, "text": "tried to make amends and tried to pretend"}, {"t": 70.0, "text": chorus}]
    heard = []
    for start, text in ((15.0, lines[0]["text"]), (25.0, chorus), (65.0, lines[2]["text"])):   # the second chorus is not heard
        heard += [(start + 0.4 * k, w) for k, w in enumerate(text.split())]
    shifts, share = words.find_lines(lines, heard)
    assert [round(s) for s in shifts] == [5, 5, 5, 5] and share == 0.75    # not 25 - 70 = -45 for the last line
