"""When is each word of a sung line sung, and are the lines where their times say?

LRCLIB gives one time per line. A chord belongs over the word that is sung when it changes, so
the words need times of their own. Two ways, both taking lines as [{"t": start, "text": ...}] in
time order and giving back, per line, one start time per word (text.split()):

  by_syllables()   no audio: a line lasts as long as its syllables need at the pace of the song
                   (not until the next line starts), and its words share it by syllables.
  align()          listens: Whisper is given the words and says where it hears each of them
                   (its cross-attention, the way it produces word timestamps). A line is "heard"
                   if that starts near the line's own time and the model finds the words likely;
                   the others fall back to by_syllables(). The share of lines heard tells whether
                   the line times fit this recording at all.

And when they do not fit (lyrics timed on the album, a video with an intro):

  transcribe()     what Whisper hears in the whole recording, word by word.
  find_lines()     where each line is in that: the shift to add to its time.
"""

import os
import re
from functools import lru_cache

import numpy as np

MAX_LINE = 8.0          # s: no sung line is longer
MIN_LINE = 0.8          # s: nor shorter
PACE_PERCENTILE = 30    # the pace of a song: seconds per syllable of its densest lines
BREATH = 1.25           # a line takes a little longer than its syllables at that pace
_VOWELS = re.compile(r"[aeiouyàâäáãåéèêëíîïóôöõúùûüœæ]+")

MODEL = os.environ.get("WHISPER_MODEL", "tiny")       # name or path of the Whisper checkpoint ("base" is
                                                      # twice as slow and no closer on our test songs)
WINDOW = 28.0           # s of audio given to the model at a time (it takes 30)
LEAD = 0.4              # s of audio before the first line of a window
TRUST = 1.0             # s: an aligned line must start this close to its own time for its word times to be used
SURE = 0.2              # a line counts as heard if the model also gives its words this probability on average
                        # (lines in the right place: 65 % pass both; lines 17 s away: 4 %)
SAME_WORDS = 0.75       # a line is found in the transcript where the text is this alike (0..1)
MIN_WORDS = 3           # shorter lines are found in too many places to say where they are
NEAR = 1.0              # s: shifts this close are the same shift
DRIFT = 5.0             # s: the shift may move this much between two lines at a small cost (lyrics timed
SMALL, BIG = 0.3, 3.0   # loosely); more is a cut or an inserted scene and must be earned by several
                        # lines found (a found line counts up to 1)


def syllables(word: str) -> int:
    """Rough syllable count of a word (vowel groups; a final mute "e" does not count)."""
    w = re.sub(r"[^\w']", "", word.lower())
    n = len(_VOWELS.findall(w))
    if n > 1 and re.search(r"[^aeiouy]es?$", w):
        n -= 1
    return max(1, n)


def by_syllables(lines: list[dict]) -> list[list[float]]:
    """Word start times from the line times alone."""
    counts = [[syllables(w) for w in ln["text"].split()] for ln in lines]
    gaps = [(lines[i + 1]["t"] - ln["t"]) if i + 1 < len(lines) else MAX_LINE for i, ln in enumerate(lines)]
    paces = [gap / sum(c) for gap, c in zip(gaps, counts) if c and 0.3 < gap < MAX_LINE]
    pace = float(np.percentile(paces, PACE_PERCENTILE)) if paces else 0.35
    out = []
    for ln, c, gap in zip(lines, counts, gaps):
        if not c:
            out.append([])
            continue
        length = min(max(gap, 0.3), MAX_LINE, max(MIN_LINE, sum(c) * pace * BREATH))
        before = np.concatenate([[0], np.cumsum(c)[:-1]])
        out.append([round(float(ln["t"] + length * b / sum(c)), 2) for b in before])
    return out


@lru_cache(maxsize=1)
def _whisper(name: str):
    import torch
    import whisper

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    return whisper.load_model(name, device="cpu")


def _windows(lines: list[dict], ends: list[float]) -> list[tuple[int, int]]:
    """Runs of consecutive lines [a, b) that fit in one window of audio."""
    out, a = [], 0
    while a < len(lines):
        b = a + 1
        while b < len(lines) and ends[b] - lines[a]["t"] + LEAD < WINDOW:
            b += 1
        out.append((a, b))
        a = b
    return out


def align(lines: list[dict], y: np.ndarray, sr: int, language: str | None = None, model: str | None = None,
          report: list | None = None) -> list[list[float]]:
    """Word start times heard in the audio (mono samples y at rate sr), see the module docstring.
    `report`, if given, receives one dict per line: was it heard, and how sure the model was."""
    import torch
    import whisper
    from whisper.audio import HOP_LENGTH, N_FRAMES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
    from whisper.timing import find_alignment
    from whisper.tokenizer import get_tokenizer

    guess = by_syllables(lines)
    if not any(guess):
        return guess
    net = _whisper(model or MODEL)
    if sr != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    y = y.astype(np.float32)
    if language is None and net.is_multilingual:     # ask the model which language is sung, on the first lines
        first = next(ln["t"] for ln, g in zip(lines, guess) if g)
        chunk = y[int(max(0.0, first - LEAD) * SAMPLE_RATE):][:30 * SAMPLE_RATE]
        with torch.no_grad():
            _, probs = net.detect_language(pad_or_trim(log_mel_spectrogram(chunk, net.dims.n_mels), N_FRAMES))
        language = max(probs, key=probs.get)
    tokenizer = get_tokenizer(net.is_multilingual, language=language or "en", task="transcribe")
    ends = [min(g[-1] + 1.5 if g else ln["t"] + 1.0, lines[i + 1]["t"] if i + 1 < len(lines) else np.inf)
            for i, (ln, g) in enumerate(zip(lines, guess))]
    out = [list(g) for g in guess]
    for a, b in _windows(lines, ends):
        start = max(0.0, lines[a]["t"] - LEAD)
        stop = min(len(y) / SAMPLE_RATE, ends[b - 1] + 0.5, start + 30.0)
        words = [w for ln in lines[a:b] for w in ln["text"].split()]
        if not words or stop - start < 0.5:
            continue
        text = " " + " ".join(words)
        tokens = tokenizer.encode(text)
        if len(tokens) > 400:                      # more text than the model can hold: keep the guess
            continue
        chunk = y[int(start * SAMPLE_RATE):int(stop * SAMPLE_RATE)]
        mel = pad_or_trim(log_mel_spectrogram(chunk, net.dims.n_mels), N_FRAMES)
        with torch.no_grad():
            heard = find_alignment(net, tokenizer, tokens, mel, num_frames=len(chunk) // HOP_LENGTH)
        # the model's "words" are pieces of our text in order: find the piece each of our words starts in
        starts, offset = [], 0
        for piece in heard:
            starts.append((offset, piece.start, piece.probability))
            offset += len(piece.word)
        offsets = np.array([o for o, *_ in starts])
        times, sure, at = [], [], 0
        for w in words:
            at = text.index(w, at)
            piece = starts[max(0, int(np.searchsorted(offsets, at, side="right")) - 1)]
            times.append(start + piece[1])
            sure.append(piece[2])
            at += len(w)
        k = 0
        for i in range(a, b):
            n = len(lines[i]["text"].split())
            got, prob = times[k:k + n], float(np.mean(sure[k:k + n])) if n else 0.0
            k += n
            late = got[0] - lines[i]["t"] if n else 0.0
            if n and abs(late) <= TRUST:               # starts where the line does: its word times are used
                last = lines[i + 1]["t"] + 0.3 if i + 1 < len(lines) else np.inf
                got = np.maximum.accumulate(np.clip(got, lines[i]["t"] - 0.3, last))
                out[i] = [round(float(t), 2) for t in got]
            if report is not None and n:               # ... and "heard" if the model also finds the words likely
                report.append({"line": i, "heard": abs(late) <= TRUST and prob >= SURE, "sure": prob, "late": late})
    return out


def language_of(y: np.ndarray, at: float = 0.0, model: str | None = None) -> str | None:
    """The language sung in the 30 s from `at` (y: mono samples at 16 kHz), as the model hears it."""
    import torch
    from whisper.audio import N_FRAMES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim

    net = _whisper(model or MODEL)
    if not net.is_multilingual:
        return None
    chunk = y[int(max(0.0, at) * SAMPLE_RATE):][:30 * SAMPLE_RATE].astype(np.float32)
    with torch.no_grad():
        _, probs = net.detect_language(pad_or_trim(log_mel_spectrogram(chunk, net.dims.n_mels), N_FRAMES))
    return max(probs, key=probs.get)


def transcribe(y: np.ndarray, language: str | None = None, model: str | None = None) -> list[tuple[float, str]]:
    """Every word the model hears in the recording (y: mono samples at 16 kHz): [(start, word)]."""
    net = _whisper(model or MODEL)
    result = net.transcribe(y.astype(np.float32), language=language, word_timestamps=True, fp16=False,
                            condition_on_previous_text=False, temperature=0.0, verbose=None)
    return [(float(w["start"]), w["word"]) for seg in result["segments"] for w in seg.get("words", [])]


def _plain(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def find_lines(lines: list[dict], heard: list[tuple[float, str]]) -> tuple[list[float], float] | None:
    """Where the lines are in what was heard: (the shift to add to each line's time, the share of
    lines found). The shift may change along the song (a video with a scene in the middle), at a
    cost, so that one line found in the wrong chorus does not move its neighbours. None if nothing
    is found."""
    from difflib import SequenceMatcher

    spoken = [(t, _plain(w)) for t, w in heard if _plain(w)]
    found: list[list[tuple[float, float]]] = []          # per line: [(shift, likeness)]
    for ln in lines:
        mine = [_plain(w) for w in ln["text"].split() if _plain(w)]
        here = []
        if len(mine) >= MIN_WORDS:
            text = " ".join(mine)
            matcher = SequenceMatcher(None, b=text, autojunk=False)
            for j in range(len(spoken) - len(mine) + 1):
                matcher.set_seq1(" ".join(w for _, w in spoken[j:j + len(mine)]))
                if matcher.quick_ratio() >= SAME_WORDS and (ratio := matcher.ratio()) >= SAME_WORDS:
                    here.append((spoken[j][0] - ln["t"], ratio))
        found.append(here)
    shifts = sorted({round(shift / NEAR) * NEAR for here in found for shift, _ in here})
    if not shifts:
        return None
    grid = np.array(shifts)

    def gain(here: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """For every candidate shift: how alike the best place of this line near it is, and its exact shift."""
        best, exact = np.zeros(len(grid)), grid.copy()
        for shift, ratio in here:
            near = np.abs(grid - shift) <= NEAR
            better = near & (ratio > best)
            best[better], exact[better] = ratio, shift
        return best, exact

    gains = [gain(here) for here in found]
    apart = np.abs(grid[:, None] - grid[None, :])
    cost = np.where(apart == 0, 0.0, np.where(apart <= DRIFT, SMALL, BIG))
    score = gains[0][0] - 1e-4 * np.abs(grid)            # between equals, the smaller shift
    back = np.zeros((len(lines), len(grid)), dtype=int)
    for i in range(1, len(lines)):                       # Viterbi: stay on a shift, or change it at a cost
        # ... and never so that this line would be sung before the one above it
        order = (lines[i]["t"] + grid[None, :]) - (lines[i - 1]["t"] + grid[:, None]) >= -NEAR
        via = np.where(order, score[:, None] - cost, -np.inf)
        back[i] = np.argmax(via, axis=0)
        score = via[back[i], np.arange(len(grid))] + gains[i][0]
    state = int(np.argmax(score))
    path = [state]
    for i in range(len(lines) - 1, 0, -1):
        state = int(back[i, state])
        path.append(state)
    path.reverse()
    hit = [bool(gains[i][0][k] > 0) for i, k in enumerate(path)]
    out = []
    for i, k in enumerate(path):
        if not hit[i]:          # a line not found takes the shift of the nearest found line on the same shift
            near = [j for j in range(len(path)) if hit[j] and path[j] == k]
            j = min(near, key=lambda j: abs(j - i)) if near else None
            out.append(float(gains[j][1][k]) if j is not None else float(grid[k]))
        else:
            out.append(float(gains[i][1][k]))
    return out, float(np.mean(hit))
