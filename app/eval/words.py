"""Score the word timing against songs whose words were timed by hand.

  python -m app.eval.words fetch [N]    get N English and N French songs (default 12 each)
  python -m app.eval.words [method ...] score: `chars` (words by position in the text until the
                                        next line, what the chart did before), `syllables`,
                                        `heard` (Whisper, the default of the pipeline)

Reference: JamendoLyrics MultiLang (github.com/f90/jamendolyrics): free-licence recordings with
the start and end of every word. They go to DATA_DIR/ref/jamendo and are never part of the
repository. Given to each method, as LRCLIB would: every line's text and start time. Scored: the
error on the start of every word, first words of lines left out (their time is given).
"""

import csv
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.pipeline import words

RAW = "https://raw.githubusercontent.com/f90/jamendolyrics/master/"
LANGUAGES = {"English": "en", "French": "fr"}
MAX_LINE = 8.0


def _root() -> Path:
    return get_settings().data_dir / "ref" / "jamendo"


def fetch(per_language: int = 12, seed: int = 7) -> int:
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    index = root / "JamendoLyrics.csv"
    if not index.exists():
        urllib.request.urlretrieve(RAW + "JamendoLyrics.csv", index)
    rows = [r for r in csv.DictReader(open(index)) if r["Genre"] != "Hip-Hop" and r["Language"] in LANGUAGES]
    random.Random(seed).shuffle(rows)
    want = dict.fromkeys(LANGUAGES, per_language)
    for r in rows:
        if want[r["Language"]] <= 0:
            continue
        name = r["Filepath"][:-4]
        for folder, ext in (("mp3", ".mp3"), ("annotations/words", ".csv"), ("annotations/lines", ".csv")):
            target = root / folder / (name + ext)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                urllib.request.urlretrieve(RAW + folder + "/" + urllib.parse.quote(name + ext), target)
        want[r["Language"]] -= 1
        print(r["Language"], name)
        time.sleep(1)
    return 0


def load(name: str) -> list[dict]:
    """Lines of a song: {"t", "text", "true": the annotated start of every word}."""
    root = _root()
    lines = list(csv.DictReader(open(root / "annotations" / "lines" / f"{name}.csv")))
    starts = [float(w["word_start"]) for w in csv.DictReader(open(root / "annotations" / "words" / f"{name}.csv"))]
    out, k = [], 0
    for ln in lines:
        n = len(ln["lyrics_line"].split())
        out.append({"t": float(ln["start_time"]), "text": ln["lyrics_line"], "true": starts[k:k + n]})
        k += n
    return out


def by_chars(lines: list[dict]) -> list[list[float]]:
    """Words spread by their position in the text over the time until the next line (8 s at most)."""
    out = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["t"] if i + 1 < len(lines) else ln["t"] + MAX_LINE
        span = max(min(nxt, ln["t"] + MAX_LINE) - ln["t"], 0.5)
        at, times = 0, []
        for w in ln["text"].split():
            at = ln["text"].index(w, at)
            times.append(ln["t"] + span * at / max(1, len(ln["text"])))
            at += len(w)
        out.append(times)
    return out


def _audio(name: str) -> np.ndarray:
    import librosa

    root = _root()
    wav = root / "wav" / f"{name}.wav"
    if not wav.exists():
        wav.parent.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(root / "mp3" / f"{name}.mp3"),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return librosa.load(wav, sr=None, mono=True)[0]


def main(argv: list[str]) -> int:
    if argv and argv[0] == "fetch":
        return fetch(int(argv[1]) if len(argv) > 1 else 12)
    root = _root()
    methods = argv or ["chars", "syllables", "heard"]
    songs = [r for r in csv.DictReader(open(root / "JamendoLyrics.csv"))
             if (root / "mp3" / r["Filepath"]).exists() and r["Language"] in LANGUAGES] if (root / "JamendoLyrics.csv").exists() else []
    if not songs:
        print("no songs: run `python -m app.eval.words fetch` first")
        return 1
    errors: dict[str, dict[str, list[np.ndarray]]] = {m: {} for m in methods}
    for r in songs:
        name = r["Filepath"][:-4]
        lines = load(name)
        given = [{"t": ln["t"], "text": ln["text"]} for ln in lines]
        for m in methods:
            if m == "chars":
                times = by_chars(given)
            elif m == "syllables":
                times = words.by_syllables(given)
            else:
                times = words.align(given, _audio(name), 16000, LANGUAGES[r["Language"]])
            err = np.array([abs(p - t) for ln, ts in zip(lines, times) for p, t in zip(ts[1:], ln["true"][1:])])
            errors[m].setdefault(r["Language"], []).append(err)
    print(f"\n{len(songs)} songs\n")
    print("method      language  words  median   within 0.3 s  within 1 s")
    for m in methods:
        for language in [*LANGUAGES, "all"]:
            err = np.concatenate([e for lang, es in errors[m].items() if language in (lang, "all") for e in es])
            print(f"{m:11} {language:8} {len(err):6}  {np.median(err):.2f} s  {np.mean(err <= 0.3):11.0%}  {np.mean(err <= 1.0):9.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
