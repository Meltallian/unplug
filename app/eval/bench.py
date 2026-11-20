"""`python -m app.eval.bench`: score the analysis against public, time-aligned annotations.

Reference: the McGill Billboard annotations (chords with times, bars, metre, tonic; CC0). They sit
in DATA_DIR/ref/billboard (see `fetch` below); the audio comes from YouTube like any other song,
matched by duration, and is never part of the repository.

Per song:
  root, majmin   share of time on the right chord root / right major-minor chord (mir_eval's
                 weighted chord symbol recall), after sliding the audio by `off` seconds
  7ths           same, counting sevenths (C7 is not C)
  shift          semitones the estimate was moved to compare (0 normally; +1 or -1 for a record that
                 sits a quarter tone off, where the nearest semitone is a toss-up; the score is then
                 the better of the two readings)
  seg            do chord changes fall where the annotated ones do (1 = same segmentation)
  bars           estimated bar length / annotated bar length (1.00 = as many bars as the annotators)
  tempo          estimated / annotated tempo of the notated beat (quarter, or dotted quarter in 6/8)
  time           time signature found (annotated)
  down           share of annotated phrase starts that fall on an estimated bar line
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.pipeline.rhythm import time_signature

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
STEP = 0.05          # s, grid used to slide the audio against the annotation
MAX_SHIFT = 8.0      # s
MAX_SCALE = 1.03     # the audio may run up to 3 % faster or slower than the annotated recording
BEATS = {"4/4": 4, "3/4": 3, "2/4": 2, "6/8": 2, "12/8": 4, "9/8": 3, "5/4": 5, "7/4": 7, "6/4": 6}
_INLINE_METRE = re.compile(r"\(\d+/\d+\)")
_REPEAT = re.compile(r"\bx(\d+)\b")


def parse_salami(text: str) -> dict:
    """Metre, tonic, end time and phrases [(start, end, bars, metre)] of a salami_chords.txt file.
    Phrases with a metre change inside are left out (their bars are not all the same length)."""
    metre, first_metre, tonic = "4/4", None, None
    rows: list[tuple[float, str, str]] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if m := re.match(r"#\s*metre:\s*(\S+)", line):
                metre = m.group(1)
                first_metre = first_metre or metre
            elif m := re.match(r"#\s*tonic:\s*(\S+)", line):
                tonic = tonic or m.group(1)
            continue
        if "\t" in line:
            t, content = line.split("\t", 1)
            rows.append((float(t), content.strip(), metre))
    phrases = []
    for (t0, content, m), (t1, _, _) in zip(rows, rows[1:]):
        if "|" not in content or _INLINE_METRE.search(content):
            continue
        inner = content[content.index("|") + 1:content.rindex("|")]
        bars = len([c for c in inner.split("|") if c.strip()])
        tail = content[content.rindex("|") + 1:]
        if r := _REPEAT.search(tail):
            bars *= int(r.group(1))
        if bars and t1 > t0:
            phrases.append((t0, t1, bars, m))
    end = next((t for t, c, _ in rows if c == "end"), rows[-1][0] if rows else 0.0)
    return {"metre": first_metre or "4/4", "tonic": tonic, "phrases": phrases, "end": end}


def reference_rhythm(salami: dict) -> dict:
    """Typical bar length (s), tempo of the notated beat and bar start times of the main metre."""
    main = [(t0, t1, n) for t0, t1, n, m in salami["phrases"] if m == salami["metre"]]
    lengths = np.repeat([(t1 - t0) / n for t0, t1, n in main], [n for *_, n in main])
    bar = float(np.median(lengths)) if len(lengths) else None
    starts = [t0 for t0, _, _ in main]
    return {"bar": bar, "tempo": BEATS.get(salami["metre"], 4) * 60 / bar if bar else None,
            "phrase_starts": starts, "metre": salami["metre"]}


def load_lab(path: Path) -> tuple[np.ndarray, list[str]]:
    """(intervals, labels) of a .lab file; blank or broken lines (some files end with one) are skipped."""
    intervals, labels = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3:
            intervals.append([float(parts[0]), float(parts[1])])
            labels.append(parts[2])
    return _contiguous(np.array(intervals)), labels


def _contiguous(intervals: np.ndarray) -> np.ndarray:
    """Close hairline gaps and overlaps (rounding in the files) between consecutive intervals."""
    out = intervals.copy()
    close = np.abs(out[1:, 0] - out[:-1, 1]) < 1e-6
    out[1:, 0] = np.where(close, out[:-1, 1], out[1:, 0])
    return out


def harte(chord) -> str:
    """Our chord (None/"N" or [root, quality, ...]) in the syntax mir_eval reads."""
    if chord is None or chord == "N":
        return "N"
    return f"{SHARP_NAMES[chord[0]]}:{chord[1]}"


def _grid(intervals: np.ndarray, labels: list[str], duration: float) -> np.ndarray:
    """Chord id (root*2 + minor, -1 = none) every STEP seconds."""
    import mir_eval

    roots, bitmaps, _ = mir_eval.chord.encode_many(labels)
    ids = np.where(roots < 0, -1, roots * 2 + ((bitmaps[:, 3] == 1) & (bitmaps[:, 4] == 0)))
    t = np.arange(0, duration, STEP) + STEP / 2
    idx = np.searchsorted(intervals[:, 0], t, side="right") - 1
    ok = (idx >= 0) & (t < intervals[np.clip(idx, 0, len(intervals) - 1), 1])
    return np.where(ok, ids[np.clip(idx, 0, len(ids) - 1)], -1)


def align(ref_iv, ref_lab, est_iv, est_lab) -> tuple[float, float]:
    """(scale, offset) such that annotation time = scale × audio time + offset agrees best.

    The YouTube upload rarely starts exactly like the annotated recording, and transfers of old
    records often run a little faster or slower."""
    duration = float(max(ref_iv[-1, 1], est_iv[-1, 1] * MAX_SCALE)) + MAX_SHIFT
    ref = _grid(ref_iv, ref_lab, duration)
    shifts = range(-int(MAX_SHIFT / STEP), int(MAX_SHIFT / STEP) + 1)

    def search(scales, shifts) -> tuple[float, float, int]:
        best = (-1.0, 1.0, 0)
        for scale in scales:
            est = _grid(est_iv * scale, est_lab, duration)
            for k in shifts:
                a, b = (ref[k:], est[:len(est) - k]) if k >= 0 else (ref[:k], est[-k:])
                agree = float(np.sum((a == b) & (a >= 0)))
                # prefer no stretch and no shift unless the gain is real (ten steps of agreement)
                agree -= 10 * (abs(scale - 1) > 1e-6) + 0.01 * abs(k)
                if agree > best[0]:
                    best = (agree, float(scale), k)
        return best

    _, scale, k = search(1 + 0.0025 * np.arange(-12, 13), shifts)                # coarse, then fine
    _, scale, k = search(scale + 0.0005 * np.arange(-4, 5), range(k - 6, k + 7))
    return scale, k * STEP


def _moved(chord, semitones: int):
    return chord if chord is None or chord == "N" or not semitones else [(chord[0] + semitones) % 12, *chord[1:]]


def _agreement(ref_iv, ref_lab, est_iv, est_lab) -> float:
    """Share of the annotated time on which both name the same major/minor chord."""
    duration = float(ref_iv[-1, 1])
    ref, est = _grid(ref_iv, ref_lab, duration), _grid(est_iv, est_lab, duration)
    return float(np.mean((ref == est) & (ref >= 0)))


def score(analysis: dict, bb_dir: Path, alignment: tuple[float, float, int] | None = None) -> dict:
    """Scores of one analysis. `alignment` (scale, offset, semitone shift) to reuse one found before."""
    import mir_eval

    salami = parse_salami((bb_dir / "salami_chords.txt").read_text())
    ref = reference_rhythm(salami)
    ref_iv, ref_lab = load_lab(bb_dir / "majmin7.lab")
    segs = analysis["segments"]
    est_iv = np.array([[s["start"], s["end"]] for s in segs])
    est_lab = [harte(s["chord"]) for s in segs]
    semis = 0
    if alignment is None:
        candidates = []
        for k in (0, 1, -1):                       # a quarter tone off: try the neighbouring semitones too
            lab = [harte(_moved(s["chord"], k)) for s in segs]
            sc, off = align(ref_iv, ref_lab, est_iv, lab)
            candidates.append((_agreement(ref_iv, ref_lab, est_iv * sc + off, lab) - 0.02 * abs(k), k, sc, off))
        _, semis, scale, offset = max(candidates)
    else:
        scale, offset, semis = alignment
    est_lab = [harte(_moved(s["chord"], semis)) for s in segs]
    shifted = _contiguous(np.clip(est_iv * scale + offset, 0, None))
    keep = shifted[:, 1] > shifted[:, 0]
    res = mir_eval.chord.evaluate(ref_iv, ref_lab, shifted[keep], [l for l, k in zip(est_lab, keep) if k])
    key = "-"
    if salami["tonic"] and analysis.get("key"):
        from app.music import pc
        tonic, est = pc(salami["tonic"]), analysis["key"]
        est_tonic = (est["tonic"] + semis) % 12
        relative = (est_tonic + (3 if est["mode"] == "minor" else 9)) % 12
        key = "exact" if est_tonic == tonic else ("rel" if relative == tonic else "off")
    row = {"off": round(offset, 2), "scale": round(scale, 4), "shift": semis, "key": key, "root": res["root"], "majmin": res["majmin"], "7ths": res["sevenths"],
           "seg": res["seg"], "time": time_signature(analysis.get("meter", 4)), "ref_time": ref["metre"]}
    downs = np.asarray(analysis.get("downbeats") or [])
    if ref["bar"] and len(downs) > 2:
        downs = downs * scale + offset                                         # in annotation time
        est_bar = float(np.median(np.diff(downs)))
        row["bars"] = est_bar / ref["bar"]
        row["tempo"] = analysis["tempo"] / scale / ref["tempo"]
        tol = 0.25 * ref["bar"] / max(BEATS.get(ref["metre"], 4), 1)          # a quarter of a beat
        starts = np.array(ref["phrase_starts"])
        starts = starts[(starts >= downs[0] - tol) & (starts <= downs[-1] + tol)]
        if len(starts):
            row["down"] = float(np.mean([np.min(np.abs(downs - s)) <= tol for s in starts]))
    return row


COLUMNS = ["root", "majmin", "7ths", "seg", "bars", "tempo", "down"]


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))


def summarise(rows: list[dict]) -> dict:
    out = {c: float(np.mean([r[c] for r in rows if r.get(c) is not None])) for c in ("root", "majmin", "7ths", "seg", "down")
           if any(r.get(c) is not None for r in rows)}
    for c in ("bars", "tempo"):      # share of songs within 6 % of the annotation
        vals = [r[c] for r in rows if r.get(c) is not None]
        if vals:
            out[c + "_ok"] = float(np.mean([abs(v - 1) <= 0.06 for v in vals]))
    out["time_ok"] = float(np.mean([r["time"] == r["ref_time"] for r in rows]))
    out["key_ok"] = float(np.mean([r["key"] in ("exact", "rel") for r in rows]))
    return out


def load_set(name: str) -> list[dict]:
    """Songs of a benchmark set: map.json written by the fetch script ({billboard id: {youtube_id, ...}})."""
    bb = get_settings().data_dir / "ref" / "billboard"
    mapping = json.loads((bb / f"{name}.json").read_text())
    return [{"bb": k, **v, "dir": bb / "McGill-Billboard" / k} for k, v in sorted(mapping.items()) if not k.startswith("_")]


def main(argv: list[str]) -> int:
    from app.pipeline.audio import decode
    from app.pipeline.chords import VERSION, analyze, with_overrides

    settings = get_settings()
    args = [a for a in argv if "=" not in a]
    overrides = {k: float(v) for k, v in (a.split("=", 1) for a in argv if "=" in a)}
    params = with_overrides(overrides) if overrides else None
    set_name = args[0] if args else "map"
    rows = []
    for song in load_set(set_name):
        song_dir = settings.songs_dir / song["youtube_id"]
        if not (song_dir / "meta.json").exists():
            continue
        analysis = analyze(decode(song_dir), song_dir / "cache" / "chords.json", params)
        rows.append({"song": f"{song['artist']} - {song['title']}"[:38], **score(analysis, song["dir"])})
    if not rows:
        print("no songs: fetch the annotations and the audio first")
        return 1
    label = VERSION + ("+" + ",".join(f"{k}={v:g}" for k, v in overrides.items()) if overrides else "")
    print(f"\n{label} on {set_name} ({len(rows)} songs)\n")
    print("song".ljust(40) + "".join(c.rjust(8) for c in ["off"] + COLUMNS) + "  time (annotated)")
    for r in rows:
        print(r["song"].ljust(40) + "".join(_fmt(r.get(c)).rjust(8) for c in ["off"] + COLUMNS)
              + f"  {r['time']} ({r['ref_time']})")
    means = summarise(rows)
    print("\nmean: " + "  ".join(f"{k}={v:.3f}" for k, v in means.items()))
    if not overrides:
        out = settings.data_dir / "eval"
        out.mkdir(exist_ok=True)
        path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-bench-{set_name}-{VERSION}.json"
        path.write_text(json.dumps({"version": VERSION, "set": set_name, "rows": rows, "means": means}, indent=1))
        print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
