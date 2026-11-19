"""`make eval`: run every fixture through the cached pipeline and print a score table.

Scores are tolerant because the reference sheets are hand-made (see the spec):
  key      exact / rel (relative major-minor) / off
  shift    transposition (semitones) that best aligns detected and sheet chords; 0 is ideal
  roots    Jaccard overlap of chord roots (sheet vs detected, ignoring major/minor)
  chords   Jaccard overlap of chords (root + major/minor)
  trans    F1 of chord-to-chord transitions (order-aware, start-agnostic)
Reference-free checks (also for songs without a sheet):
  inkey    share of time on chords that fit the detected key
  short    share of chord changes lasting under 2 beats
  chg/min  chord changes per minute
"""

import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.eval.sheet import parse_sheet
from app.music import Chord, chord_name, diatonic, key_from_chords, key_name, same_scale, transpose
from app.pipeline.audio import decode
from app.pipeline.chords import VERSION, analyze, load_segments
from app.youtube import parse_youtube_id

MIN_SHARE = 0.02  # detected chords covering less time than this are ignored for set scores


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _transitions(seq: list) -> set:
    seq = [c for c in seq if c != "N"]
    collapsed = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    return set(zip(collapsed, collapsed[1:]))


def _f1(ref: set, hyp: set) -> float:
    if not ref or not hyp:
        return 0.0
    tp = len(ref & hyp)
    return 2 * tp / (len(ref) + len(hyp))


def score_song(fixture: Path, songs_dir: Path) -> dict:
    src = json.loads((fixture / "source.json").read_text())
    youtube_id = parse_youtube_id(src["youtube_url"])
    song_dir = songs_dir / youtube_id
    if not (song_dir / "meta.json").exists():
        return {"song": src["song"], "error": "audio not downloaded"}
    result = analyze(decode(song_dir), song_dir / "cache" / "chords.json")
    segs = load_segments(result)
    det_key = (result["key"]["tonic"], result["key"]["mode"])

    total = sum(e - s for _, s, e, _ in segs) or 1.0
    dur: dict = {}
    for c, s, e, _ in segs:
        dur[c] = dur.get(c, 0) + (e - s)
    det_chords = {c for c, d in dur.items() if c != "N" and d / total >= MIN_SHARE}
    voiced = [(c, s, e, b) for c, s, e, b in segs if c != "N"]
    voiced_time = sum(e - s for _, s, e, _ in voiced) or 1.0
    in_key = sum(e - s for c, s, e, _ in voiced if c in diatonic(det_key)) / voiced_time
    short = sum(1 for *_, b in voiced if b < 2) / (len(voiced) or 1)
    row = {
        "song": src["song"],
        "key": key_name(det_key),
        "capo": result["capo"],
        "inkey": in_key,
        "short": short,
        "chg/min": len(voiced) / (total / 60),
        "detected": " ".join(chord_name(c) for c in sorted(det_chords, key=lambda c: -dur[c])),
    }

    sheet_name = src.get("primary_sheet")
    if not sheet_name:
        return row
    version = src.get("versions", {}).get(sheet_name, {})
    sheet = parse_sheet((fixture / sheet_name).read_text(), capo=version.get("capo") or 0)
    ref_seq = sheet.chords
    ref_chords = {c for c in ref_seq if c != "N"}
    ref_key = key_from_chords([(c, 1.0) for c in ref_seq if c != "N"])
    det_seq = [c for c, *_ in segs]

    def chord_score(shift: int) -> float:
        return _jaccard({transpose(c, shift) for c in ref_chords}, det_chords)
    shift = max(range(-6, 6), key=lambda k: (chord_score(k), -abs(k)))

    row.update({
        "sheet_key": key_name(ref_key),
        "key_match": "exact" if ref_key == det_key else ("rel" if same_scale(ref_key, det_key) else "off"),
        "shift": shift,
        "roots": _jaccard({c[0] for c in ref_chords}, {c[0] for c in det_chords}),
        "chords": chord_score(0),
        "trans": _f1(_transitions(ref_seq), _transitions(det_seq)),
        "sheet": " ".join(chord_name(c) for c in sorted(ref_chords)),
    })
    return row


COLUMNS = ["key_match", "shift", "roots", "chords", "trans", "inkey", "short", "chg/min"]
SUMMED = ["roots", "chords", "trans", "inkey", "short"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return "" if v is None else str(v)


def main() -> int:
    settings = get_settings()
    fixtures_dir = settings.data_dir / "fixtures"
    runs_dir = settings.data_dir / "eval"
    runs_dir.mkdir(exist_ok=True)

    rows = []
    for fixture in sorted(p for p in fixtures_dir.iterdir() if (p / "source.json").exists()):
        t0 = time.time()
        row = score_song(fixture, settings.songs_dir)
        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)

    width = max(len(r["song"]) for r in rows) + 2
    print(f"\nchord stage: {VERSION}\n")
    print("song".ljust(width) + "key   sheet  " + "".join(c.rjust(10) for c in COLUMNS))
    for r in rows:
        if "error" in r:
            print(r["song"].ljust(width) + r["error"])
            continue
        print(r["song"].ljust(width) + r["key"].ljust(6) + r.get("sheet_key", "-").ljust(7)
              + "".join(_fmt(r.get(c, "-")).rjust(10) for c in COLUMNS))
    scored = [r for r in rows if "chords" in r]
    means = {c: sum(r[c] for r in scored) / len(scored) for c in SUMMED if scored} if scored else {}
    means["inkey"] = sum(r["inkey"] for r in rows if "inkey" in r) / max(1, sum("inkey" in r for r in rows))
    means["short"] = sum(r["short"] for r in rows if "short" in r) / max(1, sum("short" in r for r in rows))
    print("\nmean (" + str(len(scored)) + " with sheets): " + "  ".join(f"{k}={v:.2f}" for k, v in means.items()))
    print("\ndetected vs sheet chords:")
    for r in scored:
        print(f"  {r['song']}: detected [{r['detected']}]  sheet [{r['sheet']}]  capo {r['capo']}")

    previous = sorted(runs_dir.glob("*.json"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = runs_dir / f"{stamp}-{VERSION}.json"
    out.write_text(json.dumps({"version": VERSION, "rows": rows, "means": means}, indent=1))
    if previous:
        prev = json.loads(previous[-1].read_text())
        print(f"\nchange vs {previous[-1].name}: " + "  ".join(
            f"{k} {means[k] - prev['means'].get(k, 0):+.2f}" for k in means if k in prev["means"]))
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
