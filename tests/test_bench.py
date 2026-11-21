import numpy as np

from app.eval import bench

SALAMI = """# title: Example
# artist: Somebody
# metre: 6/8
# tonic: A

0.0\tsilence
1.0\tA, intro, | A:min | A:min | D:min | D:min |, (piano)
9.0\tB, verse, | A:min | F:maj | x2
17.0\t| (2/4) E:7 | A:min |
20.0\tsilence
22.0\tend
"""


def test_salami_file_gives_metre_tonic_and_bars():
    s = bench.parse_salami(SALAMI)
    assert (s["metre"], s["tonic"], s["end"]) == ("6/8", "A", 22.0)
    assert s["phrases"] == [(1.0, 9.0, 4, "6/8"), (9.0, 17.0, 4, "6/8")]     # x2 counted, odd-metre phrase left out
    ref = bench.reference_rhythm(s)
    assert ref["bar"] == 2.0 and ref["tempo"] == 60.0                         # two dotted quarters per 2 s bar


def _analysis(segments, downbeats, tempo=60.0, meter=6):
    return {"segments": [{"chord": c, "start": a, "end": b} for c, a, b in segments], "downbeats": downbeats,
            "tempo": tempo, "meter": meter, "key": {"tonic": 9, "mode": "minor"}}


def test_score_finds_the_offset_and_counts_right_chords(tmp_path):
    (tmp_path / "salami_chords.txt").write_text(SALAMI)
    (tmp_path / "majmin7.lab").write_text("0.0 1.0 N\n1.0 5.0 A:min\n5.0 9.0 D:min\n9.0 11.0 A:min\n11.0 13.0 F:maj\n"
                                          "13.0 15.0 A:min\n15.0 17.0 F:maj\n17.0 18.0 E:7\n18.0 20.0 A:min\n20.0 22.0 N\n\n")
    # the same music, starting 0.5 s later in "our" audio, with one wrong chord (F heard as D minor)
    est = [(None, 0.0, 1.5), ([9, "min"], 1.5, 5.5), ([2, "min"], 5.5, 9.5), ([9, "min"], 9.5, 11.5), ([2, "min"], 11.5, 13.5),
           ([9, "min"], 13.5, 15.5), ([5, "maj"], 15.5, 17.5), ([4, "7"], 17.5, 18.5), ([9, "min"], 18.5, 20.5), (None, 20.5, 22.5)]
    row = bench.score(_analysis(est, [1.5 + 2.0 * i for i in range(10)]), tmp_path)
    assert row["off"] == -0.5 and row["shift"] == 0 and row["key"] == "exact"
    assert abs(row["majmin"] - 17 / 19) < 0.02                                # 2 s wrong out of 19 s of chords
    assert abs(row["bars"] - 1.0) < 0.01 and abs(row["tempo"] - 1.0) < 0.01 and row["down"] == 1.0
    assert (row["time"], row["ref_time"]) == ("6/8", "6/8")


def test_chords_are_written_the_way_mir_eval_reads_them():
    assert bench.harte(None) == "N" and bench.harte([6, "min7"]) == "F#:min7" and bench.harte([0, "maj"]) == "C:maj"
