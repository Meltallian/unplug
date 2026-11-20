# How a song is analysed, and how well

What happens between "audio downloaded" and "chart text written", what each part relies on, and
the numbers behind the choices. Code: `app/pipeline/` (`rhythm.py`, `btc.py`, `chords.py`,
`lyrics.py`), benchmark: `app/eval/bench.py`.

## The stages

1. **Beats and downbeats** (`rhythm.py`): the beat_this model gives beat and downbeat activations
   (cached per song in `cache/beat-activations-*.npz`). On top of it:
   - *one pulse* (`regular_beats`): the pulse is the beat length most of the song is counted in.
     Sections the tracker counted two or three times as fast lose the beats in between, slower
     ones and breaks are filled, and the grid is continued to the first and last second.
   - *bars* (`bar_grid`): a Viterbi grid over bars of 4, 3 or 6 beats. A bar may be cut short and
     the bar length may change at a bar line, each at a cost, so a 6/8 chorus and a 4/4 verse in
     the same song each get their bars. Where the downbeats cannot tell (which two bars of three
     make one 6/8 bar), the chord changes decide.
   - *what is counted*: beats in threes at 138 a minute or more are the eighth notes of 6/8, slower
     ones a waltz; in a song with 6/8 parts, the parts in four are 4/4 counted in the same eighths.
   - *tempo*: always that of the beat the time signature names (dotted quarter in 6/8), so tempo,
     time signature and bar length agree. No halving "because it looks fast": the tracker's level
     is kept, since no rule we tried agrees better with human annotators (see below).
2. **Chord probabilities** (`btc.py`): the BTC transformer (Park et al., ISMIR 2019, MIT licence,
   published large-vocabulary weights) gives, every 93 ms, a probability for 170 chords. The
   checkpoint is read with torch's restricted loader (tensors and plain numbers only).
3. **Chords on the grid** (`chords.py`): per beat, the model's probabilities are summed into the
   24 major and minor chords (a sus chord counts for both chords of its root) and a Viterbi pass
   lets chords change mostly on the "1", sometimes mid-bar, rarely elsewhere. Then each chord is
   written richer (7, m7, maj7, sus4, sus2, dim ...) only where the model gives that reading at
   least half of the chord's weight, each chord gets one spelling per song (Am7 or Am, not both),
   and in 6/8 nothing shorter than a counted beat survives.
4. **Lyrics** (`lyrics.py`): synced lyrics from LRCLIB; the shift between the lyrics' recording
   and the video is found from vocal-band activity, among the shifts that keep every sung line on
   music (where chords were found). Without that last condition a video outro pulled the lyrics of
   one song 23 s late.

The analysis result (`cache/chords.json`): `tempo`, `meter` (3 = 3/4, 4 = 4/4, 6 = 6/8, both at the
start of the song), `meters` (every change: start, meter, tempo), `downbeats` (all bar starts),
`key`, `capo`, `segments` (chord, start, end, beats).

## The benchmark

`make bench` (or `python -m app.eval.bench`) scores the analysis against the McGill Billboard
annotations: chords with times, bars, metre and tonic, made by trained annotators (CC0). The audio
is taken from YouTube and accepted only if its length matches the annotated recording within
2.5 s. `python -m app.eval.bench fetch` builds the sets: `map` (40 songs drawn at random, seed 42)
and `map-odd` (24 songs that are not in 4/4). Parameters were tuned on every second song of `map`
only; the other half and `map-odd` were not used for tuning. The chord model was not trained on
Billboard; the beat tracker was trained on many collections that may overlap with it.

Results of 2026-11-20, `main` (chroma templates, tempo halved above 135 bpm) against this version,
same songs, same alignment per song:

| set | version | root | maj/min | 7ths | segmentation | downbeats | bars right | tempo right | key right |
|---|---|---|---|---|---|---|---|---|---|
| `map`, 40 songs | main | 0.68 | 0.63 | 0.51 | 0.67 | 0.88 | 85 % | 68 % | 80 % |
| | this version | 0.87 | 0.87 | 0.75 | 0.83 | 0.91 | 90 % | 90 % | 83 % |
| `map-odd`, 24 songs | main | 0.74 | 0.69 | 0.51 | 0.76 | 0.80 | 46 % | 46 % | 71 % |
| | this version | 0.86 | 0.85 | 0.70 | 0.86 | 0.90 | 54 % | 83 % | 83 % |

- On the half of `map` used for tuning, maj/min is 0.873; on the half never used, 0.859.
- "bars right" and "tempo right": within 6 % of the annotators' bar length / tempo. "key right":
  the annotated tonic, or its relative major or minor (same key signature).
- The chord model alone (its most likely chord per frame, no beats, no bars) gives 0.83 maj/min on
  `map`; decoding on the bar grid adds the rest and puts the changes on the beat.
- On `map-odd` the time signature at the start is the annotated one on 8 songs of 24, and on 14 if
  a 12/8 written as 4/4 with the same bars counts; most others are 12/8 songs written as 6/8 with
  bars half as long (right tempo, right count in threes).
- A new song takes about 30 s on 2 vCPU and 0.9 GB of memory at the peak (main: 50 s, 1.2 GB).

What the columns mean is at the top of `app/eval/bench.py`.

## What the numbers do not say

- *Tempo level.* On a few songs the tracker counts twice as fast or as slowly as the annotators
  (same music, bars half or double: "The Joker" 83 against 166, "Goodbye Yellow Brick Road" 121
  against 60). The annotators do not always follow the usual tempo of a song either (they have
  "Help!" at 190, usually given as 95). Chord length in bars does not separate the cases, so the
  tracker's level is kept and the three values (tempo, signature, bars) are kept consistent.
- *12/8.* Slow 12/8 songs come out as 4/4 at the right tempo (tracker on the dotted quarter) or
  as 6/8 at the right tempo with bars half as long (tracker on the eighth). Both are playable;
  neither says "12/8".
- *Records a quarter tone off.* The nearest semitone is a toss-up; the benchmark then compares at
  the better of the two (`shift`), the chart shows the one the model chose.
- *Vocabulary.* No slash chords, no add9; sixths are written as plain chords.
- *Lyric timing* is per line (from LRCLIB) with one shift per song; a video that inserts a scene
  in the middle of the song still gets its later lines late.
