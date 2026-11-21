import re
import xml.etree.ElementTree as ET

import pytest

from app.pipeline import score
from app.pipeline.chart import read_score
from app.pipeline.score import choose_breaks, engrave, engraved_chord, key_signature, layout_units, tempo_mark, to_mei

MEI = "{http://www.music-encoding.org/ns/mei}"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

CHART = """{title: Wonderwall}
{artist: Oasis}
{key: A}
{capo: 2}
{tempo: 88}
{time: 4/4}

{c: Intro}
| [Em7] | [G] | % |

{c: Verse 1}
Today | [Em7]is gonna be the | [G]day that [Dsus4]they're | [C] . . [F#m7b5]gonna |
"""


def _mei(text=CHART, **kw):
    return ET.fromstring(to_mei(read_score(text), **kw).split("\n", 1)[1])


def _text(el):
    return "".join(el.itertext())


def test_key_signature_of_the_written_shapes():
    assert [key_signature(k) for k in ("C", "G", "D", "A", "E", "B", "F#")] == ["0", "1s", "2s", "3s", "4s", "5s", "6s"]
    assert [key_signature(k) for k in ("F", "Bb", "Eb", "Ab", "C#")] == ["1f", "2f", "3f", "4f", "5f"]
    assert key_signature("Am") == "0" and key_signature("F#m") == "3s" and key_signature("Em") == "1s"
    assert key_signature("Fm") == "4f" and key_signature("") == "0" and key_signature("?") == "0"


def test_chord_symbols_get_real_accidentals():
    assert engraved_chord("F#m7b5") == "F♯m7♭5" and engraved_chord("Bb/D") == "B♭/D" and engraved_chord("C/Bb") == "C/B♭"
    assert engraved_chord("Absus4") == "A♭sus4" and engraved_chord("E7#9") == "E7♯9"
    assert engraved_chord("Em7") == "Em7" and engraved_chord("Dsus4") == "Dsus4" and engraved_chord("N.C.") == "N.C."


def test_tempo_mark_is_a_quarter_or_a_dotted_quarter():
    quarter, dot = "\uECA5", "\uECB7"
    assert tempo_mark("88", (4, 4)) == (quarter, "88") and tempo_mark("120", (3, 4)) == (quarter, "120")
    assert tempo_mark("50", (6, 8)) == (quarter + dot, "50") and tempo_mark("60", (12, 8)) == (quarter + dot, "60")
    assert tempo_mark("", (4, 4)) is None and tempo_mark(None, (4, 4)) is None and tempo_mark("fast", (4, 4)) is None


def test_layout_width_is_clamped():
    assert layout_units(400) == 1100 and layout_units(900) == 2000
    assert layout_units(10) == layout_units(280) and layout_units(99999) == layout_units(1600)


def test_mei_is_one_staff_with_clef_key_and_time():
    root = _mei(key="G", capo=2)
    score_def = root.find(f".//{MEI}scoreDef")
    assert (score_def.get("keysig"), score_def.get("meter.count"), score_def.get("meter.unit")) == ("1s", "4", "4")
    staves = root.findall(f".//{MEI}staffDef")
    assert len(staves) == 1 and (staves[0].get("clef.shape"), staves[0].get("clef.line"), staves[0].get("lines")) == ("G", "2", "5")
    assert root.find(f".//{MEI}titleStmt/{MEI}title").text == "Wonderwall"


def test_mei_header_has_title_artist_capo_and_tempo():
    head = _mei(key="G", capo=2).find(f".//{MEI}pgHead")
    by_type = {r.get("type"): r for r in head.iter(f"{MEI}rend") if r.get("type")}
    assert _text(by_type["title"]) == "Wonderwall" and _text(by_type["artist"]) == "Oasis"
    assert _text(by_type["capo"]) == "Capo 2"
    assert _text(by_type["tempo"]) == "\uECA5 = 88" and by_type["tempo"].get("halign") == "left"
    assert "capo" not in {r.get("type") for r in _mei(capo=0).iter(f"{MEI}rend")}      # no capo, no line


def test_mei_has_one_slash_per_beat():
    root = _mei()
    measures = root.findall(f".//{MEI}measure")
    assert [m.get("n") for m in measures] == ["1", "2", "3", "4", "5", "6"]
    for m in measures:
        notes = m.findall(f".//{MEI}note")
        assert len(notes) == 4
        assert all((n.get("dur"), n.get("head.shape"), n.get("stem.visible"), n.get("dots")) == ("4", "slash", "false", None)
                   for n in notes)
    six_eight = _mei("{time: 6/8}\n| [C] | [G] [D] |\n")
    notes = six_eight.findall(f".//{MEI}note")
    assert len(notes) == 4 and all((n.get("dur"), n.get("dots")) == ("4", "1") for n in notes)     # two dotted quarters a bar
    assert [h.get("tstamp") for h in six_eight.iter(f"{MEI}harm")] == ["1", "1", "4"]              # the second on eighth 4
    assert len(_mei("{time: 3/4}\n| [C] |\n").findall(f".//{MEI}note")) == 3


def test_mei_chords_sit_on_their_beat_and_carry_the_chord_number():
    root = _mei()
    harms = [(h.get(XML_ID), h.get("tstamp"), _text(h), h.get("place")) for h in root.iter(f"{MEI}harm")]
    assert harms == [("chord-0", "1", "Em7", "above"), ("chord-1", "1", "G", "above"),      # "%": nothing new to print
                     ("chord-2", "1", "Em7", "above"), ("chord-3", "1", "G", "above"), ("chord-4", "3", "Dsus4", "above"),
                     ("chord-5", "1", "C", "above"), ("chord-6", "4", "F♯m7♭5", "above")]
    # transposing moves every chord; the numbers stay those of the chart text
    moved = [(h.get(XML_ID), _text(h)) for h in _mei(semis=2).iter(f"{MEI}harm")]
    assert moved[0] == ("chord-0", "F♯m7") and moved[4] == ("chord-4", "Esus4") and moved[6] == ("chord-6", "A♭m7♭5")


def test_mei_repeated_bar_prints_a_chord_only_where_it_changes():
    root = _mei("| [C] [G] | % | [Am] | % |\n")
    by_bar = [[_text(h) for h in m.iter(f"{MEI}harm")] for m in root.iter(f"{MEI}measure")]
    assert by_bar == [["C", "G"], ["C", "G"], ["Am"], []]
    assert [h.get(XML_ID) for h in root.iter(f"{MEI}harm")] == ["chord-0", "chord-1", None, None, "chord-2"]


def test_mei_words_are_under_the_beat_they_start_on():
    root = _mei()
    bars = [[_text(n) for n in m.iter(f"{MEI}note")] for m in root.iter(f"{MEI}measure")]
    assert bars[2] == ["", "", "", "Today"]                       # the pickup, on the last beat of the intro
    assert bars[3] == ["is", "gonna", "be", "the"]
    assert bars[4] == ["day", "that", "they're", ""]
    assert root.find(f".//{MEI}verse").get("n") == "1" and root.find(f".//{MEI}verse/{MEI}syl") is not None


def test_mei_sections_have_rehearsal_marks_and_double_bars():
    root = _mei()
    sections = root.findall(f".//{MEI}score/{MEI}section")
    assert [s.get("label") for s in sections] == ["Intro", "Verse 1"]
    rehs = {m.get("n"): [(r.get("type"), _text(r)) for r in m.iter(f"{MEI}reh")] for m in root.iter(f"{MEI}measure")}
    assert {n: marks for n, marks in rehs.items() if marks} == {"1": [("section", "Intro")], "4": [("section", "Verse 1")]}
    assert [m.get("right") for m in root.iter(f"{MEI}measure")] == [None, None, "dbl", None, None, "end"]


def test_mei_system_breaks_where_asked():
    root = _mei(breaks={3, 5})
    order = [el.tag.replace(MEI, "") + (el.get("n") or "") for s in root.iter(f"{MEI}section") for el in s]
    assert order == ["measure1", "measure2", "sb", "measure3", "measure4", "sb", "measure5", "measure6"]
    alone = _mei(every_bar_alone=True)
    assert len(alone.findall(f".//{MEI}sb")) == 5 and alone.find(f".//{MEI}pgHead") is None


def test_mei_meter_change_has_time_signature_and_tempo_mark():
    text = ("{time: 6/8}\n{tempo: 50}\n{c: Chorus}\n| [D]me get out | [C] |\n{c: Verse}\n{time: 4/4}\n{tempo: 75}\n"
            "| [C]It's hard | % |\n{tempo: 80}\n| [G] |\n")
    root = _mei(text)
    first = root.find(f".//{MEI}score/{MEI}scoreDef")
    assert (first.get("meter.count"), first.get("meter.unit")) == ("6", "8")
    verse = root.findall(f".//{MEI}score/{MEI}section")[1]
    assert [el.tag.replace(MEI, "") for el in verse] == ["scoreDef", "measure", "measure", "measure"]
    assert (verse[0].get("meter.count"), verse[0].get("meter.unit")) == ("4", "4")
    marks = {m.get("n"): [(r.get("type"), _text(r)) for r in m.iter(f"{MEI}reh")] for m in root.iter(f"{MEI}measure")}
    assert marks == {"1": [("section", "Chorus")], "2": [],            # the first tempo is in the header
                     "3": [("tempo", "\uECA5 = 75"), ("section", "Verse")], "4": [], "5": [("tempo", "\uECA5 = 80")]}
    assert [len(m.findall(f".//{MEI}note")) for m in root.iter(f"{MEI}measure")] == [2, 2, 4, 4, 4]
    assert _text(next(r for r in root.iter(f"{MEI}rend") if r.get("type") == "tempo")) == "\uECA5\uECB7 = 50"


def test_mei_escapes_what_users_write():
    xml = to_mei(read_score('{title: <script>alert(1)</script>}\n{c: <b>}\n| [C<x>]a&b "q" |\n'))
    root = ET.fromstring(xml.split("\n", 1)[1])                 # still well-formed
    assert "<script>" not in xml and _text(root.find(f".//{MEI}harm")) == "C<x>"


def test_breaks_fill_the_systems():
    # room for 100: bars of 29 after a lead of 10 -> three bars a system
    assert choose_breaks([29] * 8, 10, 100) == {4, 7}
    assert choose_breaks([29, 29, 50, 29], 10, 100) == {3}            # the wide bar does not fit after two
    assert choose_breaks([200, 29], 10, 100) == {2}                   # a bar too wide for a system still gets one
    assert choose_breaks([], 10, 100) == set()


def test_breaks_let_a_section_start_a_system_when_the_one_before_is_full_enough():
    assert choose_breaks([29] * 7, 10, 100, section_ends={2}) == {3, 6}      # 68 % full: break after bar 2
    assert choose_breaks([29] * 6, 10, 100, section_ends={1}) == {4}         # 39 % full: the next section joins


def test_breaks_even_out_the_systems():
    assert choose_breaks([29] * 4, 10, 100) == {3}                           # two and two, not three and one
    assert choose_breaks([29] * 7, 10, 100) == {4, 6}                        # 3 + 2 + 2, not 3 + 3 + 1
    assert choose_breaks([29, 29, 29, 60], 10, 100) == {4}                   # only when what moves fits
    # ... but a bar never leaves the end of its section to do so
    assert choose_breaks([29] * 4, 10, 100, section_ends={3}) == {4}


# --- through Verovio ------------------------------------------------------------------------------

def test_engrave_gives_svg_with_tappable_chords():
    e = engrave(CHART, width=400)
    assert len(e.pages) == 1 and e.bars == 6 and e.unbarred == 0
    svg = e.pages[0]
    assert svg.startswith("<svg") and "viewBox" in svg
    assert sorted(int(i) for i in re.findall(r'id="chord-(\d+)" class="harm"', svg)) == list(range(7))
    assert len(re.findall(r'id="bar-\d+" class="measure"', svg)) == 6
    for words in ("Wonderwall", "Oasis", "Capo 2", "Em7", "Dsus4", "Today", "gonna", "Verse 1"):
        assert f">{words}<" in svg, words
    assert "@font-face" not in svg          # the music font comes from /score-font.woff2, once


def test_engrave_follows_capo_and_key():
    assert ">Em7<" in engrave(CHART, width=400).pages[0]                    # the chart's own capo
    no_capo = engrave(CHART, width=400, capo=0).pages[0]                    # sounding chords: F#m7, A, Esus4
    assert ">m7<" in no_capo and ">Em7<" not in no_capo and ">Esus4<" in no_capo and "Capo" not in no_capo
    up = engrave(CHART, width=400, capo=2, key=1).pages[0]
    assert ">Fm7<" in up


def test_engrave_narrow_screen_gets_more_systems():
    text = "{time: 4/4}\n" + "| [C]la la la | [G]la la la | [Am]la la la | [F]la la la |\n" * 4

    def systems(width):
        return len(re.findall(r'class="system"', engrave(text, width=width).pages[0]))
    assert systems(360) > systems(1000) >= 2


def test_bar_with_too_many_words_is_set_smaller():
    crowded = "| [C]" + "word " * 30 + "| [G]fine |\n"
    svg = engrave(crowded, width=300).pages[0]
    sizes = {int(size) for size in re.findall(r'<tspan font-size="(\d+)px">(?:word|fine)', svg)}
    assert min(sizes) < 0.75 * max(sizes)                  # "fine" at the normal size, the crowded bar smaller
    root = _mei(crowded, small={1: 60})
    syls = root.findall(f".//{MEI}syl")
    assert syls[0].find(f"{MEI}rend").get("fontsize") == "60%" and syls[-1].find(f"{MEI}rend") is None


def test_engrave_is_cached_and_repeatable():
    first = engrave(CHART, width=500)
    assert engrave(CHART, width=500) is first
    score._cache.clear()
    assert engrave(CHART, width=500).pages == first.pages         # same input, same SVG


def test_engrave_old_chart_without_bars():
    e = engrave("{title: Old}\n{c: Verse}\n[Am]Is it so [C]perfect now\n[F]  [G]\n", width=400)
    assert e.bars == 4 and e.unbarred == 4 and len(e.pages) == 1
    assert engrave("{title: Empty}\n", width=400).pages == ()


def test_engrave_refuses_endless_charts():
    with pytest.raises(score.TooLong):
        engrave("| [C] |\n" * (score.MAX_BARS + 1))


def test_music_font_is_a_woff2():
    assert score.music_font()[:4] == b"wOF2"


def test_engrave_still_works_after_weasyprint_is_imported():
    # WeasyPrint registers its own default XML namespace in ElementTree's process-wide registry;
    # the MEI must not depend on that registry (a PDF download used to break every later score).
    import xml.etree.ElementTree as ET

    text = "{time: 4/4}\n{tempo: 100}\n| [C]la la | [G]la la |\n"
    assert engrave(text).pages
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    assert engrave(text + "| [Am]la |\n").pages
