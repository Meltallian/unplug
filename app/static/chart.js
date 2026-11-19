// Chart view: capo / transpose, chord diagrams, font size, auto-scroll, screen wake lock.
(function () {
  "use strict";
  const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
  const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
  // Common shapes, low E to high e (x = muted). One voicing per chord: open where possible.
  const SHAPES = {
    "C": "x32010", "C#": "x46664", "D": "xx0232", "Eb": "x68886", "E": "022100", "F": "133211",
    "F#": "244322", "G": "320003", "Ab": "466544", "A": "x02220", "Bb": "x13331", "B": "x24442",
    "Cm": "x35543", "C#m": "x46654", "Dm": "xx0231", "Ebm": "x68876", "Em": "022000", "Fm": "133111",
    "F#m": "244222", "Gm": "355333", "Abm": "466444", "Am": "x02210", "Bbm": "x13321", "Bm": "x24432",
  };
  const CHORD = /^([A-G])([#b]?)(.*?)(?:\/([A-G])([#b]?))?$/;

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };

  const sheet = document.getElementById("sheet");
  const head = document.getElementById("head");
  const chartCapo = parseInt(sheet.dataset.capo || "0", 10) || 0;
  const chartKey = sheet.dataset.key || "";
  const tempo = sheet.dataset.tempo || "";
  const state = { capo: chartCapo, key: 0, size: store.get("unplug.size", 19), speed: store.get("unplug.speed", 22) };

  function pc(letter, acc) { return (PC[letter] + (acc === "#" ? 1 : acc === "b" ? -1 : 0) + 12) % 12; }

  function moveChord(name, semis) {
    const m = CHORD.exec(name.trim());
    if (!m) return name;  // N.C. or free text: leave alone
    const root = NAMES[(pc(m[1], m[2]) + semis + 120) % 12];
    const bass = m[4] ? "/" + NAMES[(pc(m[4], m[5]) + semis + 120) % 12] : "";
    return root + m[3] + bass;
  }

  function shapeKey(name) {
    const m = CHORD.exec(name.trim());
    if (!m) return null;
    const minor = /^(m(?!aj)|min|-)/.test(m[3]);
    return NAMES[pc(m[1], m[2])] + (minor ? "m" : "");
  }

  function diagram(name) {
    const key = shapeKey(name);
    const frets = key && SHAPES[key];
    if (!frets) return "";
    const f = frets.split("").map((c) => (c === "x" ? -1 : parseInt(c, 10)));
    const played = f.filter((v) => v > 0);
    const base = Math.max(...played, 0) > 4 ? Math.min(...played) : 1;
    const x = (i) => 10 + i * 9, y = (row) => 18 + row * 12;
    let s = `<svg viewBox="0 0 64 84" role="img" aria-label="${key} chord diagram">`;
    for (let i = 0; i < 6; i++) s += `<line class="grid" x1="${x(i)}" y1="${y(0)}" x2="${x(i)}" y2="${y(5)}"/>`;
    for (let r = 0; r <= 5; r++) s += `<line class="${r === 0 && base === 1 ? "nut" : "grid"}" x1="${x(0)}" y1="${y(r)}" x2="${x(5)}" y2="${y(r)}"/>`;
    if (base > 1) s += `<text x="0" y="${y(0.8)}">${base}</text>`;
    f.forEach((v, i) => {
      if (v < 0) s += `<text x="${x(i) - 3}" y="12">×</text>`;
      else if (v === 0) s += `<circle class="mark" cx="${x(i)}" cy="10" r="2.6"/>`;
      else s += `<circle class="dot" cx="${x(i)}" cy="${y(v - base + 0.5)}" r="3.6"/>`;
    });
    return s + "</svg>";
  }

  function render() {
    const semis = chartCapo + state.key - state.capo;  // chart shapes → shapes to play now
    const used = [];
    document.querySelectorAll(".ch[data-c]").forEach((el) => {
      const name = moveChord(el.dataset.c, semis);
      el.textContent = name;
      const k = shapeKey(name);
      if (k && !used.includes(k)) used.push(k);
    });
    document.getElementById("diagrams").innerHTML = used
      .map((k) => `<div class="diagram">${diagram(k)}${k}</div>`).join("");
    const key = chartKey ? moveChord(chartKey, state.key) : "";
    document.getElementById("facts").innerHTML =
      (key ? `Key <b>${key}</b> · ` : "") + (state.capo ? `Capo <b>${state.capo}</b>` : "<b>No capo</b>") +
      (tempo ? ` · ${tempo} bpm` : "") + (state.key ? ` · transposed ${state.key > 0 ? "+" : ""}${state.key}` : "");
    document.getElementById("capo-label").textContent = "Capo " + state.capo;
    document.getElementById("key-label").textContent = state.key ? (state.key > 0 ? "+" : "") + state.key : "Key";
    document.documentElement.style.setProperty("--chart-size", state.size + "px");
  }

  // --- auto-scroll ---------------------------------------------------------------------------
  let scrolling = false, last = 0, carry = 0;
  const btn = document.getElementById("scroll-btn");
  function step(t) {
    if (!scrolling) return;
    if (last) {
      carry += (state.speed * (t - last)) / 1000;
      const px = Math.floor(carry);
      if (px) { window.scrollBy(0, px); carry -= px; }
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) setScrolling(false);
    }
    last = t;
    requestAnimationFrame(step);
  }
  function setScrolling(on) {
    scrolling = on; last = 0; carry = 0;
    btn.classList.toggle("on", on);
    btn.textContent = on ? "❚❚" : "▶";
    if (on) requestAnimationFrame(step);
  }
  sheet.addEventListener("click", (e) => {
    if (e.target.closest("a, button, form, textarea")) return;
    if (scrolling) setScrolling(false);
  });

  // --- toolbar -------------------------------------------------------------------------------
  document.getElementById("toolbar").addEventListener("click", (e) => {
    const act = e.target.closest("button")?.dataset.act;
    if (!act) return;
    if (act === "font+") state.size = Math.min(34, state.size + 2);
    if (act === "font-") state.size = Math.max(13, state.size - 2);
    if (act === "capo+") state.capo = Math.min(9, state.capo + 1);
    if (act === "capo-") state.capo = Math.max(0, state.capo - 1);
    if (act === "key+") state.key = state.key >= 6 ? -5 : state.key + 1;
    if (act === "key-") state.key = state.key <= -5 ? 6 : state.key - 1;
    if (act === "speed+") state.speed = Math.min(150, Math.round(state.speed * 1.25));
    if (act === "speed-") state.speed = Math.max(5, Math.round(state.speed / 1.25));
    if (act === "scroll") setScrolling(!scrolling);
    store.set("unplug.size", state.size);
    store.set("unplug.speed", state.speed);
    render();
  });

  // --- collapse the diagrams while scrolling down --------------------------------------------
  let lastY = window.scrollY;
  window.addEventListener("scroll", () => {
    const y = window.scrollY;
    if (y > lastY + 4 && y > 60) head.classList.add("collapsed");
    else if (y < lastY - 4 || y < 20) head.classList.remove("collapsed");
    lastY = y;
  }, { passive: true });

  // --- keep the screen awake while the chart is open -----------------------------------------
  let lock = null;
  async function wake() {
    try { if ("wakeLock" in navigator && document.visibilityState === "visible") lock = await navigator.wakeLock.request("screen"); }
    catch (e) { lock = null; }
  }
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") wake(); });
  wake();

  render();
})();
