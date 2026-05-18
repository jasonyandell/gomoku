// gomoku web UI
const BOARD_SIZE = 9;
const SVG_NS = "http://www.w3.org/2000/svg";
const PAD = 30;             // viewBox padding for coords
const VIEW = 540;           // viewBox size
const CELL = (VIEW - 2 * PAD) / (BOARD_SIZE - 1);

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// -------- shared state --------
const state = {
  checkpoints: [],
  selectedCheckpoint: null,
  tab: "play",
  // Play state
  play: {
    history: [],      // list of action ints
    humanColor: "black",
    busy: false,
    topMoves: [],
    hint: null,
  },
  // Replay state
  replay: {
    moves: [],
    plyIdx: 0,
    outcome: 0,
    playing: false,
    timer: null,
  },
};

// -------- API helpers --------
async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

// -------- board rendering --------
function colXY(c) { return PAD + c * CELL; }
function rowXY(r) { return PAD + r * CELL; }
function actionXY(a) { return { x: colXY(a % BOARD_SIZE), y: rowXY(Math.floor(a / BOARD_SIZE)) }; }

function colName(c) { return String.fromCharCode("a".charCodeAt(0) + c); }
function rowName(r) { return String(r + 1); }

function renderBoard(stonesBySide, lastAction, hintMoves) {
  // stonesBySide: { 0: [actions], 1: [actions] }
  // side 0 = first mover (black), side 1 = white
  // hintMoves: [{action, visits, q, rank}] — top moves for the side to move
  const svg = $("#board");
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // grid
  const grid = document.createElementNS(SVG_NS, "g");
  grid.setAttribute("class", "grid");
  for (let i = 0; i < BOARD_SIZE; i++) {
    const line1 = document.createElementNS(SVG_NS, "line");
    line1.setAttribute("x1", colXY(0));
    line1.setAttribute("x2", colXY(BOARD_SIZE - 1));
    line1.setAttribute("y1", rowXY(i));
    line1.setAttribute("y2", rowXY(i));
    line1.setAttribute("stroke", "#2a2a2a");
    grid.appendChild(line1);
    const line2 = document.createElementNS(SVG_NS, "line");
    line2.setAttribute("x1", colXY(i));
    line2.setAttribute("x2", colXY(i));
    line2.setAttribute("y1", rowXY(0));
    line2.setAttribute("y2", rowXY(BOARD_SIZE - 1));
    line2.setAttribute("stroke", "#2a2a2a");
    grid.appendChild(line2);
  }
  svg.appendChild(grid);

  // star points (9x9: corners of inner 5x5)
  const stars = [[2,2],[2,6],[6,2],[6,6],[4,4]];
  for (const [r, c] of stars) {
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", colXY(c));
    dot.setAttribute("cy", rowXY(r));
    dot.setAttribute("r", 3);
    dot.setAttribute("class", "star");
    svg.appendChild(dot);
  }

  // coordinates
  for (let c = 0; c < BOARD_SIZE; c++) {
    for (const y of [12, VIEW - 12]) {
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "coord");
      t.setAttribute("x", colXY(c));
      t.setAttribute("y", y);
      t.textContent = colName(c);
      svg.appendChild(t);
    }
  }
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (const x of [12, VIEW - 12]) {
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "coord");
      t.setAttribute("x", x);
      t.setAttribute("y", rowXY(r));
      t.textContent = rowName(r);
      svg.appendChild(t);
    }
  }

  // hint moves (under stones)
  if (hintMoves && hintMoves.length) {
    const maxV = Math.max(...hintMoves.map(m => m.visits));
    for (const m of hintMoves) {
      const { x, y } = actionXY(m.action);
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", x);
      c.setAttribute("cy", y);
      c.setAttribute("r", 6 + 10 * (m.visits / maxV));
      c.setAttribute("class", "hint");
      svg.appendChild(c);
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "coord");
      t.setAttribute("x", x);
      t.setAttribute("y", y - 14);
      t.setAttribute("style", "fill:#2855a3");
      t.textContent = `${Math.round(100 * m.visits / hintMoves.reduce((s,h)=>s+h.visits,0))}%`;
      svg.appendChild(t);
    }
  }

  // stones
  for (const side of [0, 1]) {
    for (const a of stonesBySide[side]) {
      const { x, y } = actionXY(a);
      const stone = document.createElementNS(SVG_NS, "circle");
      stone.setAttribute("cx", x);
      stone.setAttribute("cy", y);
      stone.setAttribute("r", CELL * 0.42);
      stone.setAttribute("class", side === 0 ? "stone-b" : "stone-w");
      svg.appendChild(stone);
    }
  }

  // last move marker
  if (lastAction != null) {
    const { x, y } = actionXY(lastAction);
    const marker = document.createElementNS(SVG_NS, "circle");
    marker.setAttribute("cx", x);
    marker.setAttribute("cy", y);
    marker.setAttribute("r", CELL * 0.18);
    marker.setAttribute("class", "last-marker");
    svg.appendChild(marker);
  }
}

function actionFromXY(svgX, svgY) {
  const c = Math.round((svgX - PAD) / CELL);
  const r = Math.round((svgY - PAD) / CELL);
  if (c < 0 || c >= BOARD_SIZE || r < 0 || r >= BOARD_SIZE) return null;
  return r * BOARD_SIZE + c;
}

function stonesFromHistory(history) {
  const out = { 0: [], 1: [] };
  for (let i = 0; i < history.length; i++) out[i % 2].push(history[i]);
  return out;
}

// -------- checkpoint dropdown --------
async function refreshCheckpoints() {
  const data = await api("/api/checkpoints");
  state.checkpoints = data.checkpoints || [];
  const sel = $("#checkpoint-select");
  sel.innerHTML = "";
  if (!state.checkpoints.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no checkpoints yet)";
    sel.appendChild(opt);
    state.selectedCheckpoint = null;
    return;
  }
  for (const c of state.checkpoints) {
    if (c.error) continue;
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.name} (${c.model_size}, epoch ${c.epoch}, ${c.total_games}g)`;
    sel.appendChild(opt);
  }
  if (!state.selectedCheckpoint || !state.checkpoints.find(c => c.path === state.selectedCheckpoint)) {
    // Default to the latest by epoch.
    const valid = state.checkpoints.filter(c => !c.error);
    state.selectedCheckpoint = valid[valid.length - 1]?.path || null;
    sel.value = state.selectedCheckpoint || "";
  }
}

// -------- PLAY tab --------
function renderPlay() {
  const stones = stonesFromHistory(state.play.history);
  const last = state.play.history.at(-1);
  renderBoard(stones, last, state.play.topMoves);

  // top moves list
  const ul = $("#play-top");
  ul.innerHTML = "";
  for (let i = 0; i < state.play.topMoves.length; i++) {
    const m = state.play.topMoves[i];
    const li = document.createElement("li");
    if (i === 0) li.className = "best";
    li.innerHTML = `<span>${m.name}</span><span>${m.visits} • Q=${m.q.toFixed(2)}</span>`;
    ul.appendChild(li);
  }

  // status line
  const ply = state.play.history.length;
  const sideToMove = ply % 2 === 0 ? "black" : "white";
  let status = `ply ${ply} — ${sideToMove} to move`;
  const humanSide = state.play.humanColor === "black" ? 0 : 1;
  if (ply % 2 === humanSide) status += " (you)"; else status += " (model)";
  $("#status").textContent = status;
}

async function modelMove() {
  if (state.play.busy) return;
  state.play.busy = true;
  $("#status").textContent = `thinking (${$("#play-sims").value} sims)…`;
  try {
    const r = await api("/api/move", {
      checkpoint: state.selectedCheckpoint,
      history: state.play.history,
      n_simulations: parseInt($("#play-sims").value, 10),
    });
    if (r.done) {
      $("#status").textContent = "game already finished";
      return;
    }
    state.play.history.push(r.best.action);
    state.play.topMoves = r.top;
    $("#play-eval").textContent = `model eval: ${r.value >= 0 ? "+" : ""}${r.value.toFixed(2)}`;
    renderPlay();
    checkPlayTerminal();
  } catch (e) {
    $("#status").textContent = "error: " + e.message;
  } finally {
    state.play.busy = false;
    maybeStepModel();
  }
}

function checkPlayTerminal() {
  // Lightweight client-side check: 5-in-a-row by the side that just moved.
  const ply = state.play.history.length;
  if (ply === 0) return false;
  const sideJustMoved = (ply - 1) % 2;
  const occ = new Set();
  const mine = new Set();
  for (let i = 0; i < state.play.history.length; i++) {
    const a = state.play.history[i];
    occ.add(a);
    if (i % 2 === sideJustMoved) mine.add(a);
  }
  if (hasFive(mine)) {
    const winner = sideJustMoved === 0 ? "black" : "white";
    const humanSide = state.play.humanColor === "black" ? 0 : 1;
    const youWon = sideJustMoved === humanSide;
    $("#status").textContent = `${winner} wins — ${youWon ? "you" : "model"} 🏆`;
    state.play.busy = true; // freeze
    return true;
  }
  if (ply >= 81) {
    $("#status").textContent = "draw";
    state.play.busy = true;
    return true;
  }
  return false;
}

function hasFive(positions) {
  const dirs = [[0,1],[1,0],[1,1],[1,-1]];
  for (const a of positions) {
    const r = Math.floor(a / BOARD_SIZE), c = a % BOARD_SIZE;
    for (const [dr, dc] of dirs) {
      let ok = true;
      for (let k = 0; k < 5; k++) {
        const rr = r + dr * k, cc = c + dc * k;
        if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) { ok = false; break; }
        if (!positions.has(rr * BOARD_SIZE + cc)) { ok = false; break; }
      }
      if (ok) return true;
    }
  }
  return false;
}

function maybeStepModel() {
  if (state.tab !== "play") return;
  if (state.play.busy) return;
  if (checkPlayTerminal()) return;
  const humanSide = state.play.humanColor === "black" ? 0 : 1;
  const sideToMove = state.play.history.length % 2;
  if (sideToMove !== humanSide && state.selectedCheckpoint) {
    modelMove();
  }
}

function newPlayGame() {
  state.play.history = [];
  state.play.topMoves = [];
  state.play.busy = false;
  $("#play-eval").textContent = "";
  renderPlay();
  maybeStepModel();
}

function undoMove() {
  // Undo two plies so it's the human's turn again.
  if (state.play.busy && !checkPlayTerminal()) return;
  state.play.history = state.play.history.slice(0, Math.max(0, state.play.history.length - 2));
  state.play.busy = false;
  state.play.topMoves = [];
  $("#play-eval").textContent = "";
  renderPlay();
  maybeStepModel();
}

// -------- REPLAY tab --------
async function generateReplay() {
  if (!state.selectedCheckpoint) {
    $("#status").textContent = "select a checkpoint first";
    return;
  }
  $("#status").textContent = "generating game…";
  $("#rep-generate").disabled = true;
  try {
    const r = await api("/api/selfplay-detailed", {
      checkpoint: state.selectedCheckpoint,
      n_simulations: parseInt($("#rep-sims").value, 10),
      temperature_moves: parseInt($("#rep-temp").value, 10),
      dirichlet_eps: parseFloat($("#rep-noise").value),
      seed: parseInt($("#rep-seed").value, 10),
    });
    state.replay.moves = r.moves;
    state.replay.plyIdx = r.moves.length;
    state.replay.outcome = r.outcome;
    $("#rep-scrub").max = r.moves.length;
    $("#rep-scrub").value = r.moves.length;
    renderReplay();
    let outStr = "draw";
    if (r.outcome > 0) outStr = "black wins";
    else if (r.outcome < 0) outStr = "white wins";
    $("#status").textContent = `${r.plies} plies — ${outStr}`;
  } catch (e) {
    $("#status").textContent = "error: " + e.message;
  } finally {
    $("#rep-generate").disabled = false;
  }
}

function renderReplay() {
  const idx = state.replay.plyIdx;
  const history = state.replay.moves.slice(0, idx).map(m => m.action);
  const stones = stonesFromHistory(history);
  const last = history.at(-1);
  // Show top moves for the move ABOUT to be made (i.e., move at position idx, if any).
  const upcoming = state.replay.moves[idx];
  const top = upcoming ? upcoming.top : [];
  renderBoard(stones, last, top);

  $("#rep-ply").textContent = `${idx} / ${state.replay.moves.length}`;
  $("#rep-scrub").value = idx;

  const ul = $("#rep-top");
  ul.innerHTML = "";
  for (let i = 0; i < top.length; i++) {
    const m = top[i];
    const li = document.createElement("li");
    if (m.action === upcoming?.action) li.className = "best";
    li.innerHTML = `<span>${m.name}</span><span>${m.visits} • Q=${m.q.toFixed(2)}</span>`;
    ul.appendChild(li);
  }
  if (upcoming) {
    $("#rep-eval").textContent = `chose ${upcoming.name} • Q=${upcoming.value.toFixed(2)}`;
  } else if (state.replay.moves.length) {
    $("#rep-eval").textContent = "(end of game)";
  } else {
    $("#rep-eval").textContent = "";
  }
}

function repStep(delta) {
  const n = state.replay.moves.length;
  if (n === 0) return;
  state.replay.plyIdx = Math.max(0, Math.min(n, state.replay.plyIdx + delta));
  renderReplay();
}

function repTogglePlay() {
  if (state.replay.playing) {
    clearInterval(state.replay.timer);
    state.replay.timer = null;
    state.replay.playing = false;
    $("#rep-play").textContent = "▶ play";
    return;
  }
  if (state.replay.plyIdx >= state.replay.moves.length) state.replay.plyIdx = 0;
  state.replay.playing = true;
  $("#rep-play").textContent = "⏸ pause";
  state.replay.timer = setInterval(() => {
    if (state.replay.plyIdx >= state.replay.moves.length) {
      repTogglePlay();
      return;
    }
    state.replay.plyIdx += 1;
    renderReplay();
  }, 600);
}

// -------- tabs --------
function setTab(name) {
  state.tab = name;
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $$(".panel").forEach(p => p.classList.toggle("active", p.id === `panel-${name}`));
  if (name === "play") renderPlay();
  else renderReplay();
}

// -------- board click handler (play tab) --------
function setupBoardClick() {
  const svg = $("#board");
  svg.addEventListener("click", (e) => {
    if (state.tab !== "play" || state.play.busy) return;
    const humanSide = state.play.humanColor === "black" ? 0 : 1;
    const sideToMove = state.play.history.length % 2;
    if (sideToMove !== humanSide) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const svgP = pt.matrixTransform(svg.getScreenCTM().inverse());
    const a = actionFromXY(svgP.x, svgP.y);
    if (a == null) return;
    if (state.play.history.includes(a)) return;
    state.play.history.push(a);
    state.play.topMoves = [];
    renderPlay();
    if (!checkPlayTerminal()) maybeStepModel();
  });
}

// -------- wire up --------
document.addEventListener("DOMContentLoaded", async () => {
  setupBoardClick();
  $$(".tab").forEach(t => t.addEventListener("click", () => setTab(t.dataset.tab)));
  $("#refresh-checkpoints").addEventListener("click", refreshCheckpoints);
  $("#checkpoint-select").addEventListener("change", (e) => {
    state.selectedCheckpoint = e.target.value || null;
  });

  $("#play-new").addEventListener("click", newPlayGame);
  $("#play-undo").addEventListener("click", undoMove);
  $("#play-resign").addEventListener("click", () => {
    $("#status").textContent = "you resigned";
    state.play.busy = true;
  });
  $("#play-color").addEventListener("change", (e) => {
    state.play.humanColor = e.target.value;
    newPlayGame();
  });

  $("#rep-generate").addEventListener("click", generateReplay);
  $("#rep-prev").addEventListener("click", () => repStep(-1));
  $("#rep-next").addEventListener("click", () => repStep(1));
  $("#rep-play").addEventListener("click", repTogglePlay);
  $("#rep-scrub").addEventListener("input", (e) => {
    state.replay.plyIdx = parseInt(e.target.value, 10);
    renderReplay();
  });

  await refreshCheckpoints();
  renderPlay();
});
