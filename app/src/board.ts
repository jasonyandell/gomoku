// SVG board rendering. Ported from web/static/app.js.

import { BOARD_SIZE, actionToStr } from "./game";

const SVG_NS = "http://www.w3.org/2000/svg";
const PAD = 30;
const VIEW = 540;
const CELL = (VIEW - 2 * PAD) / (BOARD_SIZE - 1);

export type HintMove = { action: number; visits: number; q: number };

export function colXY(c: number): number { return PAD + c * CELL; }
export function rowXY(r: number): number { return PAD + r * CELL; }
export function actionXY(a: number): { x: number; y: number } {
  return { x: colXY(a % BOARD_SIZE), y: rowXY(Math.floor(a / BOARD_SIZE)) };
}

export function actionFromXY(svgX: number, svgY: number): number | null {
  const c = Math.round((svgX - PAD) / CELL);
  const r = Math.round((svgY - PAD) / CELL);
  if (c < 0 || c >= BOARD_SIZE || r < 0 || r >= BOARD_SIZE) return null;
  return r * BOARD_SIZE + c;
}

export function stonesFromHistory(history: number[]): { 0: number[]; 1: number[] } {
  const out: { 0: number[]; 1: number[] } = { 0: [], 1: [] };
  for (let i = 0; i < history.length; i++) out[(i % 2) as 0 | 1].push(history[i]);
  return out;
}

export function renderBoard(
  svg: SVGSVGElement,
  stones: { 0: number[]; 1: number[] },
  lastAction: number | null,
  hints: HintMove[],
): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // grid
  const grid = document.createElementNS(SVG_NS, "g");
  grid.setAttribute("class", "grid");
  for (let i = 0; i < BOARD_SIZE; i++) {
    const h = document.createElementNS(SVG_NS, "line");
    h.setAttribute("x1", String(colXY(0)));
    h.setAttribute("x2", String(colXY(BOARD_SIZE - 1)));
    h.setAttribute("y1", String(rowXY(i)));
    h.setAttribute("y2", String(rowXY(i)));
    h.setAttribute("stroke", "#2a2a2a");
    grid.appendChild(h);
    const v = document.createElementNS(SVG_NS, "line");
    v.setAttribute("x1", String(colXY(i)));
    v.setAttribute("x2", String(colXY(i)));
    v.setAttribute("y1", String(rowXY(0)));
    v.setAttribute("y2", String(rowXY(BOARD_SIZE - 1)));
    v.setAttribute("stroke", "#2a2a2a");
    grid.appendChild(v);
  }
  svg.appendChild(grid);

  // star points
  for (const [r, c] of [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]] as Array<[number, number]>) {
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.setAttribute("cx", String(colXY(c)));
    dot.setAttribute("cy", String(rowXY(r)));
    dot.setAttribute("r", "3");
    dot.setAttribute("class", "star");
    svg.appendChild(dot);
  }

  // coordinates
  for (let c = 0; c < BOARD_SIZE; c++) {
    for (const y of [12, VIEW - 12]) {
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "coord");
      t.setAttribute("x", String(colXY(c)));
      t.setAttribute("y", String(y));
      t.textContent = String.fromCharCode("a".charCodeAt(0) + c);
      svg.appendChild(t);
    }
  }
  for (let r = 0; r < BOARD_SIZE; r++) {
    for (const x of [12, VIEW - 12]) {
      const t = document.createElementNS(SVG_NS, "text");
      t.setAttribute("class", "coord");
      t.setAttribute("x", String(x));
      t.setAttribute("y", String(rowXY(r)));
      t.textContent = String(r + 1);
      svg.appendChild(t);
    }
  }

  // hints
  if (hints.length > 0) {
    const maxV = Math.max(...hints.map((m) => m.visits));
    const totalV = hints.reduce((s, h) => s + h.visits, 0);
    for (const m of hints) {
      const { x, y } = actionXY(m.action);
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", String(x));
      c.setAttribute("cy", String(y));
      c.setAttribute("r", String(6 + 10 * (m.visits / Math.max(maxV, 1))));
      c.setAttribute("class", "hint");
      svg.appendChild(c);
      if (totalV > 0) {
        const t = document.createElementNS(SVG_NS, "text");
        t.setAttribute("class", "coord");
        t.setAttribute("x", String(x));
        t.setAttribute("y", String(y - 14));
        t.setAttribute("style", "fill:#2855a3");
        t.textContent = `${Math.round((100 * m.visits) / totalV)}%`;
        svg.appendChild(t);
      }
    }
  }

  // stones
  for (const side of [0, 1] as const) {
    for (const a of stones[side]) {
      const { x, y } = actionXY(a);
      const stone = document.createElementNS(SVG_NS, "circle");
      stone.setAttribute("cx", String(x));
      stone.setAttribute("cy", String(y));
      stone.setAttribute("r", String(CELL * 0.42));
      stone.setAttribute("class", side === 0 ? "stone-b" : "stone-w");
      svg.appendChild(stone);
    }
  }

  // last move marker
  if (lastAction != null) {
    const { x, y } = actionXY(lastAction);
    const marker = document.createElementNS(SVG_NS, "circle");
    marker.setAttribute("cx", String(x));
    marker.setAttribute("cy", String(y));
    marker.setAttribute("r", String(CELL * 0.18));
    marker.setAttribute("class", "last-marker");
    svg.appendChild(marker);
  }
}

export function actionName(a: number): string {
  return actionToStr(a);
}
