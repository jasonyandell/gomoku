#!/usr/bin/env python3
"""session_mindmap.py — a local web mindmap of session topics, grounded by sessions.

Reads the session_db cache, builds a topic↔session graph (topics are hubs; each
session hangs off the topics it discussed, edge width = how much), and emits a
self-contained HTML you can open or serve locally. Click any session node to get
its prompts and **copy-paste `resume` / `fork` commands**.

  python scripts/session_mindmap.py            # write ~/.claude/agent-fleet/mindmap.html
  python scripts/session_mindmap.py --serve     # serve on 127.0.0.1 + open browser
  python scripts/session_mindmap.py --out x.html # custom output path

Note: the graph library (vis-network) loads from a CDN, so the page needs network
the first time; the data itself is embedded and local.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db as sdb  # noqa: E402

PALETTE = ["#e06c75", "#61afef", "#98c379", "#e5c07b", "#c678dd",
           "#56b6c2", "#d19a66", "#abb2bf", "#be5046"]


def build_payload(graph: dict) -> dict:
    """export_graph() output -> {nodes, edges, sessions} for vis-network."""
    topic_color = {t["id"]: PALETTE[i % len(PALETTE)] for i, t in enumerate(graph["topics"])}
    weight_by_session: dict[str, int] = {}
    for e in graph["edges"]:
        weight_by_session[e["session_id"]] = weight_by_session.get(e["session_id"], 0) + e["weight"]
    linked = set(weight_by_session)

    nodes = []
    for t in graph["topics"]:
        nodes.append({"id": f"topic:{t['id']}", "label": t["label"], "group": "topic",
                      "shape": "box", "color": topic_color[t["id"]], "font": {"size": 22, "color": "#fff"},
                      "mass": 3})
    details = {}
    for s in graph["sessions"]:
        if s["session_id"] not in linked:        # only sessions grounded to ≥1 topic
            continue
        w = weight_by_session[s["session_id"]]
        label = (s["title"] or s["ai_id"])[:30]
        nodes.append({"id": f"sess:{s['session_id']}", "label": label, "group": "session",
                      "shape": "dot", "value": max(1, w), "color": "#3b4048",
                      "font": {"size": 12, "color": "#cdd3de"}})
        details[f"sess:{s['session_id']}"] = s

    edges = []
    for e in graph["edges"]:
        edges.append({"from": f"sess:{e['session_id']}", "to": f"topic:{e['topic_id']}",
                      "value": e["weight"], "color": {"color": topic_color.get(e["topic_id"], "#555"),
                                                      "opacity": 0.4}})
    return {"nodes": nodes, "edges": edges, "details": details}


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Session topic mindmap</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{margin:0;height:100%;background:#21252b;color:#cdd3de;font:14px/1.5 -apple-system,Segoe UI,sans-serif}
  #net{position:absolute;top:0;left:0;right:340px;bottom:0}
  #panel{position:absolute;top:0;right:0;width:340px;bottom:0;background:#282c34;border-left:1px solid #3b4048;
         padding:16px;overflow:auto;box-sizing:border-box}
  h1{font-size:15px;margin:0 0 4px} h2{font-size:13px;color:#61afef;margin:14px 0 4px}
  .hint{color:#7f848e;font-size:12px}
  code{display:block;background:#1b1e23;border:1px solid #3b4048;border-radius:6px;padding:8px 10px;margin:4px 0;
       font:12px/1.4 SFMono-Regular,Menlo,monospace;color:#98c379;white-space:pre-wrap;word-break:break-all}
  button{background:#3b4048;color:#cdd3de;border:0;border-radius:6px;padding:5px 10px;cursor:pointer;font-size:12px;margin:2px 0}
  button:hover{background:#4b5263} ul{margin:4px 0;padding-left:18px} li{margin:3px 0;color:#abb2bf}
</style></head><body>
<div id="net"></div>
<div id="panel">
  <h1>Session topic mindmap</h1>
  <div class="hint">Boxes = topics · dots = sessions (size = how much they touched topics).
  Click a session for its prompts and copy-paste <b>resume</b> / <b>fork</b> commands.</div>
  <div id="detail"></div>
</div>
<script>
const DATA = __DATA__;
const nodes = new vis.DataSet(DATA.nodes);
const edges = new vis.DataSet(DATA.edges);
const net = new vis.Network(document.getElementById('net'), {nodes, edges}, {
  physics:{barnesHut:{gravitationalConstant:-8000,springLength:140,springConstant:0.03},stabilization:{iterations:250}},
  interaction:{hover:true,tooltipDelay:120},
  nodes:{scaling:{min:8,max:46}}, edges:{smooth:{type:'continuous'}}
});
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function copy(btn,text){navigator.clipboard.writeText(text).then(()=>{const o=btn.textContent;btn.textContent='copied!';setTimeout(()=>btn.textContent=o,1200);});}
function cmdBlock(label,cmd){
  return `<div><b>${label}</b><code id="c">${esc(cmd)}</code>`+
         `<button onclick="copy(this,${JSON.stringify(cmd)})">copy</button></div>`;
}
function show(id){
  const d=DATA.details[id]; const el=document.getElementById('detail');
  if(!d){el.innerHTML='<div class="hint" style="margin-top:18px">Topic hub — click a session dot.</div>';return;}
  let h=`<h2>${esc(d.title||d.ai_id)}</h2><div class="hint">ai-id ${d.ai_id} · ${d.n_human} prompts · ${d.n_assistant} replies</div>`;
  h+=cmdBlock('Resume (continue it)', d.resume);
  h+=cmdBlock('Fork (new branch, original kept)', d.fork);
  if(d.prompts&&d.prompts.length){h+='<h2>Opening prompts</h2><ul>'+d.prompts.map(p=>`<li>${esc(p.slice(0,160))}</li>`).join('')+'</ul>';}
  el.innerHTML=h;
}
net.on('click',p=>{ if(p.nodes.length) show(p.nodes[0]); });
show(null);
</script></body></html>
"""


def render_html(payload: dict) -> str:
    return HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))


def main(argv=None):
    p = argparse.ArgumentParser(description="Local web mindmap of session topics.")
    p.add_argument("--repo", default=os.getcwd())
    p.add_argument("--db")
    p.add_argument("--topics", default=str(sdb.DEFAULT_TOPICS))
    p.add_argument("--out", help="output html path (default ~/.claude/agent-fleet/mindmap.html)")
    p.add_argument("--serve", action="store_true", help="serve locally + open browser")
    p.add_argument("--port", type=int, default=8782)
    args = p.parse_args(argv)

    con = sdb.connect(args.db or sdb.db_path_for(args.repo))
    sdb.sync(con, str(sdb.project_dir_for(args.repo)))
    if not con.execute("SELECT count(*) c FROM topics").fetchone()["c"]:
        sdb.recompute_topics(con, sdb.load_topics(args.topics))
    payload = build_payload(sdb.export_graph(con))

    out = Path(args.out) if args.out else sdb.CACHE_DIR / "mindmap.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload))
    print(f"wrote {out}  ({len(payload['details'])} sessions, {len(payload['nodes'])} nodes, "
          f"{len(payload['edges'])} edges)")

    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out.parent))
        url = f"http://127.0.0.1:{args.port}/{out.name}"
        print(f"serving {url}  (Ctrl-C to stop)")
        webbrowser.open(url)
        with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
            httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
