"""HTML for the Paper Map page: a d3 force-directed graph rendered inside a Streamlit
HTML component. Takes the graph from `linkmap.build_link_graph`."""
import json

D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"

_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  :root { --bg: #fff; --panel: rgba(255,255,255,0.92); --border: #e6e6e6; --text: #222; --muted: #777;
          --link: #9aa4b1; --ring: #fff; --hit: #111; --tip-bg: #222; --tip-fg: #fff; --tip-meta: #cfd4da; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0e1117; --panel: rgba(22,26,34,0.92); --border: #30363d; --text: #e6e6e6; --muted: #9a9a9a;
            --link: #8b95a5; --ring: #0e1117; --hit: #fff; --tip-bg: #f0f2f6; --tip-fg: #111; --tip-meta: #444; }
  }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--text);
               font-family: "Source Sans Pro", -apple-system, "Segoe UI", sans-serif; }
  #wrap { position: relative; width: 100%; height: __HEIGHT__px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  svg { width: 100%; height: 100%; cursor: grab; display: block; }
  svg:active { cursor: grabbing; }
  .link { stroke: var(--link); stroke-opacity: 0.35; }
  .node circle { stroke: var(--ring); stroke-width: 1.2px; cursor: pointer; }
  .node.hit circle { stroke: var(--hit); stroke-width: 2.5px; }
  .node text { font-size: 11px; fill: var(--text); pointer-events: none; paint-order: stroke; stroke: var(--bg); stroke-width: 3px; }
  #legend { position: absolute; top: 10px; left: 10px; background: var(--panel); border: 1px solid var(--border);
            border-radius: 6px; padding: 8px 10px; font-size: 12px; max-height: calc(100% - 20px); overflow: auto; max-width: 260px; }
  #legend div { display: flex; align-items: center; gap: 6px; padding: 2px 0; cursor: pointer; user-select: none; }
  #legend div.off { opacity: 0.35; }
  #legend span.sw { width: 11px; height: 11px; border-radius: 50%; flex: none; display: inline-block; }
  #legend .hint { color: var(--muted); font-size: 11px; cursor: default; margin-top: 4px; }
  #tip { position: absolute; display: none; background: var(--tip-bg); color: var(--tip-fg); padding: 7px 10px; border-radius: 6px;
         font-size: 12px; max-width: 320px; pointer-events: none; line-height: 1.35; }
  #tip b { font-weight: 600; }
  #tip .meta { color: var(--tip-meta); }
</style></head>
<body>
<div id="wrap">
  <svg></svg>
  <div id="legend"></div>
  <div id="tip"></div>
</div>
<script src="__D3__"></script>
<script>
const graph = __GRAPH__;
const highlight = __HIGHLIGHT__.trim().toLowerCase();

const wrap = document.getElementById("wrap");
const svg = d3.select("svg");
const width = wrap.clientWidth, height = wrap.clientHeight;
const tip = document.getElementById("tip");

const topics = Array.from(new Set(graph.nodes.map(n => n.topic)));
const counts = {};
graph.nodes.forEach(n => counts[n.topic] = (counts[n.topic] || 0) + 1);
topics.sort((a, b) => counts[b] - counts[a]);
const palette = topics.length <= 10 ? d3.schemeTableau10 : d3.quantize(t => d3.interpolateRainbow(t * 0.92), topics.length);
const color = d3.scaleOrdinal(topics, palette);

const nodes = graph.nodes.map(n => ({...n}));
const byKey = new Map(nodes.map(n => [n.key, n]));
const links = graph.edges.map(e => ({source: e.source, target: e.target, weight: e.weight}));
const neighbours = new Map(nodes.map(n => [n.key, new Set()]));
links.forEach(l => { neighbours.get(l.source).add(l.target); neighbours.get(l.target).add(l.source); });

const radius = d3.scaleSqrt().domain([0, d3.max(nodes, n => n.degree) || 1]).range([3.5, 11]);
const strokeW = d3.scaleLinear().domain([0.4, 1]).range([0.5, 2.2]).clamp(true);
const isHit = n => highlight && (n.title.toLowerCase().includes(highlight) || n.cite.toLowerCase().includes(highlight));
nodes.forEach(n => n.hit = isHit(n));

const g = svg.append("g");
const linkSel = g.append("g").selectAll("line").data(links).join("line")
  .attr("class", "link").attr("stroke-width", l => strokeW(l.weight));
const nodeSel = g.append("g").selectAll("g").data(nodes).join("g")
  .attr("class", n => "node" + (n.hit ? " hit" : ""));
nodeSel.append("circle").attr("r", n => radius(n.degree) + (n.hit ? 2 : 0)).attr("fill", n => color(n.topic));
const labelSel = nodeSel.append("text").attr("dx", n => radius(n.degree) + 3).attr("dy", "0.35em")
  .text(n => n.cite).style("display", n => n.hit ? null : "none");

svg.call(d3.zoom().scaleExtent([0.2, 8]).on("zoom", ev => g.attr("transform", ev.transform)));

const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(n => n.key).distance(l => 18 + 70 * (1 - l.weight)).strength(l => 0.3 + 0.7 * l.weight))
  .force("charge", d3.forceManyBody().strength(-28).distanceMax(320))
  .force("collide", d3.forceCollide(n => radius(n.degree) + 1.5))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("x", d3.forceX(width / 2).strength(0.04))
  .force("y", d3.forceY(height / 2).strength(0.06))
  .on("tick", () => {
    linkSel.attr("x1", l => l.source.x).attr("y1", l => l.source.y).attr("x2", l => l.target.x).attr("y2", l => l.target.y);
    nodeSel.attr("transform", n => `translate(${n.x},${n.y})`);
  });

nodeSel.call(d3.drag()
  .on("start", (ev, n) => { if (!ev.active) sim.alphaTarget(0.25).restart(); n.fx = n.x; n.fy = n.y; })
  .on("drag", (ev, n) => { n.fx = ev.x; n.fy = ev.y; })
  .on("end", (ev, n) => { if (!ev.active) sim.alphaTarget(0); n.fx = null; n.fy = null; }));

const hidden = new Set();
function applyVisibility(focus) {
  const near = focus ? neighbours.get(focus.key) : null;
  nodeSel.style("opacity", n => hidden.has(n.topic) ? 0.08 : (!focus || n === focus || near.has(n.key) ? 1 : 0.15));
  linkSel.style("stroke-opacity", l => {
    if (hidden.has(l.source.topic) || hidden.has(l.target.topic)) return 0.03;
    if (!focus) return 0.35;
    return (l.source === focus || l.target === focus) ? 0.9 : 0.05;
  }).style("stroke", l => focus && (l.source === focus || l.target === focus) ? "#333" : null);
  labelSel.style("display", n => n.hit || (focus && (n === focus || near.has(n.key))) ? null : "none");
}

nodeSel.on("mouseenter", (ev, n) => {
    applyVisibility(n);
    tip.innerHTML = `<b>${esc(n.title)}</b><br><span class="meta">${esc(n.cite)} · ${esc(n.topic)} · ${n.degree} link${n.degree === 1 ? "" : "s"}</span><br><span class="meta">Click to open</span>`;
    tip.style.display = "block";
    moveTip(ev);
  })
  .on("mousemove", moveTip)
  .on("mouseleave", () => { applyVisibility(null); tip.style.display = "none"; })
  .on("click", (ev, n) => { ev.stopPropagation(); window.open(n.url, "_blank", "noopener"); });

function moveTip(ev) {
  const r = wrap.getBoundingClientRect();
  let x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
  if (x + tip.offsetWidth > r.width - 8) x = ev.clientX - r.left - tip.offsetWidth - 10;
  if (y + tip.offsetHeight > r.height - 8) y = ev.clientY - r.top - tip.offsetHeight - 10;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

const legend = d3.select("#legend");
legend.selectAll("div.t").data(topics).join("div").attr("class", "t")
  .html(t => `<span class="sw" style="background:${color(t)}"></span><span>${esc(t)} (${counts[t]})</span>`)
  .on("click", (ev, t) => {
    if (hidden.has(t)) hidden.delete(t); else hidden.add(t);
    legend.selectAll("div.t").classed("off", d => hidden.has(d));
    applyVisibility(null);
  });
legend.append("div").attr("class", "hint").text("Click a topic to fade it. Hover a paper for its links, click to open.");
applyVisibility(null);
</script>
</body></html>
"""


def render_map_html(graph: dict, highlight: str = "", height: int = 720) -> str:
    """Self-contained page for `st.components.v1.html`."""
    def js(obj) -> str:
        return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__HEIGHT__", str(int(height)))
            .replace("__D3__", D3_URL)
            .replace("__GRAPH__", js(graph))
            .replace("__HIGHLIGHT__", js(highlight or "")))
