from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dynamic_gguf import GGML_TYPE_BPW


TYPE_COLORS = {
    "F16": "#e6e0d0",
    "BF16": "#e6e0d0",
    "Q3_K": "#2f73b8",
    "Q4_K": "#75acd6",
    "Q5_K": "#c96f22",
    "Q6_K": "#8f493e",
    "Q8_0": "#1d1717",
    "IQ4_NL": "#7b8f4a",
}


@dataclass(frozen=True)
class AllocationUIOptions:
    plan_path: Path
    output_dir: Path
    title: str = "OpenTQ Allocation Explorer"
    metrics_path: Path | None = None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_metrics(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    if isinstance(payload.get("tensors"), list):
        return {str(row.get("name") or row.get("hf_name")): row for row in payload["tensors"]}
    if isinstance(payload, dict):
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    return {}


def build_allocation_ui_data(options: AllocationUIOptions) -> dict[str, Any]:
    plan = _load_json(options.plan_path)
    metrics = _load_metrics(options.metrics_path)

    tensors: list[dict[str, Any]] = []
    for index, row in enumerate(plan.get("tensors", [])):
        if row.get("mode") == "skip":
            continue
        ggml_type = str(row.get("ggml_type") or "F16")
        metric = metrics.get(str(row.get("hf_name"))) or metrics.get(str(row.get("gguf_name"))) or {}
        tensors.append(
            {
                "id": index,
                "hf_name": row.get("hf_name"),
                "gguf_name": row.get("gguf_name"),
                "category": row.get("category"),
                "layer": row.get("layer_index"),
                "mode": row.get("mode"),
                "ggml_type": ggml_type,
                "reason": row.get("reason"),
                "relative_size": round(float(GGML_TYPE_BPW.get(ggml_type, 16.0)), 4),
                "color": TYPE_COLORS.get(ggml_type, "#9b9488"),
                "metrics": metric,
            }
        )

    by_category = Counter(str(row["category"]) for row in tensors)
    by_type = Counter(str(row["ggml_type"]) for row in tensors)
    by_layer = Counter(str(row["layer"]) for row in tensors if row.get("layer") is not None)

    return {
        "schema": "opentq.allocation_ui_data.v1",
        "title": options.title,
        "source_plan": str(options.plan_path),
        "model_id": plan.get("model_id"),
        "profile": plan.get("profile", {}).get("name"),
        "summary": {
            "tensor_count": len(tensors),
            "category_counts": dict(sorted(by_category.items())),
            "type_counts": dict(sorted(by_type.items())),
            "layer_count": len(by_layer),
            "metrics_attached": bool(metrics),
        },
        "legend": [{"type": key, "color": value} for key, value in TYPE_COLORS.items()],
        "tensors": tensors,
    }


def _standalone_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(str(payload["title"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --ink: #191816;
      --muted: #6f6a61;
      --paper: #faf8f2;
      --line: #ded8ca;
      --blue: #1747ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 32px; }}
    h1 {{ font-size: 42px; margin: 0 0 6px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 16px; margin-bottom: 24px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 220px 220px;
      gap: 12px;
      margin: 18px 0 20px;
    }}
    input, select {{
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 24px;
      align-items: start;
    }}
    .treemap {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(26px, 1fr));
      gap: 3px;
      padding: 14px;
      border: 1px solid var(--line);
      background: #f0ece2;
      min-height: 580px;
    }}
    .cell {{
      min-height: 24px;
      border: 1px solid rgba(25, 24, 22, 0.18);
      cursor: pointer;
      position: relative;
    }}
    .cell:hover {{ outline: 2px solid var(--blue); z-index: 2; }}
    aside {{
      border: 1px solid var(--line);
      background: #fffdf8;
      padding: 18px;
      border-radius: 8px;
      position: sticky;
      top: 18px;
    }}
    .metric {{
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      padding: 8px 0;
      gap: 18px;
    }}
    .label {{ color: var(--muted); }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      background: #fffdf8;
    }}
    .swatch {{ width: 12px; height: 12px; border-radius: 3px; border: 1px solid rgba(0,0,0,.18); }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    @media (max-width: 900px) {{
      main {{ padding: 18px; }}
      .toolbar, .grid {{ grid-template-columns: 1fr; }}
      aside {{ position: static; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <div class="sub" id="subtitle"></div>
  <div class="legend" id="legend"></div>
  <div class="toolbar">
    <input id="search" placeholder="Filter tensor name" />
    <select id="category"></select>
    <select id="type"></select>
  </div>
  <div class="grid">
    <section class="treemap" id="treemap" aria-label="Tensor treemap"></section>
    <aside>
      <h2 id="side-title">Select a tensor</h2>
      <div id="details"></div>
    </aside>
  </div>
</main>
<script>
const DATA = {data};
const state = {{ search: "", category: "all", type: "all", selected: DATA.tensors[0] }};
const el = id => document.getElementById(id);
function options(values, label) {{
  return `<option value="all">${{label}}</option>` + [...values].sort().map(v => `<option value="${{v}}">${{v}}</option>`).join("");
}}
function renderFilters() {{
  el("category").innerHTML = options(new Set(DATA.tensors.map(t => t.category)), "all families");
  el("type").innerHTML = options(new Set(DATA.tensors.map(t => t.ggml_type)), "all types");
}}
function filtered() {{
  const q = state.search.toLowerCase();
  return DATA.tensors.filter(t =>
    (state.category === "all" || t.category === state.category) &&
    (state.type === "all" || t.ggml_type === state.type) &&
    (!q || String(t.hf_name).toLowerCase().includes(q) || String(t.gguf_name).toLowerCase().includes(q))
  );
}}
function renderTreemap() {{
  const rows = filtered();
  el("subtitle").textContent = `${{DATA.model_id}} · ${{DATA.profile}} · ${{rows.length}}/${{DATA.summary.tensor_count}} tensors visible`;
  el("treemap").innerHTML = rows.map((t, i) =>
    `<button class="cell" data-i="${{i}}" title="${{t.hf_name}}" style="background:${{t.color}}"></button>`
  ).join("");
  [...el("treemap").children].forEach((node, i) => {{
    node.onclick = () => {{ state.selected = rows[i]; renderDetails(); }};
  }});
}}
function renderLegend() {{
  el("legend").innerHTML = DATA.legend.map(row =>
    `<span class="pill"><span class="swatch" style="background:${{row.color}}"></span>${{row.type}}</span>`
  ).join("");
}}
function metric(label, value) {{ return `<div class="metric"><span class="label">${{label}}</span><strong>${{value ?? ""}}</strong></div>`; }}
function renderDetails() {{
  const t = state.selected;
  el("side-title").textContent = t ? t.category : "Select a tensor";
  if (!t) {{ el("details").innerHTML = ""; return; }}
  el("details").innerHTML = [
    metric("GGUF type", `<code>${{t.ggml_type}}</code>`),
    metric("Layer", t.layer ?? "global"),
    metric("Mode", t.mode),
    metric("Relative size", t.relative_size),
    `<p><strong>Reason</strong><br>${{t.reason}}</p>`,
    `<p><strong>HF tensor</strong><br><code>${{t.hf_name}}</code></p>`,
    `<p><strong>GGUF tensor</strong><br><code>${{t.gguf_name}}</code></p>`
  ].join("");
}}
el("search").oninput = e => {{ state.search = e.target.value; renderTreemap(); }};
el("category").onchange = e => {{ state.category = e.target.value; renderTreemap(); }};
el("type").onchange = e => {{ state.type = e.target.value; renderTreemap(); }};
renderFilters(); renderLegend(); renderTreemap(); renderDetails();
</script>
</body>
</html>
"""


def write_allocation_ui(options: AllocationUIOptions) -> dict[str, Any]:
    payload = build_allocation_ui_data(options)
    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "allocation-ui-data.json"
    html_path = output_dir / "index.html"
    readme_path = output_dir / "README.md"

    data_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(_standalone_html(payload), encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# OpenTQ Allocation UI Artifact",
                "",
                f"- Source plan: `{payload['source_plan']}`",
                f"- Tensor count: `{payload['summary']['tensor_count']}`",
                f"- Metrics attached: `{payload['summary']['metrics_attached']}`",
                "",
                "Open `index.html` locally to inspect tensor-family allocation, layer filters, tensor-type filters, and policy rationale.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload["outputs"] = {
        "data": str(data_path),
        "html": str(html_path),
        "readme": str(readme_path),
    }
    return payload
