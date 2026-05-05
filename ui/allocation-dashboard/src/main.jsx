import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const fallbackData = {
  title: "OpenTQ Allocation Dashboard",
  model_id: "Qwen/Qwen3.6-27B",
  profile: "load allocation-ui-data.json",
  summary: { tensor_count: 0, category_counts: {}, type_counts: {} },
  legend: [],
  tensors: [],
};

function App() {
  const [data, setData] = useState(fallbackData);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [type, setType] = useState("all");
  const [selected, setSelected] = useState(null);

  useEffect(() => {
    fetch("/allocation-ui-data.json")
      .then((response) => (response.ok ? response.json() : fallbackData))
      .then((payload) => {
        setData(payload);
        setSelected(payload.tensors?.[0] ?? null);
      })
      .catch(() => setData(fallbackData));
  }, []);

  const categories = useMemo(() => [...new Set(data.tensors.map((tensor) => tensor.category))].sort(), [data]);
  const types = useMemo(() => [...new Set(data.tensors.map((tensor) => tensor.ggml_type))].sort(), [data]);
  const visible = useMemo(() => {
    const lowered = query.toLowerCase();
    return data.tensors.filter((tensor) => {
      const text = `${tensor.hf_name} ${tensor.gguf_name}`.toLowerCase();
      return (
        (category === "all" || tensor.category === category) &&
        (type === "all" || tensor.ggml_type === type) &&
        (!lowered || text.includes(lowered))
      );
    });
  }, [category, data, query, type]);

  return (
    <main>
      <header>
        <h1>{data.title}</h1>
        <p>
          {data.model_id} · {data.profile} · {visible.length}/{data.summary.tensor_count} tensors visible
        </p>
      </header>

      <section className="legend">
        {data.legend.map((row) => (
          <span className="pill" key={row.type}>
            <span className="swatch" style={{ background: row.color }} />
            {row.type}
          </span>
        ))}
      </section>

      <section className="toolbar">
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter tensor name" />
        <select value={category} onChange={(event) => setCategory(event.target.value)}>
          <option value="all">all families</option>
          {categories.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
        <select value={type} onChange={(event) => setType(event.target.value)}>
          <option value="all">all types</option>
          {types.map((item) => (
            <option key={item} value={item}>
              {item}
            </option>
          ))}
        </select>
      </section>

      <section className="layout">
        <div className="treemap">
          {visible.map((tensor) => (
            <button
              className="cell"
              key={tensor.id}
              title={tensor.hf_name}
              style={{ background: tensor.color }}
              onClick={() => setSelected(tensor)}
            />
          ))}
        </div>
        <aside>
          <h2>{selected?.category ?? "Select a tensor"}</h2>
          {selected ? (
            <dl>
              <dt>GGUF type</dt>
              <dd>{selected.ggml_type}</dd>
              <dt>Layer</dt>
              <dd>{selected.layer ?? "global"}</dd>
              <dt>Reason</dt>
              <dd>{selected.reason}</dd>
              <dt>HF tensor</dt>
              <dd>{selected.hf_name}</dd>
              <dt>GGUF tensor</dt>
              <dd>{selected.gguf_name}</dd>
            </dl>
          ) : null}
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
