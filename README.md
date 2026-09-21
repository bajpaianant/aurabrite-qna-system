# 🌿 AuraBrite Consumer Brands — Enterprise Multi-Agent QnA Prototype

A production-grade, end-to-end **multi-agent Question-and-Answer** prototype for
a fictional global FMCG conglomerate, **AuraBrite Consumer Brands**. It answers
cross-modal enterprise questions over a synthetic warehouse (DuckDB/SQLite),
narrative documents (hybrid RAG), web signals, and executable analytics — all
orchestrated by a hierarchical **supervisor / worker** agent graph with a
cyclical validation loop.

> The system is **fully offline by default** (deterministic mock LLM provider),
> and swaps in any real provider — OpenAI, Anthropic, or anything LiteLLM can
> reach — via a one-line change in `.env`.

---

## ✨ Highlights

| Capability | Implementation |
|---|---|
| Multi-agent orchestration | LangGraph-equivalent state graph in `agents/orchestrator.py`; optional real LangGraph runner in `agents/langgraph_runner.py`. |
| Structured analytics | DuckDB warehouse (SQLite fallback) with allow-listed read-only SQL tool. |
| Unstructured analytics | Hybrid **BM25 + TF-IDF** retriever with heading-boost fusion. |
| Executable analytics | AST-validated Python sandbox — no imports, no filesystem, no dunders. |
| External signals | Pluggable web-search tool (DuckDuckGo → Tavily → offline stub). |
| Cyclical validation | Deterministic validator + bounded re-plan loop. |
| Deterministic mock LLM | Keyword-routed responses so the whole graph runs without API keys. |
| Evaluation harness | Gold Q&A set + routing / citation / keyword recall metrics. |
| CLI + Streamlit UI + Notebook | `aurabrite ask`, `aurabrite chat`, `aurabrite ui`, pre-run `notebooks/demo.ipynb`. |

---

## 🏛️ Architecture at a glance

```
           ┌───────────────────┐
  user →   │   Supervisor      │  (plans which workers to invoke)
           └────────┬──────────┘
                    │ plan = [sql?, docs?, web?, python?]
     ┌─────────┬────┴────┬─────────┬──────────┐
     ▼         ▼         ▼         ▼          
  ┌─────┐  ┌──────┐  ┌──────┐  ┌──────────┐   
  │ SQL │  │ Docs │  │ Web  │  │ Python   │   
  │DuckD│  │BM25+ │  │DDG /  │  │AST-safe  │   
  │B/SQL│  │TF-IDF│  │Tavily │  │sandbox   │   
  └──┬──┘  └───┬──┘  └───┬──┘  └────┬─────┘   
     └─────────┴─────────┴──────────┘          
                    │ Evidence[]                
                    ▼                           
           ┌───────────────────┐                
           │   Validator       │ ── revise? ──┐  
           └────────┬──────────┘              │  
                    │ pass                    │  
                    ▼                         │  
           ┌───────────────────┐              │  
           │  Synthesizer      │              │  
           └────────┬──────────┘              │  
                    ▼                         │  
               final answer                   │  
                                              │  
         ▲───────────────── loop (bounded) ───┘  
```

See [`docs/architecture.md`](docs/architecture.md) for the full write-up.

---

## 🚀 Quickstart

```bash
# 1. Create a venv (Python 3.11+ required — DuckDB deps)
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install the package (in editable mode for hacking)
pip install -e ".[dev]"

# 3. Bootstrap the synthetic warehouse + narrative docs + RAG index
aurabrite init

# 4. Ask a question
aurabrite ask "Why did AuraGlow sales drop in APAC in Q4 2025, and by how much?"

# 5. Interactive REPL
aurabrite chat

# 6. Streamlit UI
aurabrite ui
```

Copy `.env.example` → `.env` if you want to swap the LLM provider. Default is
the deterministic offline mock — perfect for CI and demos on a plane.

---

## 🧪 Evaluation

An offline gold Q&A set drives the evaluation harness:

```bash
aurabrite eval --output /tmp/report.json
```

Baseline scores with the bundled mock provider (deterministic):

| Metric | Score |
|---|---|
| n_questions | 8 |
| pass_rate | 1.000 |
| avg_keyword_recall | 0.875 |
| avg_citation_recall | 1.000 |
| avg_routing_precision | 0.938 |
| avg_routing_recall | 1.000 |
| avg_latency_ms | 5.5 |

The gold set lives at [`src/aurabrite_qna/eval/gold_qna.json`](src/aurabrite_qna/eval/gold_qna.json) — extend it freely.

---

## 🧬 Data — deterministic but overlapping

`scripts/generate_synthetic_data.py` produces a **repeatable** warehouse and
narrative documents that *reference each other*. Selected overlaps enforced by
tests:

* **APAC AuraGlow + Dentafresh** revenue drops ≈ 14 % during the Port of
  Singapore strike (Oct 20 – Nov 12 2025). Root cause explained in
  `Supply_Chain_Disruption_Report_Q4_2025.md`.
* **AuraGlow Niacinamide Booster** drops ≈ 7 % globally Q4 2025 — same doc.
* **Dentafresh EMEA** drops ≈ 9 % due to palm-oil crunch (Nov 15 2025 – Jan 31
  2026).
* **NutriVita E-Commerce** grows ≈ 11 % in H1 2026 (clean-label lift, see
  `Consumer_Insights_Q1_2026.md`).
* Rebate rates in the retail-partnership document match the modelled discount
  rates in `fact_sales.discount_rate`.

Regenerate any time:

```bash
python scripts/generate_synthetic_data.py --seed 20260921
```

---

## 🧭 Package layout

```
src/aurabrite_qna/
├── config.py                # Frozen Settings from .env
├── cli.py                   # Typer CLI: init | ask | chat | eval | ui
├── data/                    # Schema + generator + warehouse wrapper
│   ├── schema.py
│   ├── generator.py
│   └── warehouse.py
├── llm/                     # Provider abstraction + mock + factory
│   ├── base.py
│   ├── mock.py
│   ├── providers.py         # OpenAI / Anthropic / LiteLLM
│   └── factory.py
├── rag/                     # Chunking, hybrid index, retriever
├── tools/                   # SQL, sandbox, web search
├── agents/                  # State, prompts, workers, orchestrator
│   ├── state.py
│   ├── prompts.py
│   ├── workers.py
│   ├── orchestrator.py
│   └── langgraph_runner.py  # optional real LangGraph topology
├── eval/                    # Gold set + runner
└── ui/                      # Streamlit app
```

---

## 🧪 Tests

```bash
pytest -q
```

17 hermetic tests covering:

* Warehouse row-counts + narrative consistency (APAC dip, NutriVita EC lift).
* SQL tool safety (rejects DDL, unknown tables, multi-statement).
* Sandbox safety (blocks `import`, dunder access, forbidden builtins).
* Hybrid RAG (finds supply-chain report, campaign-spending section).
* Agent orchestration (routing, evidence assembly, bounded validation).
* Offline gold-suite evaluation.

---

## 🔧 Configuration reference

All settings are optional and driven by env vars (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock` \| `openai` \| `anthropic` \| `litellm` |
| `LLM_MODEL` | `mock-supervisor` | Model name (provider-specific) |
| `OPENAI_API_KEY` | *(unset)* | Required for `openai` provider |
| `ANTHROPIC_API_KEY` | *(unset)* | Required for `anthropic` provider |
| `GEMINI_API_KEY` | *(unset)* | Read by LiteLLM when using a `gemini/…` model |
| `WEB_SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo` \| `tavily` \| `none` |
| `AURABRITE_WAREHOUSE_PATH` | `./data/warehouse/aurabrite.duckdb` | Warehouse file |
| `AURABRITE_MAX_AGENT_STEPS` | `8` | Safety cap on graph iterations |
| `AURABRITE_VALIDATION_ROUNDS` | `2` | Max validator re-plan loops |

---

## 📓 Notebook demo

A pre-executed walkthrough lives at
[`notebooks/demo.ipynb`](notebooks/demo.ipynb). Open it directly on GitHub —
no environment needed to read the outputs.

---

## ☁️ Deploy to Streamlit Community Cloud

The repo is deployment-ready. Files that make it work out-of-the-box:

| File | Purpose |
|---|---|
| `streamlit_app.py` (repo root) | Cloud's default entrypoint — thin shim that puts `src/` on `sys.path` and calls the real UI. |
| `requirements.txt` | Cloud's package installer. Mirrors `pyproject.toml` deps. |
| `.streamlit/config.toml` | Dark theme + server defaults. |
| `.streamlit/secrets.toml.example` | Template for the app's *Secrets* panel. |

### One-time deploy walkthrough

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the same
   GitHub account that owns the repo. Grant it access to the repo (private
   repos are supported on the Community tier — you just have to authorize the
   Streamlit GitHub App).
2. Click **"Create app"** → **"Deploy a public app from GitHub"**.
3. Fill in:
   * **Repository**: `bajpaianant/aurabrite-qna-system`
   * **Branch**: `cursor/fmcg-multi-agent-qna-prototype`
     (or `main` if you push it later)
   * **Main file path**: `streamlit_app.py`
   * **App URL (optional)**: e.g. `aurabrite-qna`
4. Open **"Advanced settings"** and paste the contents of
   `.streamlit/secrets.toml.example` (edit values first). Leaving it empty is
   also fine — the app defaults to the deterministic offline mock LLM.
5. Click **Deploy**. First build takes ~3 min while Cloud installs deps and
   warms up. On first load the app auto-runs the equivalent of
   `aurabrite init` (generates warehouse + docs + RAG index) in ~10 s.

### After deploy

* The public URL is `https://<slug>.streamlit.app`.
* Every push to the tracked branch triggers a rebuild.
* To use a real LLM, add the provider secrets in the app's *Secrets* panel.
  No code change needed. Example configs:

  ```toml
  # OpenAI
  LLM_PROVIDER = "openai"
  LLM_MODEL    = "gpt-4o-mini"
  OPENAI_API_KEY = "sk-..."

  # Anthropic
  LLM_PROVIDER = "anthropic"
  LLM_MODEL    = "claude-3-5-sonnet-latest"
  ANTHROPIC_API_KEY = "sk-ant-..."

  # Google Gemini (via LiteLLM — GEMINI_API_KEY is auto-detected)
  LLM_PROVIDER = "litellm"
  LLM_MODEL    = "gemini/gemini-2.5-flash"
  GEMINI_API_KEY = "AIza..."
  ```
  After editing secrets, click **Manage app → ⋮ → Reboot app** to pick up
  the new values (Python module-level `SETTINGS` is captured on first import).
* Cloud caches the warehouse to the container's writable disk; it will
  regenerate on cold starts (deterministic seed, ~10 s).

---

## 🛡️ Safety perimeters

* **SQL** — only `SELECT` / `WITH` on the allow-listed tables; multi-statement,
  DDL, DML, PRAGMA, ATTACH are refused.
* **Python** — AST validation blocks `import`, dunder attribute access, and any
  reference to `eval`, `exec`, `compile`, `open`, `__import__`, etc.
* **LLM output** — the *final answer* is composed by a deterministic templater,
  not by the LLM. Every claim is backed by an evidence item (SQL row, doc
  chunk, web hit, or sandbox result). This eliminates hallucination on the
  synthesis boundary.

---

## 🗺️ Roadmap ideas

* Swap the TF-IDF vector channel for real sentence-transformer embeddings when
  offline weights are permitted.
* Add a graph-cache to short-circuit repeated identical questions.
* Wire the LangGraph runner to `langgraph.checkpoint.SqliteSaver` for persistent
  conversation state.
* Add a "Fact Card" export button in the Streamlit UI.
