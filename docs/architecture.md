# Architecture

A single-repo, single-process reference implementation of an enterprise
multi-agent QnA system. The design goals are:

1. **Correctness > cleverness.** Every claim in the final answer must trace
   back to a concrete evidence item.
2. **Offline first.** A first-time contributor should be able to `pip install`
   and `aurabrite ask` with zero API keys.
3. **Provider agnosticism.** LLM providers are behind a single 40-line
   interface (`LLMClient`). Swapping OpenAI ↔ Anthropic ↔ mock is a one-line
   `.env` change.
4. **Small, honest surface.** No hidden network calls, no runtime plugin
   loading, no monkey-patching. Everything is discoverable via `grep`.

---

## 1. Layers

```
┌──────────────────────────────────────────────────────────────────┐
│  UI       CLI  |  Streamlit  |  Jupyter Notebook                 │
├──────────────────────────────────────────────────────────────────┤
│  AGENTS   Supervisor  ──▶  Workers  ──▶  Validator  ──▶  Synth   │
├──────────────────────────────────────────────────────────────────┤
│  TOOLS    SQLTool  |  HybridRetriever  |  WebSearchTool  |  Sandbox │
├──────────────────────────────────────────────────────────────────┤
│  DATA     DuckDB / SQLite warehouse    +   Markdown documents      │
├──────────────────────────────────────────────────────────────────┤
│  LLM      MockLLMClient  |  LiteLLM  |  OpenAI  |  Anthropic       │
└──────────────────────────────────────────────────────────────────┘
```

### Why this separation?

* Every layer has a **stable, narrow public interface**. The `agents/` layer
  never imports LangChain classes directly; it depends on the `LLMClient`
  Protocol and the tool classes. This is what makes the mock provider possible.
* The `tools/` layer is pure business logic — no LLM knowledge. Each tool can
  be unit-tested independently (see `tests/test_tools.py`).
* The `data/` layer is deterministic. The generator is seeded and the schema
  is a single source of truth (`data/schema.py`).

---

## 2. Multi-agent graph

Nodes:

| Node | Type | Reads from | Writes to |
|---|---|---|---|
| `supervisor` | Planner | `question` | `plan[]` |
| `sql_analyst` | Worker | `question`, `schema` | `evidence[kind=sql]` |
| `doc_researcher` | Worker | `question` | `evidence[kind=docs]` |
| `web_researcher` | Worker | `question` | `evidence[kind=web]` |
| `python_analyst` | Worker | `evidence[kind=sql].rows` | `evidence[kind=python]` |
| `validator` | Critic | full `evidence`, `plan` | `validator_verdict`, `issues`, `confidence` |
| `synthesizer` | Writer | full `evidence` | `answer`, `done=True` |

### Control flow

Pseudo-code from `orchestrator.py`:

```python
while not state.done and steps < max_steps:
    if not state.plan or state.validator_verdict == "revise":
        state = supervisor(state)
    for step in state.plan:
        state = worker_for(step.agent)(state)
    state = validator(state)
    if verdict == "pass" or rounds >= max_rounds:
        state = synthesizer(state)
        break
```

Two invariants are enforced at every iteration:

* **Bounded work** — `max_steps` (default 8) and `max_validation_rounds`
  (default 2) prevent runaway loops.
* **Evidence-only synthesis** — the synthesizer template *never* injects text
  that isn't already in `state.evidence`.

### Cyclical validation

The validator combines:

* Deterministic pre-checks (missing worker output, empty SQL result set),
  which cannot be talked out of by a fluent LLM.
* An LLM verdict for semantic gaps.

If either fires, `verdict = "revise"` and the loop re-invokes the supervisor
with the accumulated evidence attached, letting it plan additional workers.

---

## 3. Tool contracts

### SQLTool

* Backed by DuckDB (falls back to SQLite when DuckDB isn't importable).
* `validate(sql)`: allow-lists `SELECT`/`WITH`, blocks DDL/DML/PRAGMA/ATTACH,
  rejects multi-statement input, forces a `LIMIT`, checks table names against
  the allow-list.
* `run(sql)`: returns a `SQLExecution` with typed rows + Markdown preview.

### PythonSandbox

* Parses the code with `ast`, walks it and rejects:
  * `Import` / `ImportFrom`
  * `Attribute` whose name starts with `__`
  * Any `Name` or `Call` targeting `eval`, `exec`, `compile`, `open`,
    `__import__`, `getattr`, `setattr`, `input`, `globals`, `locals`, `vars`.
* Executes in a locked-down namespace exposing only `math`, `statistics`, and
  an allow-list of safe builtins.

### HybridRetriever

Score = **RRF( BM25, TF-IDF cosine ) + heading-boost**.

The heading boost — `0.02 × |query_terms ∩ heading_terms|` — is a cheap way
to break ties, and empirically raises the "Campaign Spending" section above
sibling sections for the query *"AuraGlow campaign spending"*.

Chunks are heading-aware, so citations look like:

```
Supply_Chain_Disruption_Report_Q4_2025.md § Impact on APAC (AuraGlow & Dentafresh)
```

### WebSearchTool

Adapter chain: **Tavily → DuckDuckGo → offline stub**. The stub returns
curated snippets keyed off topical keywords so the graph exercises the
`web` code path even on a laptop with no internet.

---

## 4. Data model

Schema (see `data/schema.py`):

* **Dimensions**: `dim_product`, `dim_geography`, `dim_channel`,
  `dim_warehouse`, `dim_kpi_metadata`.
* **Facts**: `fact_sales(date, sku_id, brand_id, market_id, channel_id,
  gross_revenue_usd, net_revenue_usd, volume_units, cogs_usd,
  discount_rate)`, `fact_inventory(date, sku_id, warehouse_id, stock_on_hand,
  days_of_inventory_outstanding, out_of_stock_events)`.

Granularity: **monthly**, Jan 2024 → Jun 2026.

Cross-modal integrity is enforced by `tests/test_data_generator.py` — any
change to the generator that breaks the narrative claims trips CI.

---

## 5. LLM providers

`llm/factory.py` picks a provider based on `SETTINGS.llm_provider`:

| Provider | Requires | Fallback if missing |
|---|---|---|
| `mock` | nothing | itself (default) |
| `litellm` | `litellm` package | mock |
| `openai` | `OPENAI_API_KEY` + `openai` package | mock |
| `anthropic` | `ANTHROPIC_API_KEY` + `anthropic` package | mock |

The `MockLLMClient` is not a random word generator — it inspects the system
prompt's `[[AGENT: xxx]]` tag and returns purposeful JSON so the whole graph
exercises real code paths (plan, SQL, docs queries, sandbox code,
validator verdict). This makes the evaluation harness deterministic and CI
possible without a network.

---

## 6. Evaluation

`eval/runner.py` loads a gold JSON file (`eval/gold_qna.json`) and for each
item measures:

* `keyword_recall` — fraction of expected keywords in the final answer.
* `citation_recall` — fraction of expected documents cited in the assembled
  evidence.
* `routing_precision / recall` — vs. the expected agent set.
* `latency_ms` — end-to-end.
* `passed` — combined heuristic (all recalls ≥ 0.5).

Because the mock provider is deterministic, `aurabrite eval` returns the same
numbers every run — perfect regression signal.

---

## 7. Failure modes & mitigations

| Failure | Mitigation |
|---|---|
| LLM returns non-JSON | `llm.base.extract_json` tolerates fenced blocks and balances braces to recover. |
| SQL agent hallucinates unknown tables | `SQLTool.validate` rejects at the boundary. |
| Sandbox code tries `import os` | AST validation raises `SandboxError` before execution. |
| Network dead | Web tool falls back to offline stub; RAG + SQL still work. |
| Retriever misses the answer | Multi-query aggregation + heading-boost + max-score fusion. |
| Runaway loop | `max_steps` and `max_validation_rounds` caps. |

---

## 8. Extending the system

* **Add a new document** → drop a `.md` file in `data/documents/` and run
  `aurabrite init` (or delete `data/warehouse/vector_index/` to trigger a
  rebuild).
* **Add a KPI** → append to `KPI_METADATA` in `data/schema.py`. It appears in
  `dim_kpi_metadata` automatically.
* **Add a new worker** → subclass or write a dataclass in `agents/workers.py`,
  register in `Orchestrator._resolve_worker`, and extend the mock supervisor
  in `llm/mock.py` (or your real LLM's system prompt) to route to it.
* **Wire a real LangGraph runtime** → import
  `aurabrite_qna.agents.langgraph_runner.build_langgraph_app` and call
  `.invoke({...})`.
