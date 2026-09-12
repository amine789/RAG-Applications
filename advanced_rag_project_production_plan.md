# Neuroscience Corrective/Self-RAG Assistant — Production-Grade Plan

Builds on `advanded_rag_project.txt`. That file is the *concept* (what the
pipeline does); this file is the *system* (how it stays correct, cheap,
observable, and deployable once someone other than you is hitting it).

Assumption stated up front: "production" here means "a real service another
person could call and depend on" — not hyperscale. Sized for a single-region
deployment serving a scoped corpus, not millions of QPS. Revisit the scaling
section if that assumption changes.

---

## 0. Phasing

Build in this order — each phase should be runnable and demoable before the
next starts. Don't build the API layer before the eval-gated pipeline works;
don't add observability before there's something to observe.

0. **Phase 0 — Scaffolding.** Real package layout (`src/rag/` with
   `ingest/`, `retrieval/`, `grading/`, `generation/`, `orchestration/`
   subpackages), split `requirements.txt`/`requirements-dev.txt`,
   `.env.example` (+ confirm `.env` is gitignored), a `config.yaml` for
   tunables (chunk size, model IDs per stage, correction-round cap, eval
   thresholds), and an empty `tests/` wired to `pytest` so Phase 1 writes
   tests as it goes. No Docker, no API, no CI yet — those stay in Phase 2.
1. **Phase 1 — Correct pipeline.** Ingestion → hybrid retrieval → grading →
   correction → generation, as plain testable Python functions. Eval suite
   passing with a documented before/after (corrective loop on vs. off).
2. **Phase 2 — Service.** Wrap the pipeline in a FastAPI service with a real
   request/response contract, containerize it, stand up CI (lint, unit
   tests, eval gate).
3. **Phase 3 — Operability.** Structured logging, tracing, cost/latency
   metrics, alerting, a written incident runbook.
4. **Phase 4 — Deploy + polish.** Deployed environment (dev/staging/prod),
   demo UI calling the API (not the pipeline directly), docs.

### 10-day schedule

Assumes basic RAG experience (you have it) but no prior experience with the
grading/correction loop specifically — that's where the slack lives.

| Day | Work | Phase |
|---|---|---|
| 1 | Scaffolding (`src/rag/` layout, config.yaml, `.env.example`, `tests/` wired to pytest) + corpus selection/collection | 0 |
| 2 | Ingestion & chunking (loaders, token-aware splitter, metadata for citations) | 1 |
| 3 | Hybrid indexing: chromadb vector store + BM25, reciprocal rank fusion | 1 |
| 4 | Retrieval grading (binary, Haiku) | 1 |
| 5 | Corrective actions: rewrite/retry, web fallback, capped-round state machine | 1 |
| 6 | Answer generation with citations + draft eval set (10-30 Q&A pairs) | 1 |
| 7 | Run eval suite, produce before/after (corrective loop on vs. off), iterate on grading threshold/prompts | 1 |
| 8 | FastAPI wrapper (Pydantic schemas, `/query` endpoint) + Dockerfile + basic CI (lint, unit tests, eval gate) | 2 |
| 9 | Structured logging + cost/latency-per-query logging + security pass (prompt-injection guard on fetched web content, secrets check) | 3 (trimmed) |
| 10 | Demo UI (Streamlit calling the API) + README + short runbook + buffer | 4 |

Day 7 and the trailing "buffer" on Day 10 are the two days most likely to
slip — grading/correction tuning always takes longer than expected the
first time, and it's fine to borrow from Day 10's polish to cover it.
Multi-environment deploy and dashboards/alerting stay out of scope at 10
days too — "log it, don't dashboard it" still applies, just with real
structured logs instead of print statements.

---

## 1. Architecture

Two clearly separated paths — conflating them is the most common reason
prototypes never become services:

- **Offline/batch path** (ingestion): runs on a schedule or on-demand
  trigger, not per-request. Loads sources → chunks → embeds → upserts into
  the vector store and BM25 index. Idempotent — re-running it on an
  unchanged corpus should not duplicate vectors or bump costs.
- **Online/serving path** (query time): retrieve → grade → correct
  (rewrite/retry or web fallback) → generate → respond. Every stage is a
  pure-ish function with a typed input/output, independently unit-testable
  with mocked LLM calls — not just nodes wired into a LangGraph graph you
  can only debug by running the whole graph.

Keep the orchestration layer (LangGraph or a hand-rolled state machine — see
§7) thin. It should call into the stage functions, not contain business
logic itself.

---

## 2. Data & ingestion pipeline

- **Corpus manifest**: a versioned list of sources (URL/DOI, license,
  fetched date, content hash). Check redistribution license before bulk
  storing PMC/arXiv text — some publishers restrict full-text redistribution
  even for open-access papers.
- **Reproducible chunking**: chunking parameters (size, overlap, splitter
  type) live in a config file, not hardcoded in a notebook cell, so a corpus
  rebuild is deterministic and diffable.
- **Idempotent upserts**: key each chunk by `hash(source_id, chunk_index,
  chunk_text)` so re-ingestion updates changed chunks only, instead of
  duplicating the whole index.
- **Re-ingestion trigger**: manual CLI command is fine at this scale
  (`python -m ingest run`) — don't build a scheduler until there's a reason
  the corpus changes on its own cadence.

---

## 3. Storage & indexing

- **Vector store**: move off raw in-process FAISS for the serving path.
  `chromadb` is already in `requirements.txt` and gives you persistence,
  metadata filtering (filter by paper/topic before similarity search), and
  incremental upsert/delete without a full index rebuild — the things FAISS
  doesn't provide on its own (see the FAISS-vs-vector-DB distinction from
  earlier in this conversation). Keep FAISS only for offline experimentation
  if useful, not the serving path.
- **Keyword index**: `rank_bm25` in-process is fine at this corpus size
  (tens to low hundreds of papers). Only move to a real search engine
  (OpenSearch) if the corpus grows past what fits comfortably in memory.
- **Fusion**: reciprocal rank fusion between vector and BM25 results, as
  planned. Log the fused ranking, not just the final top-k, so retrieval
  quality is debuggable after the fact.

---

## 4. Retrieval grading & correction

- **Binary grading first** (relevant / not relevant per chunk or per
  retrieval set) — a 1-5 score adds a threshold you'll have to tune, and
  that tuning time is better spent on the eval suite. Add scoring later only
  if the eval shows binary is too coarse.
- **Model tiering**: use `claude-haiku-4-5` for the grading/rewrite calls —
  it's a cheap classification-shaped task and doesn't need Opus-level
  reasoning. Reserve `claude-opus-5` for final answer generation, where
  quality actually matters to the user. This is the single biggest cost
  lever in the whole system.
- **Explicit correction policy** (don't leave this implicit):
  1. Grade fails → rewrite query, retry local hybrid retrieval once.
  2. Still fails → fall back to web search (ddgs) + fetch + extract.
  3. Still fails → **hard stop**. Answer "insufficient evidence for a
     confident answer" rather than looping — cap corrective rounds at 2 and
     enforce it in code, not just as a plan bullet. An uncapped retry loop
     is a real cost and latency incident waiting to happen.

---

## 5. Security

- **Treat all fetched web content as data, never instructions.** Wrap
  scraped page text in a clearly delimited block in the prompt and instruct
  the model explicitly not to follow directives found inside it — a
  neuroscience-adjacent query is a plausible prompt-injection target via a
  malicious or compromised page in the fallback path. This is the same
  data-vs-instructions boundary that matters for any tool whose output
  becomes model input.
- **Secrets**: `ANTHROPIC_API_KEY` and friends via environment variables /
  a secrets manager, never committed. `.env` stays gitignored (check now —
  `python-dotenv` is already a dependency, confirm no `.env` is tracked).
- **Input validation** at the API boundary: query length cap, rate limiting
  per API key/IP, reject empty/binary input before it reaches the model.

---

## 6. API layer (Phase 2)

- FastAPI service wrapping the pipeline. Endpoint: `POST /query` → `{answer,
  citations, retrieval_trail, corrective_rounds_used}`. Return the
  correction trail in the response — it's what makes the UI (§10) show its
  work, and it's free once the pipeline already tracks it internally.
- Auth: simple API key header is enough at this scale.
- Async endpoints so grading/generation calls don't block the event loop.

---

## 7. Orchestration

Don't reach for LangGraph until the plain-function version (Phase 1) works
and you specifically want persistence/checkpointing/replay of runs. A
`retrieve → grade → branch → generate` loop with a capped retry count is a
short, explicit state machine — easier to unit test and step through than a
graph you can only run end-to-end. Revisit LangGraph in Phase 3+ if you want
built-in run persistence for the incident runbook.

---

## 8. Testing & CI

- Unit tests per stage (retriever, grader, corrector, generator) with
  mocked Claude calls — no live API spend in CI for unit tests.
- **Eval suite as a CI gate**: the 10-30 question neuroscience Q&A set from
  the original plan runs against a real (or cached/recorded) pipeline in CI;
  fail the build if faithfulness/groundedness or retrieval precision drops
  below a checked-in baseline. This is what turns "the corrective loop
  helps" from a one-time claim into a regression-tested property.
- Golden-file tests for prompts — catch accidental prompt drift in diffs.

---

## 9. Observability (Phase 3)

- Structured logs (JSON) with a request ID threading through
  retrieve/grade/correct/generate.
- Per-request metrics: latency per stage, corrective rounds used, web
  fallback triggered (bool), token usage, cost.
- Aggregate metrics to actually watch: retrieval-grade pass rate over time
  (drift signal — if it drops, the corpus or the query distribution
  changed), web-fallback rate (should be low; a spike means local retrieval
  is failing more), cost per query (catch a runaway loop before it's a
  bill).
- Alert thresholds on: fallback rate spike, grading pass rate drop, p95
  latency, daily cost ceiling.

---

## 10. Deployment

- `Dockerfile` for the API service; `docker-compose.yml` for
  app + chromadb locally.
- CI/CD (GitHub Actions): lint → unit tests → eval gate → build image →
  deploy. Eval gate blocks deploy on a regression, same as any other test
  gate.
- Env separation via config, not code branches: dev/staging/prod each get
  their own corpus namespace and API keys.
- Demo UI (Streamlit/Gradio) calls the deployed API over HTTP — it does not
  import the pipeline directly. That's what makes the API independently
  testable and reusable (e.g. from a CLI or a second UI) later.

---

## 11. Cost management

- Model tiering (§4) is the biggest lever — cheap model for grading/rewrite,
  capable model only for the final cited answer.
- Prompt caching on the (large, static) grading and generation system
  prompts.
- Hard per-request cost/round cap (§4) doubles as a cost control.
- Log cost per query from day one — it's much easier to have the number and
  ignore it than to retrofit tracking after a bill surprise.

---

## 12. Documentation & runbook

- README: architecture diagram, setup, how to re-run ingestion, how to add
  an eval question.
- Runbook (a short doc, not aspirational): what to do if grading pass rate
  drops, if web-fallback rate spikes, if daily cost exceeds budget. Even
  three bullets per scenario beats nothing when something actually breaks.

---

## What changed vs. the original plan

- Vector store: raw FAISS → `chromadb` (already a dependency) for the
  serving path — persistence, filtering, incremental updates.
- Grading/rewrite calls moved to a cheap model (Haiku); generation stays on
  Opus — the original plan didn't specify tiering, which is where most of
  the run-cost lives.
- Correction loop given an explicit, capped policy (max 2 rounds + hard
  stop) instead of an open-ended "rewrite and/or fallback."
- Added: API layer, security (prompt-injection handling for fetched web
  content), CI eval gate, observability/cost metrics, deployment, runbook —
  everything that separates "notebook that works on my machine" from
  "service someone else can depend on."
- UI now calls the API instead of importing the pipeline — keeps the API
  independently testable/reusable.
