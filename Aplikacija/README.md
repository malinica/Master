# EstAIte

A LangGraph real-estate search agent with LangMem-backed long-term memory, built as
the practical part of a master's thesis on agent memory systems. It compares a
**baseline** agent (no memory beyond the current conversation) against a **memory**
agent across three memory types (semantic, episodic, procedural) and three memory
extraction modes (hot-path, background, both).

## Setup

Prerequisites: Python 3.12+ (the scripts pass `loop_factory` to `asyncio.run`), Docker,
an OpenAI API key, and the town database `gradovi.db` in the project root. `gradovi.db`
is listed in `.gitignore`, so it has to be copied in separately.

1. **Environment variables.** Copy `.env.example` to `.env` in the project root and fill in
   your OpenAI key. The file contains:

   ```
   OPENAI_API_KEY=sk-...

   PG_USER=sir_agent
   PG_PASSWORD=sir_agent_pw
   PG_DB=sir_agent
   PG_HOST=localhost
   PG_PORT=5435
   PG_CONN_STRING=postgresql://sir_agent:sir_agent_pw@localhost:5435/sir_agent
   ```

   Postgres runs on host port **5435**, not the default 5432 — chosen because 5432
   and a couple of other local ports were already in use by other projects. If that
   port is free on your machine you can change it, just keep `docker-compose.yml`,
   `PG_PORT` and `PG_CONN_STRING` consistent. `PG_CONN_STRING` wins when it is set;
   otherwise the connection string is built from the `PG_*` values (defaults as above).

   **Optional: LangSmith tracing.** Add these lines to `.env`, with a free API key from
   [smith.langchain.com](https://smith.langchain.com), to get a full trace of every
   LLM/tool/graph-step call (latency, token usage, cost, the exact system prompt sent)
   for every run — no code changes needed, LangChain picks the env vars up automatically.
   Useful for debugging *why* the agent did or didn't call a memory tool on a given turn.

   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=estaite
   ```

2. **Install dependencies** (LangChain 1.x is required for `create_agent` and agent
   middleware):

   ```
   pip install -r requirements.txt
   ```

3. **Start Postgres** (uses `pgvector/pgvector:pg16`, needed for the LangMem store's
   similarity search):

   ```
   docker compose up -d
   ```

4. **Run migrations** — idempotent, safe to re-run any time. Enables `pgvector`, creates
   the custom tables (`users`, `threads`, `procedural_prompts`, `eval_runs`) plus
   LangGraph's own checkpointer and store tables (vector index on
   `openai:text-embedding-3-small`, 1536 dims), and fills in the `kind` field on store
   records written before it existed:

   ```
   python -m db.migrate
   ```

5. **Run the app**:

   ```
   python app.py
   ```

   Opens on `http://127.0.0.1:7860`. Log in with a username and password (the account
   is created on first login; passwords are stored as salted PBKDF2 hashes), pick an
   agent variant, memory types and extraction mode for a new chat (locked in once the
   chat starts), and start chatting.

6. **Run the tests** (16 tests; no network, database or API key needed — they use fake
   HTTP responses, LangGraph's `InMemoryStore` and a scripted fake chat model):

   ```
   python -m pytest
   ```

   They cover the `current_weather` tool, background profile extraction (one profile per
   user, strategy selection, `both` skipping the episodic pass), `ToolErrorMiddleware`
   on `update_user_profile`, and the strict accuracy metric.

## Project layout

```
app.py       Gradio entry point
gradovi.db   SQLite town database used by real_estate_search (not in git)
realestate/  query analysis -> SQL -> scoring -> formatting pipeline (the domain logic)
graph/       builder.py::build_graph (the single parameterized entry point used by both
             app.py and eval/runner.py), prompts.py (system prompt + memory prompt
             middleware), tools.py (real_estate_search, current_weather)
memory/      LangMem integration: config.py (enums), schemas.py (UserProfile,
             SearchEpisode), semantic.py, episodic.py, background.py,
             reflection_executors.py, procedural.py, namespaces.py (single source of
             truth for store namespacing), stamped_store.py (stamps each stored record
             with its `kind`), optimize_procedural_prompt.py (offline manual trigger,
             see below)
db/          schema.sql + repos (users, threads, procedural_prompts, eval_runs),
             connection.py, migrate.py, runtime.py (the app's one persistent event loop)
ui/          Gradio UI (layout.py, handlers.py, theme.py)
eval/        Evaluation harness (scenarios.py, runner.py, metrics.py, report.py,
             rescore_strict.py) + results/
tests/       pytest suite
```

## Agent variants

Both variants are built with LangChain's `create_agent` (a LangGraph graph) on
`gpt-4o-mini` (temperature 0.2), use a Postgres checkpointer for thread-scoped
short-term memory, and have two tools: `real_estate_search` (local SQLite town
database) and `current_weather` (live weather from Open-Meteo).

- **baseline** — no long-term memory, no cross-session recall.
- **memory** — the same agent plus a LangMem-backed Postgres store. `memory_types`
  (semantic / episodic / procedural, all three when omitted) decides which memory is
  active at all; `extraction_mode` decides how semantic and episodic memories get
  *written*.

**Reading.** A dynamic-prompt middleware (`graph/prompts.py::build_memory_prompt`) adds
the active memory types to the system prompt on every turn:

- semantic — the user's single profile document, as `CURRENT PROFILE`, with an
  instruction to build searches from it instead of asking the user again;
- episodic — the 3 past searches most relevant to the latest message, as
  `RELEVANT PAST SEARCHES`, ranked by the search strategy: `similarity` (the store's
  vector search, default), `keyword` (word overlap, no embeddings) or `hybrid`
  (0.5 · similarity + 0.5 · keyword overlap);
- procedural — the user's latest optimized system prompt from `procedural_prompts`
  replaces the base prompt.

**Writing** (`extraction_mode`):

- `hot_path` — the agent gets the `update_user_profile` (semantic) and
  `log_search_episode` (episodic) tools, with prompt instructions on when to call them.
- `background` — no memory tools; after each turn, extraction is handed to a per-user
  LangMem `ReflectionExecutor`. Semantic uses fetch-merge-write onto the single profile
  by default (`SemanticBackgroundStrategy.FETCH_MERGE`; LangMem's store manager is kept
  as `STORE_MANAGER` for comparison), episodic uses LangMem's store manager. The UI
  doesn't wait for these writes; the eval runner waits for them before the next turn.
- `both` — the hot-path tools are bound *and* the background pass runs for the same
  turn. Semantic writes combine safely (the background pass merges onto the profile by
  its known id). Episodic is an append-only log, so a turn could be recorded twice; the
  background episodic pass is skipped in this mode because its records are not
  format-compatible with the hot-path tool's.

`ToolErrorMiddleware` turns a malformed `update_user_profile` call (an update without
the profile id) into a message the agent can correct, instead of aborting the run.

Procedural memory (a versioned, optimized system prompt per user) is produced
out-of-band, not on the request path. This gathers the user's past memory-variant
conversations, runs LangMem's prompt optimizer over them and stores the next version in
`procedural_prompts`:

```
python -m memory.optimize_procedural_prompt <username>
```

The UI's "Memory types" selector (visible once "memory" is picked) is a checkbox group —
pick any combination of `semantic` / `episodic` / `procedural` — and the extraction mode
(`hot_path` / `background` / `both`) is chosen independently of that combination. This is
more permissive than the eval harness's two controlled axes below, which only exercise
specific combinations for the reported comparisons. The UI always ranks episodes with
`similarity`; the eval runner can switch the strategy.

## Evaluation harness

Two independent axes, each compared against its own baseline per scenario:

- **mechanism axis (B)**: `baseline` / `memory_hot_path` / `memory_background` /
  `memory_both`, memory types held constant (all) — does *when* memory is written matter?
- **type axis (C)**: `baseline` / `semantic_only_{hot_path,background,both}` /
  `episodic_only_{hot_path,background,both}` — full cross product of type x
  mechanism, so each type's contribution can be checked under any extraction
  mechanism. (`procedural_only` isn't part of this battery — see D5 below.)

Six scripted multi-session scenarios (`eval/scenarios.py`). Accuracy only counts the
marker groups the search tool's own output can't supply:

| Scenario | Sessions | Counted markers | Not counted (printed by the tool) | Distractor |
|---|---|---|---|---|
| `budget_and_climate_preference` | 2 | budget $200,000 | "humid" (Humidity) | — |
| `state_and_school_preference` | 2 | Texas, Arizona | "school" (Schools nearby) | — |
| `dislike_persists_across_three_sessions` | 3 | cold / winter | "snow" (Snowy days) | — |
| `episodic_recall_of_past_search` | 2 | Phoenix | — | — |
| `episodic_disambiguation_of_two_searches` | 3 | bakery | — | "daughter" |
| `distractor_relevant_vs_irrelevant` | 2 | $250,000, Oregon | — | "Vermont" |

Plus two small proof-of-concept checks that don't fit the scenario-matrix shape:
**D5 procedure-following** (seed a conversation → optimize the procedural prompt →
check a fresh procedural-only agent's behavior actually changed) and **D6 update**
(does a changed preference — budget, preferred state — overwrite the old one, not just
add to it; checked in the same session and in a new one).

```
python -m eval.runner
```

With no arguments this runs everything in one go: all scenarios, both axes, the D5 and
D6 checks, 5 repeats each (330 scenario runs plus D5/D6, executed one after another, so
expect it to take hours and to use a meaningful amount of API budget).

Writes `eval/results/<timestamp>/run_config.json`, `summary_mechanism.csv`,
`summary_type.csv`, `procedure_following.json`, `update_proof_of_concept.json` (D5/D6
only with `--axis all`), one JSON transcript per run
(`<axis>_r<repeat>__<scenario>__<variant>.json`: every message and reply, plus per-turn
logs of tool calls and arguments, tool results, retrieved episodes and the profile after
each write) and rows in the `eval_runs` table. Narrow a run down for a quick check:

```
python -m eval.runner --scenarios budget_and_climate_preference --axis type --variants baseline,semantic_only_hot_path --repeats 1
```

Other options: `--repeats N` (default 5) runs each (scenario, variant) N times with fresh
users and also writes `summary_*_aggregated.csv` (mean + stdev); `--search-strategy
similarity|keyword|hybrid` switches how episodes are ranked; `--llm-model` (default
`gpt-4o-mini`) sets the model used by the town search's query analysis.

Latency is measured until the user gets the reply. Background writes are awaited after
that, before the next message, so every later message and session works on up-to-date
memory without the write counting toward latency.

**Interrupted runs.** If the OpenAI credit runs out, the run stops, keeps every finished
run in `eval/results/<timestamp>/` and prints the command to continue. After adding credit:

```
python -m eval.runner --resume                 # the latest run, or: --resume <timestamp>
```

It reads the settings from that folder's `run_config.json`, skips every run that already
has a transcript and rebuilds the summary CSVs from all of them. It also retries runs that
were skipped earlier because of other errors. A run cut off mid-way is redone from scratch
with a fresh user; D5 and D6 are all-or-nothing, so an interrupted one restarts.

**Metrics** (`eval/metrics.py`). Markers are matched as case-insensitive substrings of the
agent's replies from the scenario's `check_from_session` on, so they measure whether the
remembered data shows up in the answer, not the overall quality of the recommendation.

- `memory_accuracy_strict` — share of the scenario's discriminative marker groups
  (`discriminative_groups`) that appear in any later reply;
- `behavioral_consistency` — share of those groups that, once recalled, stay recalled in
  every later session;
- `false_memory_rate` — share of distractor facts that leak into later replies (only in
  scenarios with distractors);
- `episodic_retrieval_accuracy` — share of later turns whose top retrieved episode has an
  expected marker in its `query` or `summary`;
- `avg_latency_seconds` and `latency_overhead_vs_baseline_seconds`;
- D5: `procedure_adherence`; D6: `memory_update_accuracy` (0.5 for adopting the new
  value + 0.5 for dropping the old one).

### Recomputing the strict metric from saved transcripts

The initial evaluation was scored with the original `memory_accuracy`, which also counted
markers the search tool prints on its own. Its transcripts keep every reply, so the strict
metric can be computed from them without calling the model:

```
python -m eval.rescore_strict 20260918T051127Z           # prints mean ± stdev per (axis, scenario, variant)
python -m eval.rescore_strict 20260918T051127Z --csv     # also writes summary_<axis>_strict.csv
```

It needs no database, API key or network, and stops if a transcript's messages differ
from `eval/scenarios.py`.

### Splitting strict accuracy into write and use

The semantic profile is one document with a known key that is always put into the
system prompt when it exists, so its retrieval is trivially correct. What can still go
wrong is whether the profile was written and whether the agent used it. The per-turn
logs (`turn_logs`) are enough to tell these apart, again without calling the model:

```
python -m eval.memory_usage 20261012T055746Z
```

Per (axis, scenario, variant) it prints `write` (share of the discriminative marker
groups in the stored profile after the first phase), `search_use` (share that made it
into a `real_estate_search` query in a later session), `strict` and how many runs scored
on `strict` without any search (the agent only repeated the stored preference). It also
counts, for `memory_hot_path` and `memory_both`, how often an episode was logged in
turns with a town search, split by whether the profile was written in the same turn.
Only the 2026-10-12 runs have `turn_logs`.

### Results used in the thesis

| Folder in `eval/results/` | What it is | In the thesis |
|---|---|---|
| `20260918T051127Z` | Initial evaluation of the prototype: 4 scenarios, mechanism axis without `both`, type axis without `*_both`, 5 repeats, D5, D6. Scored with the original `memory_accuracy`; strict values come from `eval.rescore_strict`. | Tables 9.3, 9.4, Figure 9.1 |
| `20261011T000048Z`, `20261011T001617Z` | Episode search strategy comparison: `keyword` and `hybrid`, type axis, budget and episodic-recall scenarios, 5 repeats (`similarity` comes from the final run). | Search strategy comparison |
| `20261012T002249Z` | First full evaluation of the improved system, with the original hot-path prompt: 6 scenarios, both axes, 5 repeats, D5, D6. | Table 9.8 (before the prompt change) |
| `20261012T055746Z` | Final evaluation, after the prompt change in `graph/prompts.py`; same setup. | Tables 9.5, 9.6, 9.8 (after), Figure 9.2, chapters 9.7 and 9.8 |

The older folders were written by earlier versions of the runner, so their transcripts
differ slightly: `20260918T051127Z` has only `memory_accuracy` and no `turn_logs`, and the
`20261011` runs also record `memory_accuracy` and `background_mechanism`. None of them has
a `run_config.json`, so they can't be resumed.
