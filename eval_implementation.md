RAG Evaluation Pipeline — Architecture Spec & Rewrite Plan

Runtime cleanup note: evaluation deployments are now registered by default and
optional eval dependencies are checked only when an eval flow executes.
`EVAL_ENABLED` and `EVAL_POST_INGESTION_HOOK` have been replaced by runtime
gating and `EVAL_STATIC_POST_INGESTION_HOOK`.

 Target repo: /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking
 Audience: downstream execution agent
 Constraints honored:
 - Phoenix Experiments SDK is source of truth (input mode + storage).
 - RAGAS + phoenix.evals layered.
 - All automated triggers feature-flagged, default OFF (open-source Day-1 contract).
 - No new Postgres tables. CSV permitted only as optional Prefect artifact.

 ---
 1. Context

 Why this rewrite

 Current evaluation/ package is a research-grade script, not a production loop. Concrete failure modes:

 - evaluation/evaluate.py runs RAGPipeline.run in a synchronous for loop — no concurrency, no per-case error isolation, no linkage to Phoenix traces. A single bad case kills the run; successful traces emitted by
 LangChainInstrumentor cannot be correlated to a specific evaluation case or experiment.
 - Judge model hardcoded to amazon.nova-pro-v1:0; embeddings hardcoded to HuggingFace; outputs dumped to CSV at data/synthetic/eval_results_detailed.csv. No run identity, no comparison surface, no UI.
 - Module-level imports of ragas, langchain_huggingface, pandas mean any import evaluation.evaluate from a Prefect registrar forces heavy optional deps. Blocks open-source Day-1: a fresh clone running make ingest must not require
  RAGAS.
 - question_generation.py reads a pickle of the document store and dumps JSON to SYNTHETIC_QA_PERSIST_PATH. Not orchestrated, not idempotent, not versioned.
 - No automation. No scheduled or post-ingestion eval. No threshold gates.

 Outcome

 - Golden dataset lives in Phoenix Datasets, uploaded from evaluation/datasets/ground_truth_dataset.csv, content-hashed for idempotency.
 - Every eval run is a Phoenix Experiment: per-example RAG span tree (auto-captured by existing LangChainInstrumentor) + per-evaluator scores from RAGAS + phoenix.evals.
 - Evaluations execute inside Prefect flows (eval_golden_flow, eval_trace_flow), feature-flagged off by default. Fresh clone with make ingest never triggers eval code and never imports RAGAS / Phoenix Experiments SDK at module
 top level.
 - Prefect emits a markdown artifact per run with the Phoenix Experiment URL, aggregate metric table, pass/fail vs configurable thresholds. CSV stays available as an opt-in Prefect artifact.
 - No new Postgres tables. No filesystem dumps as primary storage.

 ---
 2. Architectural Flow

 2.1 ASCII overview

                        ┌──────────────────────────────────────────┐
                        │  evaluation/datasets/ground_truth_dataset.csv         │
                        │  (golden, hand-curated, OSS-bundled)     │
                        └────────────────┬─────────────────────────┘
                                         │ (1) one-time + on hash change
                                         ▼
                        ┌──────────────────────────────────────────┐
                        │  loaders.ensure_dataset_uploaded()       │
                        │  → phoenix.Client().upload_dataset(...)  │
                        │  registers Phoenix Dataset by name+hash  │
                        └────────────────┬─────────────────────────┘
                                         │
                                         ▼
                        ┌──────────────────────────────────────────┐
                        │  Phoenix Dataset: "rag-golden-v1"        │
                        │  (UI-visible, versioned by Phoenix)      │
                        └────────────────┬─────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  Prefect: eval_golden_flow(dataset_name, experiment_name)                │
    │   - guard: if not EvalConfig.enabled → log + return                      │
    │   - init_observability("advanced-rag-eval")                              │
    │   - construct RAGPipeline ONCE                                           │
    │   - resolve Phoenix Dataset by name                                      │
    │   - phoenix.experiments.run_experiment(                                  │
    │        dataset=ds,                                                       │
    │        task=run_rag_for_example,    # evaluation/pipeline/task.py        │
    │        evaluators=[ragas_*, phoenix_judges_*],                           │
    │        experiment_name=...,                                              │
    │        concurrency=LLM_MAX_PARALLEL_REQUEST,                             │
    │        project_name=EVAL_PHOENIX_PROJECT)                                │
    │   - apply thresholds → pass/fail flag                                    │
    │   - emit Prefect markdown artifact w/ Phoenix URL                        │
    └─────────────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────────────────┐
                        │  Phoenix Experiments UI                  │
                        │   - per-example traces                   │
                        │   - per-example evaluator scores         │
                        │   - aggregate metrics, comparisons       │
                        └──────────────────────────────────────────┘

 2.2 Two evaluation modes (both default-OFF)

 A) Scheduled offline eval (golden dataset) — eval_golden_flow
 - Inputs: a Phoenix Dataset (registered from ground_truth_dataset.csv).
- Triggers: EVAL_STATIC_SCHEDULE_CRON (Prefect deployment schedule) and/or EVAL_STATIC_POST_INGESTION_HOOK (called from end-of-ingestion if enabled).
 - Evaluators: RAGAS Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision, AnswerCorrectness (ground-truth-bearing) + phoenix.evals reference-free judges (Hallucination, Relevance, QA-Correctness).

 B) Online trace-based eval — eval_trace_flow
 - Inputs: time window over Phoenix spans in production project (e.g. last hour of advanced-rag).
- Triggers: EVAL_DRIFT_SCHEDULE_CRON.
 - Evaluators: phoenix.evals reference-free only (no ground truth). Writes annotations back onto live spans via Phoenix annotation API — not a separate Experiment.

 Both flows share evaluator code paths and the same EvalConfig flag surface.

 2.3 Trace correlation

 Inside task=run_rag_for_example, the existing LangChainInstrumentor auto-emits spans for the entire RAG call tree (retriever, LLM, reranker if instrumented). phoenix.experiments.run_experiment opens an Experiment-scoped trace
 context per example, so spans land with the experiment ID attached. No manual span tagging is required.

 The eval flow MUST call init_observability("advanced-rag-eval") (or whatever EVAL_PHOENIX_PROJECT resolves to) before invoking run_experiment so the spans go to the eval project, not the production project.

 ---
 3. Component Breakdown

 3.1 Proposed evaluation/ tree

 evaluation/
 ├── __init__.py                          # empty
 ├── config.py                            # EvalConfig(BaseSettings); EVAL_* env; feature flags
 ├── phoenix_bootstrap.py                 # lazy phoenix.Client() factory
 ├── datasets/
 │   ├── __init__.py                      # empty
 │   ├── ground_truth_dataset.csv                      # KEEP (golden, hand-curated)
 │   └── loaders.py                       # CSV→Phoenix Dataset upload; idempotent by name+sha256
 ├── generation/
 │   ├── __init__.py                      # empty
 │   └── synthetic.py                     # async QA generation; uploads to Phoenix Dataset (not JSON)
 ├── pipeline/
 │   ├── __init__.py                      # empty
 │   ├── task.py                          # run_rag_for_example(example) — Phoenix task fn
 │   └── adapter.py                       # RAG output dict → RAGAS sample shape
 ├── evaluators/
 │   ├── __init__.py                      # empty
 │   ├── ragas_metrics.py                 # RAGAS metric→Phoenix evaluator callables (factory)
 │   ├── phoenix_judges.py                # phoenix.evals reference-free evaluators
 │   └── thresholds.py                    # pass/fail policy (metric → min score)
 ├── reporting/
 │   ├── __init__.py                      # empty
 │   └── artifacts.py                     # Prefect markdown artifact w/ Phoenix UI URL
 └── orchestration/
     ├── __init__.py                      # empty
     ├── flows.py                         # @flow eval_golden_flow, eval_trace_flow, synthetic_qa_flow
     ├── tasks.py                         # @task wrappers (one-task-per-stage)
     └── deployments.py                   # eval deployment specs, returned only if EVAL_ENABLED

 3.2 Per-file responsibilities & import discipline

 All module-level imports below must be standard library + Pydantic + Prefect only unless explicitly noted as "heavy". Heavy deps (ragas, phoenix, phoenix.experiments, phoenix.evals, langchain_huggingface, pandas) must be
 imported lazily inside functions so import evaluation.<anything> is safe on a fresh clone without the [eval] extra installed.

 evaluation/config.py

 - Purpose: EvalConfig(BaseSettings) — feature flags + dataset/project/judge config.
 - Public: EvalConfig, get_eval_config() (@lru_cache).
 - Top-level imports: pydantic, pydantic_settings, functools.
 - Forbidden top-level: phoenix, ragas, rag_core.
 - Fields (env prefix EVAL_):
   - enabled: bool = False (master kill switch)
   - schedule_cron: str | None = None
   - trace_schedule_cron: str | None = None
   - post_ingestion_hook: bool = False
   - dataset_name: str = "rag-golden-v1"
   - dataset_csv_path: str = "evaluation/datasets/ground_truth_dataset.csv"
   - phoenix_project: str = "advanced-rag-eval"
   - phoenix_endpoint: str | None = None
   - judge_provider: str = "aws"
   - judge_model: str | None = None
   - concurrency: int = 4
   - trace_lookback_minutes: int = 60
   - csv_artifact_enabled: bool = False
   - thresholds: dict[str, float] = {}
   - ragas_metrics: list[str] = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
   - phoenix_evaluators: list[str] = ["hallucination", "qa_correctness", "relevance"]
   - max_examples: int | None = None (hard cap to bound judge cost)

 Why evaluation/config.py instead of extending orchestration/config.py: PrefectConfig covers Prefect runtime (API URL, work pool). Eval flags are a different domain (judge model, evaluator names, dataset path) and not
 Prefect-runtime concerns. Co-locating with consumer keeps orchestration/config.py minimal and lets OSS users disable eval entirely by skipping the [eval] extra. orchestration/initialize.py reads EvalConfig only.

 evaluation/phoenix_bootstrap.py

 - Purpose: lazy Phoenix client factory; resolves endpoint from EvalConfig.phoenix_endpoint → PHOENIX_COLLECTOR_ENDPOINT env → derived from OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (strip /v1/traces).
 - Public: get_phoenix_client() -> "phoenix.Client", phoenix_experiment_url(experiment_id) -> str.
 - Top-level: os, logging.
 - Lazy: phoenix as px.

 evaluation/datasets/loaders.py

 - Purpose: read ground_truth_dataset.csv, compute sha256 over normalized rows, upload to Phoenix as Dataset named EvalConfig.dataset_name only if hash differs from latest registered version's metadata.
 - Public: ensure_dataset_uploaded(eval_config: EvalConfig) -> str (returns dataset handle/id), load_csv_rows(path) -> list[dict], compute_dataset_hash(rows) -> str.
 - Top-level: csv, hashlib, logging, pathlib.
 - Lazy: phoenix, pandas (only when csv_artifact_enabled).
 - Idempotency: do not re-upload if Phoenix already has a dataset with same name AND a content_sha256 metadata key equal to current hash. If hash differs, upload a new version (Phoenix versions datasets natively).

 evaluation/generation/synthetic.py

 - Purpose: rewrite of question_generation.py. Preserves async fanout + semaphore. No JSON output. Uploaded directly as Phoenix Dataset (e.g. rag-synthetic-v{timestamp}).
 - Public: async generate_synthetic_dataset(config: dict, eval_config: EvalConfig) -> str.
 - Top-level: asyncio, json, logging, time.
 - Lazy: phoenix, rag_core.agents, rag_core.prompts, rag_core.schemas.
 - Triggered by optional synthetic_qa_flow (registered only if EVAL_ENABLED).

 evaluation/pipeline/task.py

 - Purpose: function Phoenix Experiments SDK calls once per example.
 - Pseudo-signature: run_rag_for_example(example: dict, *, rag_pipeline: RAGPipeline) -> dict.
 - Returns: {"response": str, "retrieved_contexts": list[str]}.
 - Top-level: logging, re.
 - No heavy imports (RAGPipeline passed in via closure — instantiated once per flow run, not per example).
 - Behavior: calls rag_pipeline.run(example["question"]), strips [[file_ref]] markers via adapter, unpacks contexts: list[Document] → list[str].
 - Per-example error isolation: catches Exception, logs, returns {"response": "", "retrieved_contexts": [], "error": str(exc)} so Phoenix records the failure instead of aborting the whole experiment.
 - Phoenix SDK receives partial(run_rag_for_example, rag_pipeline=pipeline) from the flow body.

 evaluation/pipeline/adapter.py

 - Purpose: pure conversion fns between Phoenix example dict, RAG output dict, RAGAS sample shape.
 - Public: to_ragas_sample(example, output) -> dict, strip_file_refs(text: str) -> str (preserve legacy regex r"\s*\[\[.*?\]\]").
 - Top-level: re.
 - No heavy deps.

 evaluation/evaluators/ragas_metrics.py

 - Purpose: factory returning list of Phoenix-compatible evaluator callables, one per RAGAS metric in eval_config.ragas_metrics.
 - Public: build_ragas_evaluators(config: dict, eval_config: EvalConfig) -> list[Callable].
 - Each evaluator: Phoenix signature (input, output, expected, metadata) -> float | dict. Inside, builds 1-row EvaluationDataset, runs the single metric, returns scalar.
 - Top-level: logging.
 - Lazy: ragas, ragas.metrics, ragas.llms.LangchainLLMWrapper, ragas.embeddings.LangchainEmbeddingsWrapper, langchain_huggingface.HuggingFaceEmbeddings, rag_core.agents.AgentFactory.
 - Singletons (built once per flow run, not per call): judge LLM wrapper, embeddings.
 - Metric map: "faithfulness"→Faithfulness, "answer_relevancy"→AnswerRelevancy, "context_recall"→ContextRecall, "context_precision"→ContextPrecision, "answer_correctness"→AnswerCorrectness(weights=[0.8, 0.2]).

 evaluation/evaluators/phoenix_judges.py

 - Purpose: reference-free LLM-as-judge evaluators via phoenix.evals (Hallucination, Relevance, QACorrectness). Usable on online traces (no ground truth).
 - Public: build_phoenix_evaluators(config: dict, eval_config: EvalConfig) -> list.
 - Top-level: logging.
 - Lazy: phoenix.evals, rag_core.agents.
 - Re-uses same judge LLM singleton as RAGAS.

 evaluation/evaluators/thresholds.py

 - Purpose: apply eval_config.thresholds to aggregate metrics; return PassFailReport.
 - Public: @dataclass PassFailReport(failed: bool, breaches: list[tuple[str, float, float]]), evaluate_thresholds(aggregates, eval_config) -> PassFailReport.
 - Top-level: dataclasses, logging.
 - Empty thresholds dict ⇒ never fails (warn-only mode).

 evaluation/reporting/artifacts.py

 - Purpose: emit Prefect markdown artifact per experiment run.
 - Public: emit_experiment_artifact(experiment, pass_fail, eval_config), optional emit_csv_artifact(experiment, eval_config).
 - Top-level: logging.
 - Lazy: prefect.artifacts, pandas (CSV only).
 - Markdown sections: Phoenix UI link, aggregate metric table, threshold pass/fail column, first 5 failed examples.

 orchestration/evaluation/flows.py

 - Purpose: Prefect @flow definitions.
 - Public: eval_golden_flow, eval_trace_flow, synthetic_qa_flow.
 - Top-level: logging, prefect, prefect.exceptions, evaluation.config.
 - Lazy (inside flow bodies, after the enabled guard): RAGPipeline, evaluator factories, phoenix bootstrap, everything heavy.
 - Pattern mirrors orchestration/ingestion/flows.py: _get_logger() helper with MissingContextError fallback.

 orchestration/evaluation/tasks.py

 - Purpose: thin @task wrappers — one per stage. Each starts with setup_logging() per ingestion convention.
 - Public: resolve_dataset_task, run_experiment_task, apply_thresholds_task, emit_artifact_task.
 - Decorator pattern: @task(name="eval-<stage>", retries=1, retry_delay_seconds=5). run_experiment_task uses retries=0 (re-runs would create duplicate Experiments).

 orchestration/evaluation/deployments.py

 - Purpose: declarative deployment specs for the eval flows. Consumed by orchestration/initialize.py.
 - Public: build_eval_deployment_specs(eval_config: EvalConfig) -> list[tuple]. Returns [] if not eval_config.enabled — this is the kill switch.
 - Top-level: evaluation.config + flow objects from orchestration.evaluation.flows. Both safe without [eval] extra.

 3.3 Disposition of existing files

 ┌───────────────────────────────────────────┬────────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
 │                   File                    │                               Action                               │                                        Reason                                        │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ evaluation/evaluate.py                    │ DELETE                                                             │ Replaced by pipeline/task.py + evaluators/ragas_metrics.py + orchestration/flows.py. │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ evaluation/question_generation.py         │ DELETE                                                             │ Replaced by generation/synthetic.py (uploads to Phoenix Dataset, not pickle/JSON).   │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ evaluation/__init__.py                    │ KEEP (empty).                                                      │                                                                                      │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ evaluation/datasets/ground_truth_dataset.csv           │ KEEP. Source of truth for bundled golden set.                      │                                                                                      │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ data/synthetic/eval_question_answers.json │ DEPRECATE (don't delete in OSS — users may have it). Stop reading. │                                                                                      │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ data/synthetic/eval_results_detailed.csv  │ DEPRECATE. No longer primary output.                               │                                                                                      │
 ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
 │ SYNTHETIC_QA_PERSIST_PATH in .env.example │ REMOVE (or mark deprecated). Replaced by EVAL_STATIC_DATASET_NAME.   │                                                                                      │
 └───────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

 ---
 4. Orchestration Strategy

 4.1 Flow/task decomposition

 eval_golden_flow(dataset_name: str | None, experiment_name: str | None, evaluator_names: list[str] | None):
 1. Guard (before any heavy import): if not eval_config.enabled: logger.info("EVAL_ENABLED=false, skipping"); return None.
 2. init_observability(eval_config.phoenix_project) — first heavy call. Idempotent (existing _instrumented flag in rag_core/core/telemetry.py).
 3. Lazy-import: RAGPipeline, evaluator factories, phoenix bootstrap, phoenix.experiments.
 4. Construct RAGPipeline(load_config(), provider=..., prompt_name=...) once. Bind via functools.partial(run_rag_for_example, rag_pipeline=pipeline).
 5. Build evaluators once (judge LLM, embeddings — singletons).
 6. resolve_dataset_task(eval_config) → dataset handle.
 7. run_experiment_task(dataset, task_fn, evaluators, experiment_name, concurrency, project_name) → wraps single call to phoenix.experiments.run_experiment(...).
 8. apply_thresholds_task(experiment, eval_config) → PassFailReport.
 9. emit_artifact_task(experiment, pass_fail, eval_config) → markdown artifact.
 10. If pass_fail.failed and eval_config.thresholds: raise RuntimeError to mark Prefect flow Failed.

 eval_trace_flow(time_window_minutes, evaluator_names):
 - Same guard + init pattern.
 - Pulls recent spans via Phoenix API, runs reference-free evaluators, annotates spans.

 synthetic_qa_flow(corpus_source: str = "document_store"):
 - Same guard + init pattern.
 - Calls asyncio.run(generate_synthetic_dataset(...)).

 4.2 Memory & concurrency

 - One pipeline per flow run. RAGPipeline holds LLM clients, retriever, prompt template — instantiation is expensive. Never inside per-example task. Same applies to RAGAS judge wrapper + embeddings.
 - Concurrency delegated to Phoenix. phoenix.experiments.run_experiment(concurrency=eval_config.concurrency) handles parallel execution across examples. Do NOT parallelize at Prefect task level — keep one Prefect task = one
 run_experiment call. Avoids double-counted concurrency budgets and keeps retry semantics simple.
 - Default concurrency = 4 (falls back to LLM_MAX_PARALLEL_REQUEST=3 style).

 4.3 Logging

 First line of every @task body: setup_logging() (from rag_core.core.config) — same convention as orchestration/ingestion/tasks.py. Logger acquired via _get_logger() helper with MissingContextError fallback to stdlib. Update
 .env.example so PREFECT_LOGGING_EXTRA_LOGGERS=rag_core,evaluation (extend to capture evaluation.* logs in the Prefect UI per recent commit d84f156).

 4.4 Feature-flag enforcement at registration time

 orchestration/initialize.py currently has a module-level _DEPLOYMENT_SPECS list. The eval rewrite extends register_deployments() (intent — execution agent implements):

 - Import at top of orchestration/initialize.py: from orchestration.evaluation.deployments import build_eval_deployment_specs and from evaluation.config import get_eval_config. Safe to import without [eval] extra installed.
 - Inside register_deployments(), after the existing loop:
   - eval_specs = build_eval_deployment_specs(get_eval_config())
   - If empty (because EVAL_ENABLED=false): log Eval deployments disabled (EVAL_ENABLED=false) — skipping and return.
   - If non-empty: iterate and register using the same client.create_flow_from_name + client.create_deployment pattern. Entrypoint format: orchestration/evaluation/flows.py:eval_golden_flow (similar for others).
 - Schedule: if eval_config.schedule_cron set, attach CronSchedule via client.create_deployment(schedules=[...]).

 Result: fresh clone with EVAL_ENABLED unset registers exactly the existing two ingestion deployments — Day-1 UX preserved.

 4.5 Phoenix project routing

 - Default eval project: advanced-rag-eval.
 - Override: EVAL_PHOENIX_PROJECT.
 - Set as service.name resource attribute when constructing the Phoenix client AND pass as project_name= kwarg to phoenix.experiments.run_experiment(...). Eval spans don't pollute production advanced-rag project.
 - OTLP exporter endpoint stays the same. Projects are a Phoenix-side organizational construct.

 4.6 Optional post-ingestion hook

 If EVAL_POST_INGESTION_HOOK=true AND EVAL_ENABLED=true, the existing end_to_end_flow in orchestration/ingestion/flows.py calls eval_golden_flow() as a final non-blocking step. Implementation:
 - Lazy-import eval_golden_flow inside end_to_end_flow body to keep ingestion-only deployments lean.
 - Wrap in try/except so eval failures never break ingestion.
 - Plain function call (not Prefect subflow) — already inside a flow context.

 ---
 5. Strict Execution Plan

 Execute in order. Each step: exact path, function, decorator, deletions.

 1. Add evaluation/config.py. Define EvalConfig(BaseSettings) with field-level validation_alias="EVAL_*" (mirror orchestration/config.py). Define get_eval_config() with @lru_cache. No heavy imports.
 2. Add evaluation/phoenix_bootstrap.py. Define get_phoenix_client() (lazy import phoenix as px; resolves endpoint from EvalConfig.phoenix_endpoint → PHOENIX_COLLECTOR_ENDPOINT env → derived from
 OTEL_EXPORTER_OTLP_TRACES_ENDPOINT by stripping /v1/traces). Define phoenix_experiment_url(experiment_id).
 3. Add evaluation/datasets/loaders.py. Define load_csv_rows(path) (stdlib csv), compute_dataset_hash(rows) (sha256 over JSON-sorted rows), ensure_dataset_uploaded(eval_config) returning dataset handle. Lazy import phoenix as px
 inside function. Idempotent on (name, content_sha256).
 4. Add evaluation/pipeline/adapter.py. Define strip_file_refs(text) (preserve legacy regex r"\s*\[\[.*?\]\]"), to_ragas_sample(example, output).
 5. Add evaluation/pipeline/task.py. Define run_rag_for_example(example, *, rag_pipeline) returning {"response": str, "retrieved_contexts": list[str]}. Strip refs via adapter. Per-example exception → {"response": "",
 "retrieved_contexts": [], "error": ...}. No heavy top-level imports.
 6. Add evaluation/evaluators/ragas_metrics.py. Define build_ragas_evaluators(config, eval_config):
   - Lazy imports: ragas, ragas.metrics, ragas.llms.LangchainLLMWrapper, ragas.embeddings.LangchainEmbeddingsWrapper, langchain_huggingface.HuggingFaceEmbeddings, rag_core.agents.AgentFactory.
   - Build judge LLM ONCE: AgentFactory.create(eval_config.judge_provider, temperature=0, model_name=eval_config.judge_model, config=config).
   - For each name in eval_config.ragas_metrics, instantiate metric, wrap as Phoenix callable (input, output, expected, metadata) -> float, return list.
   - Metric map: see §3.2.
   - Inside evaluator closure: build 1-row EvaluationDataset.from_list([...]), call evaluate(dataset=..., metrics=[metric], llm=ragas_llm, embeddings=ragas_emb, run_config=RunConfig(timeout=60, max_retries=3, max_workers=1)),
 return scalar.
 7. Add evaluation/evaluators/phoenix_judges.py. Define build_phoenix_evaluators(config, eval_config):
   - Lazy import phoenix.evals + rag_core.agents.
   - Map: "hallucination"→HallucinationEvaluator, "qa_correctness"→QAEvaluator, "relevance"→RelevanceEvaluator.
   - Re-use same judge LLM singleton as RAGAS.
 8. Add evaluation/evaluators/thresholds.py. @dataclass PassFailReport(failed: bool, breaches: list[tuple[str, float, float]]). evaluate_thresholds(aggregates, eval_config) -> PassFailReport. Empty thresholds dict ⇒ never fails.
 9. Add evaluation/reporting/artifacts.py. emit_experiment_artifact(experiment, pass_fail, eval_config) via prefect.artifacts.create_markdown_artifact(...). Markdown contents: Phoenix UI link (via phoenix_experiment_url),
 aggregate metric table, threshold pass/fail column, first 5 failed examples. Optional emit_csv_artifact(experiment, eval_config) only if csv_artifact_enabled (prefer prefect.artifacts.create_table_artifact or link artifact
 pointing to Phoenix export URL — no filesystem dump).
 10. Add orchestration/evaluation/tasks.py. Define:
   - @task(name="eval-resolve-dataset", retries=1, retry_delay_seconds=5) resolve_dataset_task(eval_config) → calls ensure_dataset_uploaded.
   - @task(name="eval-run-experiment", retries=0) run_experiment_task(dataset, task_fn, evaluators, experiment_name, concurrency, project_name) → single call to phoenix.experiments.run_experiment. retries=0 to avoid duplicate
 Experiments.
   - @task(name="eval-apply-thresholds") apply_thresholds_task(experiment, eval_config) -> PassFailReport.
   - @task(name="eval-emit-artifact") emit_artifact_task(experiment, pass_fail, eval_config).
   - Each task starts: setup_logging() + _get_logger() with MissingContextError fallback.
 11. Add orchestration/evaluation/flows.py. Define:
   - @flow(name="rag-eval-golden-flow", log_prints=True) def eval_golden_flow(dataset_name: str | None = None, experiment_name: str | None = None, evaluator_names: list[str] | None = None) -> dict | None:
       - Top: eval_config = get_eval_config(); if not eval_config.enabled: _get_logger().info("EVAL_ENABLED=false, skipping"); return None.
     - Then init_observability(eval_config.phoenix_project).
     - Lazy import inside body: from rag_core.pipelines.pipeline import RAGPipeline, evaluator factories, from functools import partial.
     - Build pipeline once. Build evaluators once.
     - Call sequence: resolve_dataset_task → run_experiment_task → apply_thresholds_task → emit_artifact_task.
     - If pass_fail.failed: raise RuntimeError(...).
   - @flow(name="rag-eval-trace-flow", log_prints=True) def eval_trace_flow(...).
   - @flow(name="rag-synthetic-qa-flow", log_prints=True) def synthetic_qa_flow(...).
 12. Add orchestration/evaluation/deployments.py. Define build_eval_deployment_specs(eval_config) -> list[tuple] returning specs analogous to _DEPLOYMENT_SPECS in orchestration/initialize.py. Returns [] when not
 eval_config.enabled. Schedule attached via CronSchedule if eval_config.schedule_cron.
 13. Modify orchestration/initialize.py. Inside register_deployments(), after existing loop, call build_eval_deployment_specs(get_eval_config()). If non-empty, iterate and register using same client.create_flow_from_name +
 client.create_deployment pattern. Entrypoint format: orchestration/evaluation/flows.py:<func_name>. If empty, single info log line. Do NOT add eval specs to module-level _DEPLOYMENT_SPECS literal — they are dynamic.
 14. Modify orchestration/ingestion/flows.py (only if EVAL_POST_INGESTION_HOOK honored). End of end_to_end_flow, after end_to_end_ingestion_task: lazy-import eval_golden_flow, call inside try/except if
 get_eval_config().post_ingestion_hook. Failures log, never raise.
 15. Delete legacy files.
   - rm evaluation/evaluate.py
   - rm evaluation/question_generation.py
   - Clean any stale re-exports from evaluation/__init__.py.
 16. Update .env.example. Append:
 # ─── Evaluation (default OFF; enable to run RAG eval) ──────────────────
 EVAL_ENABLED=false
 EVAL_STATIC_SCHEDULE_CRON=
 EVAL_DRIFT_SCHEDULE_CRON=
 EVAL_STATIC_POST_INGESTION_HOOK=false
 EVAL_STATIC_DATASET_NAME=ground-truth-v1
 EVAL_STATIC_DATASET_CSV_PATH=evaluation/datasets/ground_truth_dataset.csv
 EVAL_PHOENIX_PROJECT=advanced-rag-eval
 EVAL_PHOENIX_ENDPOINT=
 EVAL_JUDGE_PROVIDER=aws
 EVAL_JUDGE_MODEL=
 EVAL_CONCURRENCY=4
 EVAL_DRIFT_LOOKBACK_MINUTES=60
 EVAL_STATIC_CSV_ARTIFACT_ENABLED=false
 EVAL_STATIC_MAX_EXAMPLES=
 EVAL_RAGAS_METRICS=faithfulness,answer_relevancy,context_recall,context_precision,answer_correctness
 EVAL_PHOENIX_EVALUATORS=hallucination,qa_correctness,relevance
 EVAL_THRESHOLDS=
 16. Remove (or mark deprecated) SYNTHETIC_QA_PERSIST_PATH. Expand PREFECT_LOGGING_EXTRA_LOGGERS=rag_core,evaluation.
 17. Update pyproject.toml. Add optional-dependencies group:
 [project.optional-dependencies]
 eval = [
   "ragas>=0.4.3",
   "arize-phoenix>=4.0,<5.0",
   "arize-phoenix-evals>=0.14",
   "langchain-huggingface>=1.2.1",
   "pandas>=2.3.3",
 ]
 17. Move ragas>=0.4.3 out of [dependency-groups] dev (or keep in dev for tests, but eval is canonical runtime install). Confirm openinference-instrumentation-langchain already in base deps. Pin arize-phoenix>=4.0,<5.0 against
 run_experiment API drift.
 18. Add CI smoke test (no Phoenix required). Create tests/test_evaluation_imports.py:
   - Import evaluation.config, orchestration.evaluation.flows, orchestration.evaluation.deployments and assert success without ragas / phoenix installed.
   - Call eval_golden_flow() with monkeypatched EVAL_ENABLED=false → assert returns None, no raise (Day-1 contract).
   - Call build_eval_deployment_specs(EvalConfig(enabled=False)) → assert [].
 19. Update Makefile. Add:
 eval:
     docker compose ... ingestion-worker uv run --no-sync python -c "from orchestration.evaluation.flows import eval_golden_flow; eval_golden_flow()"
 register-eval-deployment:
     # requires EVAL_ENABLED=true
 20. Verification commands — see §6.

 ---
 6. Verification

 6.1 Day-1 open-source contract (no Phoenix configured, default .env.example)

 git clone <repo> && cd <repo>
 cp .env.example .env
 make up
 make ingest
 Expected: ingestion completes. No eval deployments registered. orchestration/initialize.py logs Eval deployments disabled (EVAL_ENABLED=false) — skipping. No ragas or phoenix-experiments import errors. CI smoke test passes.

 6.2 Local dry-run with eval flag flipped but Phoenix unreachable

 Set EVAL_ENABLED=true, stop Phoenix container, trigger eval_golden_flow from Prefect UI. Expected: flow fails fast on resolve_dataset_task with Could not connect to Phoenix at <endpoint>. Flow run marked Failed in Prefect UI —
 not silent.

 6.3 Full happy-path with Phoenix running

 # 1. Upload golden dataset (idempotent)
 docker compose run --rm ingestion-worker uv run --no-sync python -c \
   "from evaluation.datasets.loaders import ensure_dataset_uploaded; \
    from evaluation.config import get_eval_config; \
    print(ensure_dataset_uploaded(get_eval_config()))"

 # 2. Trigger an experiment (one-shot, not via deployment)
 docker compose run --rm ingestion-worker uv run --no-sync python -c \
   "from orchestration.evaluation.flows import eval_golden_flow; \
    eval_golden_flow(experiment_name='manual-smoke-1')"

 # 3. Verify Phoenix UI (http://localhost:6006):
 #    - Project "advanced-rag-eval" exists
 #    - Dataset "rag-golden-v1" appears
 #    - Experiment "manual-smoke-1" exists with row count == golden set row count
 #    - Each row has columns: faithfulness, answer_relevancy, context_recall,
 #      context_precision, answer_correctness, hallucination, qa_correctness, relevance
 #    - Clicking an example shows full RAG span tree (retrieval, LLM, etc.)

 # 4. Verify Prefect artifact: open flow run, check Artifacts tab for markdown
 #    artifact with Phoenix experiment URL and aggregate metric table.

 6.4 Regression: legacy CSV path is gone

 data/synthetic/eval_results_detailed.csv should NOT be created by a successful eval run. evaluation/evaluate.py and evaluation/question_generation.py absent. tests/test_evaluation_imports.py passes.

 6.5 Threshold gate works

 Set EVAL_THRESHOLDS={"faithfulness": 0.9, "context_recall": 0.85} (JSON in env). Re-run flow. If aggregate faithfulness < 0.9, flow fails in Prefect UI with RuntimeError("Threshold breached: faithfulness=0.83 < 0.9") and the
 markdown artifact lists the breach.

 ---
 7. Risks & Open Questions

 1. Judge cost explosion on large golden sets. RAGAS metrics each invoke judge ≥1×/example; Faithfulness/AnswerCorrectness decompose into multiple sub-prompts. 100-row golden set × 5 RAGAS + 3 phoenix.evals ≈ 1,500+ judge
calls/run. Mitigation: default golden set is 9 rows; honor EVAL_STATIC_MAX_EXAMPLES hard cap; document per-run cost.
 2. Phoenix ingest volume from frequent runs. Nightly cron × 100 examples × 30 spans = 3,000 spans/day in eval project. Mitigation: keep eval project separate from prod (advanced-rag-eval vs advanced-rag); document retention
 expectations; recommend project-level cleanup cron for OSS long-term users.
 3. Feature-flag-off invisibility. With EVAL_ENABLED=false default, easy to ship broken eval that no one notices. Mitigation: CI smoke test (step 18) imports all eval modules and runs disabled-by-default flow path. Follow-up (out
  of scope): second CI job runs full eval flow against local Phoenix container on PRs.
 4. RAGAS↔Phoenix evaluator signature mismatch. RAGAS expects whole EvaluationDataset; Phoenix calls evaluators per-row. Wrapping per-row (build 1-row dataset, run metric) works but adds serialization overhead and may behave
 differently for dataset-normalized metrics (none of the five listed normalize across rows, but AnswerRelevancy is sensitive). Mitigation: spot-check parity by running same 9-row set through batch-mode RAGAS once and confirm
 aggregates match within 5%. Pin RAGAS version.
 5. Phoenix Experiments SDK version drift. phoenix.experiments.run_experiment API has changed across 4.x minor versions. Mitigation: pin arize-phoenix>=4.0,<5.0. Document tested version in README.
 6. PREFECT_LOGGING_EXTRA_LOGGERS missing evaluation. Eval flow logs may not appear in Prefect UI. Mitigation: update .env.example to PREFECT_LOGGING_EXTRA_LOGGERS=rag_core,evaluation.
 7. Open: judge provider for OSS users without AWS. EVAL_JUDGE_PROVIDER=aws default won't work for Ollama/LM Studio users. Recommend: fall back to GENERATION_MODEL_PROVIDER from existing config when EVAL_JUDGE_PROVIDER empty.
 Document in .env.example.
 8. Open: post-ingestion hook scope. If EVAL_POST_INGESTION_HOOK=true, should it block ingestion success on threshold failure, or log-only? Recommend: log-only default with a separate EVAL_POST_INGESTION_BLOCKS_ON_FAILURE flag
 (default false). Confirm before implementing.

 ---
 Critical Files Touched

 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
 │                                              Path                                               │                                        Action                                         │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/config.py                    │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/phoenix_bootstrap.py         │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/datasets/loaders.py          │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/generation/synthetic.py      │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/pipeline/task.py             │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/pipeline/adapter.py          │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/evaluators/ragas_metrics.py  │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/evaluators/phoenix_judges.py │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/evaluators/thresholds.py     │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/reporting/artifacts.py       │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/orchestration/evaluation/flows.py       │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/orchestration/evaluation/tasks.py       │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/orchestration/evaluation/deployments.py │ NEW                                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/orchestration/initialize.py             │ MODIFY (extend register_deployments)                                                  │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/orchestration/ingestion/flows.py        │ MODIFY (optional post-ingestion hook)                                                 │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/evaluate.py                  │ DELETE                                                                                │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/evaluation/question_generation.py       │ DELETE                                                                                │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/.env.example                            │ MODIFY (add EVAL_* block, deprecate SYNTHETIC_QA_PERSIST_PATH, extend logging extras) │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/pyproject.toml                          │ MODIFY (add [project.optional-dependencies] eval)                                     │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/tests/test_evaluation_imports.py        │ NEW (CI smoke test)                                                                   │
 ├─────────────────────────────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
 │ /Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/Makefile                                │ MODIFY (add eval + register-eval-deployment)                                          │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘
