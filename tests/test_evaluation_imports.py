"""CI smoke tests for the evaluation module.

These tests verify the Day-1 open-source contract:
  - Importing evaluation modules does NOT require ragas / phoenix
  - Eval deployments are registered without importing heavy deps
  - Eval flows validate optional deps at execution time

No Phoenix server, no heavy ML deps required.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


class TestEvaluationImports:
    """Verify that importing eval modules is safe without [eval] extra."""

    def test_import_config(self):
        """evaluation.config imports cleanly (stdlib + pydantic only)."""
        import evaluation.config

        assert hasattr(evaluation.config, "EvalConfig")
        assert hasattr(evaluation.config, "get_eval_config")

    def test_import_flows(self):
        """orchestration.evaluation.flows imports cleanly."""
        import orchestration.evaluation.flows

        assert hasattr(orchestration.evaluation.flows, "eval_golden_flow")
        assert hasattr(orchestration.evaluation.flows, "eval_trace_flow")
        assert hasattr(orchestration.evaluation.flows, "synthetic_qa_flow")

    def test_import_deployments(self):
        """orchestration.evaluation.deployments imports cleanly."""
        import orchestration.evaluation.deployments

        assert hasattr(
            orchestration.evaluation.deployments, "build_eval_deployment_specs"
        )

    def test_import_tasks(self):
        """orchestration.evaluation.tasks imports cleanly."""
        import orchestration.evaluation.tasks

        assert hasattr(orchestration.evaluation.tasks, "resolve_dataset_task")
        assert hasattr(orchestration.evaluation.tasks, "run_experiment_task")

    def test_import_adapter(self):
        """evaluation.pipeline.adapter imports cleanly (stdlib re only)."""
        import evaluation.pipeline.adapter

        assert hasattr(evaluation.pipeline.adapter, "strip_file_refs")
        assert hasattr(evaluation.pipeline.adapter, "to_ragas_sample")

    def test_import_thresholds(self):
        """evaluation.evaluators.thresholds imports cleanly."""
        import evaluation.evaluators.thresholds

        assert hasattr(evaluation.evaluators.thresholds, "PassFailReport")
        assert hasattr(evaluation.evaluators.thresholds, "evaluate_thresholds")

    def test_import_custom_judges(self):
        """Custom judges import without Phoenix or Ragas."""
        import evaluation.evaluators.custom_judges

        assert hasattr(
            evaluation.evaluators.custom_judges,
            "SourceReferenceFormatJudge",
        )

    def test_import_eval_registry(self):
        """The CE registry imports without optional eval dependencies."""
        import evaluation.eval_registry

        assert hasattr(evaluation.eval_registry, "GROUND_TRUTH_EVALUATORS")
        assert hasattr(evaluation.eval_registry, "DRIFT_EVALUATORS")

    def test_import_prefect_ce_flows(self):
        """CE Prefect flows import cleanly."""
        import orchestration.evaluation.prefect_flows

        assert hasattr(
            orchestration.evaluation.prefect_flows,
            "run_ground_truth_evaluation",
        )
        assert hasattr(
            orchestration.evaluation.prefect_flows,
            "run_nightly_drift_detection",
        )


class TestEvalRuntimeGatingContract:
    """Verify that eval is runtime-gated instead of registration-gated.

    These tests call the underlying flow function (``flow_obj.fn``) directly
    to avoid Prefect attempting to connect to its API server, which is
    unavailable in CI.  This is a standard Prefect unit-test pattern.
    """

    @patch.dict(os.environ, {"EVAL_ENABLED": "false"}, clear=False)
    def test_golden_flow_checks_runtime_even_when_legacy_enabled_false(self):
        """Legacy EVAL_ENABLED=false must not skip a requested eval run."""
        from evaluation.config import get_eval_config
        from evaluation.runtime import EvaluationRuntimeError

        get_eval_config.cache_clear()
        try:
            from orchestration.evaluation import flows, prefect_flows

            with patch.object(
                prefect_flows,
                "_assert_eval_runtime_ready",
                side_effect=EvaluationRuntimeError("runtime check called"),
            ):
                with pytest.raises(EvaluationRuntimeError, match="runtime check called"):
                    flows.eval_golden_flow.fn()
        finally:
            get_eval_config.cache_clear()

    @patch.dict(os.environ, {"EVAL_ENABLED": "false"}, clear=False)
    def test_deployment_specs_return_all_eval_deployments(self):
        """build_eval_deployment_specs always returns the eval deployment set."""
        from evaluation.config import EvalConfig, get_eval_config
        from orchestration.evaluation.deployments import build_eval_deployment_specs

        get_eval_config.cache_clear()
        try:
            config = EvalConfig(_env_file=None)
            specs = build_eval_deployment_specs(config)
            assert [spec[1] for spec in specs] == [
                "evaluate-ground-truth",
                "evaluate-traces",
                "generate-synthetic-qa",
            ]
            assert specs[0][0].fn.__name__ == "run_ground_truth_evaluation"
            assert specs[1][0].fn.__name__ == "run_nightly_drift_detection"
        finally:
            get_eval_config.cache_clear()

    @patch.dict(os.environ, {"EVAL_ENABLED": "false"}, clear=False)
    def test_trace_flow_delegates_to_drift_detection(self):
        """Legacy trace wrapper delegates to the canonical CE drift flow."""
        from evaluation.config import get_eval_config

        get_eval_config.cache_clear()
        try:
            from orchestration.evaluation import flows
            from orchestration.evaluation import prefect_flows

            with patch.object(
                prefect_flows.run_nightly_drift_detection,
                "fn",
                return_value={"status": "delegated"},
            ) as delegated:
                result = flows.eval_trace_flow.fn()
            assert result == {"status": "delegated"}
            delegated.assert_called_once_with(lookback_minutes=None)
        finally:
            get_eval_config.cache_clear()


class TestEvalConfigParsing:
    """Verify user-friendly .env formats parse into typed config values."""

    @patch.dict(
        os.environ,
        {
            "EVAL_RAGAS_METRICS": "faithfulness,answer_relevancy",
            "EVAL_PHOENIX_EVALUATORS": "hallucination,qa_correctness",
        },
        clear=False,
    )
    def test_csv_evaluator_lists_parse(self):
        """Comma-separated .env values are accepted for evaluator lists."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.ragas_metrics == ["faithfulness", "answer_relevancy"]
        assert config.phoenix_evaluators == [
            "hallucination",
            "qa_correctness",
        ]

    @patch.dict(
        os.environ,
        {
            "EVAL_RAGAS_METRICS": '["faithfulness", "context_recall"]',
            "EVAL_PHOENIX_EVALUATORS": '["hallucination", "relevance"]',
        },
        clear=False,
    )
    def test_json_evaluator_lists_parse(self):
        """JSON arrays remain accepted for users who prefer that format."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.ragas_metrics == ["faithfulness", "context_recall"]
        assert config.phoenix_evaluators == ["hallucination", "relevance"]

    @patch.dict(os.environ, {"EVAL_THRESHOLDS": ""}, clear=False)
    def test_empty_thresholds_env_parses_as_empty_dict(self):
        """An empty EVAL_THRESHOLDS value keeps warn-only mode."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.thresholds == {}

    @patch.dict(os.environ, {"EVAL_MAX_EXAMPLES": ""}, clear=False)
    def test_empty_max_examples_env_parses_as_none(self):
        """An empty EVAL_MAX_EXAMPLES value means no hard cap."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.max_examples is None

    @patch.dict(os.environ, {"EVAL_AUTO_RUN_AFTER_INGESTION": "true"}, clear=False)
    def test_auto_run_after_ingestion_env_parses(self):
        """Post-ingestion eval auto-run uses the new env variable name."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.auto_run_after_ingestion is True

    @patch.dict(
        os.environ,
        {
            "EVAL_TIMEOUT": "120",
            "EVAL_MAX_RETRIES": "1",
            "EVAL_MAX_WORKERS": "4",
        },
        clear=False,
    )
    def test_eval_tuning_env_parses(self):
        """Evaluation tuning values parse into typed config values."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.timeout == 120
        assert config.max_retries == 1
        assert config.max_workers == 4
        assert config.concurrency == 4

    @patch.dict(os.environ, {"EVAL_CONCURRENCY": "7"}, clear=False)
    def test_legacy_concurrency_env_maps_to_max_workers(self):
        """Legacy EVAL_CONCURRENCY remains a compatibility alias."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.max_workers == 7
        assert config.concurrency == 7


class TestFactualityWeight:
    """Verify the EVAL_FACTUALITY_WEIGHT config for AnswerCorrectness tuning."""

    def test_factuality_weight_default(self):
        """Default factuality weight matches RAGAS upstream default (0.75)."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.factuality_weight == 0.75
        assert config.semantic_weight == pytest.approx(0.25)

    @patch.dict(os.environ, {"EVAL_FACTUALITY_WEIGHT": "0.85"}, clear=False)
    def test_factuality_weight_from_env(self):
        """EVAL_FACTUALITY_WEIGHT env var is parsed into a float."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.factuality_weight == pytest.approx(0.85)
        assert config.semantic_weight == pytest.approx(0.15)

    @patch.dict(os.environ, {"EVAL_FACTUALITY_WEIGHT": "0.0"}, clear=False)
    def test_factuality_weight_zero_all_semantic(self):
        """Setting factuality_weight=0.0 gives full weight to semantics."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.factuality_weight == 0.0
        assert config.semantic_weight == pytest.approx(1.0)

    @patch.dict(os.environ, {"EVAL_FACTUALITY_WEIGHT": "1.0"}, clear=False)
    def test_factuality_weight_one_all_factual(self):
        """Setting factuality_weight=1.0 gives full weight to factuality."""
        from evaluation.config import EvalConfig

        config = EvalConfig(_env_file=None)

        assert config.factuality_weight == 1.0
        assert config.semantic_weight == pytest.approx(0.0)

    def test_factuality_weight_rejects_above_one(self):
        """factuality_weight > 1.0 is rejected by Pydantic ge/le constraint."""
        from pydantic import ValidationError
        from evaluation.config import EvalConfig

        with pytest.raises(ValidationError):
            EvalConfig(_env_file=None, factuality_weight=1.5)

    def test_factuality_weight_rejects_negative(self):
        """factuality_weight < 0.0 is rejected by Pydantic ge/le constraint."""
        from pydantic import ValidationError
        from evaluation.config import EvalConfig

        with pytest.raises(ValidationError):
            EvalConfig(_env_file=None, factuality_weight=-0.1)

    def test_weights_always_sum_to_one(self):
        """factuality_weight + semantic_weight always sums to 1.0."""
        from evaluation.config import EvalConfig

        for value in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            config = EvalConfig(_env_file=None, factuality_weight=value)
            assert config.factuality_weight + config.semantic_weight == pytest.approx(1.0)

class TestEvalRuntimeChecks:
    """Verify eval dependency checks fail with actionable messages."""

    def test_ragas_compat_shims_removed_vertexai_import_path(self):
        """RAGAS can import its stale VertexAI path without supporting VertexAI."""
        from evaluation.ragas_compat import apply_ragas_compat

        module_name = "langchain_community.chat_models.vertexai"
        sys.modules.pop(module_name, None)

        apply_ragas_compat()

        module = importlib.import_module(module_name)
        with pytest.raises(RuntimeError, match="VertexAI is not supported"):
            module.ChatVertexAI()

    def test_runtime_check_reports_missing_eval_dependency(self):
        """Missing eval deps produce actionable error message."""
        from evaluation.runtime import EvaluationRuntimeError
        from evaluation.runtime import assert_eval_runtime_ready

        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "ragas":
                raise ModuleNotFoundError(
                    "No module named 'ragas'",
                    name="ragas",
                )
            return real_import_module(name, package=package)

        with patch("importlib.import_module", side_effect=fake_import_module):
            with pytest.raises(EvaluationRuntimeError) as exc_info:
                assert_eval_runtime_ready()

        message = str(exc_info.value)
        assert "ragas" in message
        assert "standard worker image" in message
        assert "--extra eval" in message


class TestEvalDatasetUpload:
    """Verify Phoenix dataset upload uses the current client contract."""

    def _write_csv(self, tmp_path):
        path = tmp_path / "ground_truth_dataset.csv"
        path.write_text(
            "question,ground_truth\n"
            "What is A?,A answer\n"
            "What is B?,B answer\n",
            encoding="utf-8",
        )
        return path

    def test_dataset_upload_uses_dataframe_metadata_keys(self, tmp_path):
        """Dataframe uploads must not pass per-example metadata records."""
        from evaluation.datasets.loaders import ensure_dataset_uploaded

        client = Mock()
        client.datasets.get_dataset.side_effect = ValueError("not found")
        config = SimpleNamespace(
            dataset_csv_path=self._write_csv(tmp_path),
            dataset_name="rag-golden-v1",
            max_examples=None,
        )

        with patch(
            "evaluation.phoenix_bootstrap.get_phoenix_client",
            return_value=client,
        ):
            result = ensure_dataset_uploaded(config)

        assert result == "rag-golden-v1"
        kwargs = client.datasets.create_dataset.call_args.kwargs
        assert kwargs["name"] == "rag-golden-v1"
        assert kwargs["input_keys"] == ["input"]
        assert kwargs["output_keys"] == ["expected"]
        assert kwargs["metadata_keys"] == ["content_sha256"]
        assert "metadata" not in kwargs
        assert set(kwargs["dataframe"].columns) == {
            "input",
            "expected",
            "content_sha256",
        }

    def test_dataset_upload_skips_existing_matching_hash(self, tmp_path):
        """A matching Phoenix dataset version should not upload again."""
        from evaluation.datasets.loaders import compute_dataset_hash
        from evaluation.datasets.loaders import ensure_dataset_uploaded
        from evaluation.datasets.loaders import load_csv_rows

        csv_path = self._write_csv(tmp_path)
        content_hash = compute_dataset_hash(load_csv_rows(csv_path))
        examples = {
            "1": SimpleNamespace(metadata={"content_sha256": content_hash}),
            "2": SimpleNamespace(metadata={"content_sha256": content_hash}),
        }
        client = Mock()
        client.datasets.get_dataset.return_value = SimpleNamespace(examples=examples)
        config = SimpleNamespace(
            dataset_csv_path=csv_path,
            dataset_name="rag-golden-v1",
            max_examples=None,
        )

        with patch(
            "evaluation.phoenix_bootstrap.get_phoenix_client",
            return_value=client,
        ):
            result = ensure_dataset_uploaded(config)

        assert result == "rag-golden-v1"
        client.datasets.create_dataset.assert_not_called()

    def test_dataset_upload_reuploads_when_hash_differs(self, tmp_path):
        """A stale Phoenix dataset should upload a new version."""
        from evaluation.datasets.loaders import ensure_dataset_uploaded

        examples = {
            "1": SimpleNamespace(metadata={"content_sha256": "old"}),
            "2": SimpleNamespace(metadata={"content_sha256": "old"}),
        }
        client = Mock()
        client.datasets.get_dataset.return_value = SimpleNamespace(examples=examples)
        config = SimpleNamespace(
            dataset_csv_path=self._write_csv(tmp_path),
            dataset_name="rag-golden-v1",
            max_examples=None,
        )

        with patch(
            "evaluation.phoenix_bootstrap.get_phoenix_client",
            return_value=client,
        ):
            ensure_dataset_uploaded(config)

        client.datasets.create_dataset.assert_called_once()

    def test_dataset_upload_hash_respects_max_examples(self, tmp_path):
        """The idempotency hash should describe capped uploaded rows only."""
        from evaluation.datasets.loaders import ensure_dataset_uploaded

        client = Mock()
        client.datasets.get_dataset.side_effect = ValueError("not found")
        config = SimpleNamespace(
            dataset_csv_path=self._write_csv(tmp_path),
            dataset_name="rag-golden-v1",
            max_examples=1,
        )

        with patch(
            "evaluation.phoenix_bootstrap.get_phoenix_client",
            return_value=client,
        ):
            ensure_dataset_uploaded(config)

        df = client.datasets.create_dataset.call_args.kwargs["dataframe"]
        assert len(df) == 1
        assert df["input"].tolist() == ["What is A?"]


class TestRagasTuning:
    """Verify RAGAS picks up evaluation tuning from EvalConfig."""

    def test_build_ragas_evaluators_uses_eval_config_tuning(self):
        from evaluation.config import EvalConfig
        from evaluation.evaluators import ragas_metrics

        captured = {}

        class FakeRunConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        class FakeEmbeddingsWrapper:
            def __init__(self, adapter, run_config=None):
                self.adapter = adapter
                self.run_config = run_config

        config = EvalConfig(_env_file=None)
        config.ragas_metrics = []
        config.timeout = 120
        config.max_retries = 1
        config.max_workers = 4

        fake_embedding_model = SimpleNamespace(
            model_name="fake",
            embed_query=lambda text: [0.0],
            embed_documents=lambda texts: [[0.0] for _ in texts],
        )

        with patch("rag_core.agents.AgentFactory.create", return_value=SimpleNamespace()):
            with patch("rag_core.embeddings.create_embedding_model", return_value=fake_embedding_model):
                with patch("ragas.run_config.RunConfig", FakeRunConfig):
                    with patch("ragas.llms.LangchainLLMWrapper", lambda llm: llm):
                        with patch("ragas.embeddings.LangchainEmbeddingsWrapper", FakeEmbeddingsWrapper):
                            result = ragas_metrics.build_ragas_evaluators({}, config)

        assert result == []
        assert captured == {"timeout": 120, "max_retries": 1, "max_workers": 4}


class TestThresholdsLogic:
    """Verify threshold evaluation without heavy deps."""

    def test_empty_thresholds_never_fails(self):
        """Empty thresholds dict → warn-only mode (never fails)."""
        from evaluation.evaluators.thresholds import evaluate_thresholds

        class FakeConfig:
            thresholds = {}

        report = evaluate_thresholds(
            {"faithfulness": 0.5, "context_recall": 0.3}, FakeConfig()
        )
        assert report.failed is False
        assert report.breaches == []

    def test_threshold_breach_detected(self):
        """Scores below threshold are correctly flagged."""
        from evaluation.evaluators.thresholds import evaluate_thresholds

        class FakeConfig:
            thresholds = {"faithfulness": 0.9, "context_recall": 0.85}

        report = evaluate_thresholds(
            {"faithfulness": 0.83, "context_recall": 0.90}, FakeConfig()
        )
        assert report.failed is True
        assert len(report.breaches) == 1
        assert report.breaches[0][0] == "faithfulness"

    def test_all_thresholds_pass(self):
        """Scores above threshold → pass."""
        from evaluation.evaluators.thresholds import evaluate_thresholds

        class FakeConfig:
            thresholds = {"faithfulness": 0.8}

        report = evaluate_thresholds({"faithfulness": 0.95}, FakeConfig())
        assert report.failed is False


class TestAdapterLogic:
    """Verify adapter functions without heavy deps."""

    def test_strip_file_refs(self):
        """strip_file_refs removes [[file.pdf]] markers."""
        from evaluation.pipeline.adapter import strip_file_refs

        text = "The answer is 42 [[report.pdf]] based on analysis [[data.csv]]."
        result = strip_file_refs(text)
        assert "[[" not in result
        assert "report.pdf" not in result
        assert "The answer is 42" in result

    def test_to_ragas_sample(self):
        """to_ragas_sample produces correct structure."""
        from evaluation.pipeline.adapter import to_ragas_sample

        example = {"input": "What is X?", "expected": "X is Y."}
        output = {
            "response": "X is Y [[doc.pdf]]",
            "retrieved_contexts": ["context 1", "context 2"],
        }
        sample = to_ragas_sample(example, output)
        assert sample["user_input"] == "What is X?"
        assert sample["reference"] == "X is Y."
        assert "[[" not in sample["response"]
        assert len(sample["retrieved_contexts"]) == 2


class TestEvaluationRagTask:
    """Verify RAG task output preserves structured citation metadata."""

    def test_run_rag_for_example_returns_agent_output_for_custom_judges(self):
        from langchain_core.documents import Document

        from evaluation.pipeline.task import _run_rag_for_example_async

        class FakeRagService:
            async def generate_chat_completion(self, chat_req, include_source_docs=False):
                return SimpleNamespace(
                    agent_output={
                        "answer": "Revenue rose. 【1】",
                        "citations": [{"document_index": 1, "asset_key": "doc"}],
                    },
                    source_docs=[Document(page_content="context")],
                )

        output = asyncio.run(
            _run_rag_for_example_async(
                {"input": "What happened?"},
                rag_service=FakeRagService(),
            )
        )

        assert output["response"] == "Revenue rose. 【1】"
        assert output["retrieved_contexts"] == ["context"]
        assert output["agent_output"]["citations"][0]["document_index"] == 1
        assert json.loads(output["raw_response"])["citations"][0]["document_index"] == 1

    def test_to_ragas_sample_accepts_phoenix_dataset_example_object(self):
        """Phoenix v16 dataset example objects expose input/output attrs."""
        from evaluation.pipeline.adapter import to_ragas_sample

        example = SimpleNamespace(
            input={"input": "What is X?"},
            output={"expected": "X is Y."},
        )
        output = {"response": "X is Y", "retrieved_contexts": []}

        sample = to_ragas_sample(example, output)

        assert sample["user_input"] == "What is X?"
        assert sample["reference"] == "X is Y."


class TestContinuousEvaluationRegistry:
    """Verify the CE registry uses the ground-truth/drift semantic boundary."""

    def test_ground_truth_registry_contains_reference_metrics(self):
        from evaluation.eval_registry import GROUND_TRUTH_EVALUATORS

        names = [item.name for item in GROUND_TRUTH_EVALUATORS]

        assert names == [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "reference_format_judge",
        ]

    def test_drift_registry_excludes_reference_dependent_metrics(self):
        from evaluation.eval_registry import DRIFT_EVALUATORS

        names = [item.name for item in DRIFT_EVALUATORS]

        assert names == [
            "faithfulness",
            "answer_relevancy",
            "reference_format_judge",
        ]
        assert "context_precision" not in names
        assert "context_recall" not in names


class TestSourceReferenceFormatJudge:
    """Verify deterministic citation-contract scoring."""

    def test_exposes_clean_phoenix_callable_name(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        judge = SourceReferenceFormatJudge()

        assert judge.name == "reference_format_judge"
        assert judge.__name__ == "reference_format_judge"
        assert judge.__qualname__ == "reference_format_judge"
        assert repr(judge) == "reference_format_judge"
        assert str(judge) == "reference_format_judge"

    def test_accepts_json_ascii_citation_with_matching_metadata(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {
            "answer": "The answer is 42. [1]",
            "citations": [{"document_index": 1, "asset_key": "doc"}],
        }

        result = SourceReferenceFormatJudge().evaluate(json.dumps(payload))

        assert result["score"] == 1.0
        assert result["label"] == "formatted_and_matched_correctly"

    def test_accepts_json_fullwidth_citation_with_matching_metadata(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {
            "answer": "Revenue rose. 【1】",
            "citations": [{"document_index": 1, "asset_key": "doc"}],
        }

        result = SourceReferenceFormatJudge().evaluate(json.dumps(payload))

        assert result["score"] == 1.0
        assert result["label"] == "formatted_and_matched_correctly"

    def test_accepts_json_without_inline_citations(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {"answer": "Revenue rose.", "citations": []}

        result = SourceReferenceFormatJudge().evaluate(json.dumps(payload))

        assert result["score"] == 1.0
        assert result["label"] == "no_citations_needed"

    def test_rejects_json_inline_citation_without_structured_citations(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {"answer": "Revenue rose. [1]", "citations": []}

        result = SourceReferenceFormatJudge().evaluate(json.dumps(payload))

        assert result["score"] == 0.0
        assert result["label"] == "missing_structured_citations"

    def test_rejects_json_unmatched_structured_citations(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {
            "answer": "Revenue rose. [2]",
            "citations": [{"document_index": 1, "asset_key": "doc"}],
        }

        result = SourceReferenceFormatJudge().evaluate(json.dumps(payload))

        assert result["score"] == 0.0
        assert result["label"] == "unmatched_citations_[2]"

    def test_plain_text_without_citations_passes(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        result = SourceReferenceFormatJudge().evaluate("Revenue rose.")

        assert result["score"] == 1.0
        assert result["label"] == "no_citations_needed"

    def test_plain_text_with_inline_citation_fails_without_metadata(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        result = SourceReferenceFormatJudge().evaluate("Revenue rose. [1]")

        assert result["score"] == 0.0
        assert result["label"] == "missing_structured_citations"

    def test_call_prefers_structured_agent_output_over_plain_response(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        result = SourceReferenceFormatJudge()(
            input={"input": "question"},
            output={
                "response": "Revenue rose. 【1】",
                "retrieved_contexts": ["context"],
                "agent_output": {
                    "answer": "Revenue rose. 【1】",
                    "citations": [{"document_index": 1, "asset_key": "doc"}],
                },
            },
            expected={},
            metadata={},
        )

        assert result["score"] == 1.0
        assert result["label"] == "formatted_and_matched_correctly"

    def test_call_uses_raw_response_when_agent_output_is_unavailable(self):
        from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge

        payload = {
            "answer": "Revenue rose. 【1】",
            "citations": [{"document_index": 1, "asset_key": "doc"}],
        }

        result = SourceReferenceFormatJudge()(
            output={
                "response": "Revenue rose. 【1】",
                "raw_response": json.dumps(payload),
            }
        )

        assert result["score"] == 1.0
        assert result["label"] == "formatted_and_matched_correctly"


class TestContinuousEvaluationFlowHelpers:
    """Verify CE flow helper behavior without Phoenix or Ragas calls."""

    def test_safe_evaluate_row_does_not_raise_for_evaluator_failure(self):
        from orchestration.evaluation.prefect_flows import _safe_evaluate_row

        def broken_evaluator(**kwargs):
            raise RuntimeError("judge failed")

        broken_evaluator.name = "broken_judge"

        result = _safe_evaluate_row(
            evaluator=broken_evaluator,
            question="What is revenue?",
            contexts=["context"],
            answer="answer",
            ground_truth="truth",
        )

        assert result["name"] == "broken_judge"
        assert result["score"] == 0.0
        assert result["label"] == "error"
        assert "judge failed" in result["error"]

    def test_failed_verification_filter_targets_otel_attribute(self):
        from orchestration.evaluation.prefect_flows import _FAILED_VERIFICATION_FILTER

        assert _FAILED_VERIFICATION_FILTER == (
            "attributes['custom.verification_status'] == 'failed'"
        )
