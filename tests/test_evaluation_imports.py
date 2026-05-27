"""CI smoke tests for the evaluation module.

These tests verify the Day-1 open-source contract:
  - Importing evaluation modules does NOT require ragas / phoenix
  - Eval deployments are registered without importing heavy deps
  - Eval flows validate optional deps at execution time

No Phoenix server, no heavy ML deps required.
"""

from __future__ import annotations

import os
import importlib
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
        """evaluation.orchestration.flows imports cleanly."""
        import evaluation.orchestration.flows

        assert hasattr(evaluation.orchestration.flows, "eval_golden_flow")
        assert hasattr(evaluation.orchestration.flows, "eval_trace_flow")
        assert hasattr(evaluation.orchestration.flows, "synthetic_qa_flow")

    def test_import_deployments(self):
        """evaluation.orchestration.deployments imports cleanly."""
        import evaluation.orchestration.deployments

        assert hasattr(
            evaluation.orchestration.deployments, "build_eval_deployment_specs"
        )

    def test_import_tasks(self):
        """evaluation.orchestration.tasks imports cleanly."""
        import evaluation.orchestration.tasks

        assert hasattr(evaluation.orchestration.tasks, "resolve_dataset_task")
        assert hasattr(evaluation.orchestration.tasks, "run_experiment_task")

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
            from evaluation.orchestration import flows

            with patch.object(
                flows,
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
        from evaluation.orchestration.deployments import build_eval_deployment_specs

        get_eval_config.cache_clear()
        try:
            config = EvalConfig(_env_file=None)
            specs = build_eval_deployment_specs(config)
            assert [spec[1] for spec in specs] == [
                "evaluate-ground-truth",
                "evaluate-traces",
                "generate-synthetic-qa",
            ]
        finally:
            get_eval_config.cache_clear()

    @patch.dict(os.environ, {"EVAL_ENABLED": "false"}, clear=False)
    def test_trace_flow_runs_when_runtime_ready(self):
        """Trace flow no longer returns None due to legacy EVAL_ENABLED=false."""
        from evaluation.config import get_eval_config

        get_eval_config.cache_clear()
        try:
            from evaluation.orchestration import flows

            with patch.object(flows, "_assert_eval_runtime_ready"):
                result = flows.eval_trace_flow.fn()
            assert result == {"status": "not_implemented", "lookback_minutes": 60}
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
        """Missing eval deps point users at make eval."""
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
        assert "make eval" in message
        assert "Evaluation flows require the [eval] extras" in message


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
