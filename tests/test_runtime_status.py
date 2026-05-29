from types import SimpleNamespace

from rag_core.core import config


def test_log_runtime_status_logs_key_runtime_state(monkeypatch, caplog) -> None:
    monkeypatch.setattr(
        config,
        "get_generation_settings",
        lambda: SimpleNamespace(
            model_provider="aws",
            reasoning_model="reasoning-model",
            verifier_model="verifier-model",
            fast_model="fast-model",
            enable_intent_routing=True,
            enable_grounding_verifier=False,
        ),
    )
    monkeypatch.setattr(
        config,
        "get_retrieval_settings",
        lambda: SimpleNamespace(
            backend="qdrant",
            top_k=10,
            oversample_factor=5,
            hybrid_alpha=0.5,
        ),
    )
    monkeypatch.setattr(
        config,
        "get_reranker_settings",
        lambda: SimpleNamespace(
            enabled=True,
            provider="bedrock",
            model_name="amazon.rerank-v1:0",
            top_k=3,
        ),
    )

    with caplog.at_level("INFO"):
        config.log_runtime_status()

    messages = "\n".join(record.message for record in caplog.records)
    assert "Runtime status: retrieval backend=qdrant top_k=10 oversample=5 hybrid_alpha=0.50" in messages
    assert "Runtime status: reranker enabled provider=bedrock model=amazon.rerank-v1:0 top_k=3" in messages
    assert "Runtime status: generation provider=aws reasoning=reasoning-model verifier=verifier-model fast=fast-model intent_routing=True grounding_verifier=False" in messages
