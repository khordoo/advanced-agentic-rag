from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_core.prompts.loader import AgentPrompt
from rag_core.prompts.profile_config import (
    clear_prompt_profile_cache,
    resolve_prompt_profile_context,
)


def _reset_prompt_caches() -> None:
    clear_prompt_profile_cache()
    AgentPrompt._cache.clear()


def test_resolve_profile_context_from_default_profile():
    _reset_prompt_caches()
    context = resolve_prompt_profile_context("primary_rag_agent")
    assert context["company_context"] == "Procter & Gamble (P&G)"
    assert context["industry_domain"] == "corporate finance and SEC filings"
    assert context["role"] == "Senior Financial Data Extraction Specialist"
    assert "impairment charges" in context["expertise_field"]


def test_general_assistant_prompt_renders_with_profile_values():
    _reset_prompt_caches()
    prompt = AgentPrompt.from_package("general_assistant").chat_template.invoke(
        {"question": "hi"}
    )
    rendered = "\n".join(str(message.content) for message in prompt.to_messages())
    assert "polite AI financial analyst assistant" in rendered
    assert "Procter & Gamble (P&G)" in rendered
    assert "corporate finance and SEC filings" in rendered


def test_unknown_agent_key_rejected_in_strict_mode(tmp_path: Path, monkeypatch):
    invalid_profile_path = tmp_path / "invalid_profile.yaml"
    invalid_profile_path.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "company_context": "Demo Corp",
                    "industry_domain": "enterprise reports",
                },
                "agents": {
                    "non_existent_agent": {
                        "role": "invalid",
                        "expertise_field": "invalid",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPTS_PROFILE_PATH", str(invalid_profile_path))
    monkeypatch.setenv("PROMPTS_PROFILE_STRICT", "true")

    _reset_prompt_caches()
    with pytest.raises(RuntimeError, match="unknown agent package IDs"):
        resolve_prompt_profile_context("primary_rag_agent")


def test_unknown_agent_key_ignored_when_strict_disabled(
    tmp_path: Path, monkeypatch
):
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "company_context": "Demo Corp",
                    "industry_domain": "enterprise reports",
                },
                "agents": {
                    "primary_rag_agent": {
                        "role": "Data Specialist",
                        "expertise_field": "retrieval-grounded answers",
                    },
                    "non_existent_agent": {
                        "role": "invalid",
                        "expertise_field": "invalid",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPTS_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("PROMPTS_PROFILE_STRICT", "false")

    _reset_prompt_caches()
    context = resolve_prompt_profile_context("primary_rag_agent")
    assert context["company_context"] == "Demo Corp"
    assert context["industry_domain"] == "enterprise reports"
    assert context["role"] == "Data Specialist"


def test_partial_agent_override_keeps_fallback_fields(tmp_path: Path, monkeypatch):
    profile_path = tmp_path / "partial_profile.yaml"
    profile_path.write_text(
        yaml.safe_dump(
            {
                "global": {
                    "company_context": "Demo Corp",
                    "industry_domain": "enterprise reports",
                },
                "agents": {"primary_rag_agent": {"role": "Data Specialist"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PROMPTS_PROFILE_PATH", str(profile_path))
    monkeypatch.setenv("PROMPTS_PROFILE_STRICT", "true")

    _reset_prompt_caches()
    context = resolve_prompt_profile_context("primary_rag_agent")
    assert context["role"] == "Data Specialist"
    assert "impairment charges" in context["expertise_field"]
