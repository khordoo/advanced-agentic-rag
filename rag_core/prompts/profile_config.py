from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent
_PROMPT_PACKAGES_DIR = _PROMPTS_DIR / "packages"
_DEFAULT_PROFILE_PATH = _PROMPTS_DIR / "profiles" / "default.yaml"
_PROFILE_PATH_ENV = "PROMPTS_PROFILE_PATH"
_PROFILE_STRICT_ENV = "PROMPTS_PROFILE_STRICT"


class PromptGlobalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_context: str
    industry_domain: str


class PromptAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str | None = None
    expertise_field: str | None = None
    response_style: str | None = None


class PromptProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    global_config: PromptGlobalConfig = Field(alias="global")
    agents: dict[str, PromptAgentConfig] = Field(default_factory=dict)


def _parse_bool_env(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _known_prompt_package_ids() -> set[str]:
    if not _PROMPT_PACKAGES_DIR.exists():
        return set()
    return {
        path.name
        for path in _PROMPT_PACKAGES_DIR.iterdir()
        if path.is_dir() and (path / "template.yaml").exists()
    }


def _build_fallback_profile() -> PromptProfile:
    # Mirrors the current financial defaults so existing behavior remains stable
    # when no profile file is provided.
    return PromptProfile.model_validate(
        {
            "global": {
                "company_context": "Procter & Gamble (P&G)",
                "industry_domain": "corporate finance and SEC filings",
            },
            "agents": {
                "general_assistant": {
                    "role": "polite AI financial analyst assistant",
                    "expertise_field": "general financial concepts and conversational navigation",
                    "response_style": "warm, helpful, and strictly professional",
                },
                "primary_rag_agent": {
                    "role": "Senior Financial Data Extraction Specialist",
                    "expertise_field": "complex financial tables, Q3 revenue margins, and impairment charges",
                    "response_style": "clinical, highly concise, and objective",
                },
                "intent_router": {
                    "role": "analytical triage coordinator",
                    "expertise_field": "classifying inquiries into financial analysis vs. general conversation",
                },
            },
        }
    )


def _resolve_profile_path() -> tuple[Path, bool]:
    override_path = os.environ.get(_PROFILE_PATH_ENV)
    if override_path:
        return Path(override_path).expanduser(), True
    return _DEFAULT_PROFILE_PATH, False


def _validate_agent_keys(
    profile: PromptProfile, known_packages: set[str], strict: bool
) -> PromptProfile:
    unknown_agents = sorted(set(profile.agents.keys()) - known_packages)
    if not unknown_agents:
        return profile

    message = (
        "Prompt profile contains unknown agent package IDs: "
        f"{', '.join(unknown_agents)}"
    )
    if strict:
        raise ValueError(message)

    logger.warning("%s. Ignoring unknown entries because strict mode is disabled.", message)
    filtered_agents = {
        key: value for key, value in profile.agents.items() if key in known_packages
    }
    return PromptProfile(
        global_config=profile.global_config.model_dump(),
        agents=filtered_agents,
    )


def _merge_with_fallback_profile(profile: PromptProfile) -> PromptProfile:
    fallback = _build_fallback_profile()
    merged_agents = dict(fallback.agents)

    for package_id, override in profile.agents.items():
        base = merged_agents.get(package_id, PromptAgentConfig())
        merged_agents[package_id] = PromptAgentConfig(
            role=override.role if override.role is not None else base.role,
            expertise_field=(
                override.expertise_field
                if override.expertise_field is not None
                else base.expertise_field
            ),
            response_style=(
                override.response_style
                if override.response_style is not None
                else base.response_style
            ),
        )

    return PromptProfile(
        global_config=profile.global_config.model_dump(),
        agents=merged_agents,
    )


@lru_cache(maxsize=1)
def get_prompt_profile() -> PromptProfile:
    strict_mode = _parse_bool_env(os.environ.get(_PROFILE_STRICT_ENV), default=True)
    profile_path, is_explicit_path = _resolve_profile_path()
    known_packages = _known_prompt_package_ids()
    fallback_profile = _build_fallback_profile()

    if not profile_path.exists():
        if is_explicit_path:
            raise FileNotFoundError(
                f"Prompt profile file not found at {profile_path} "
                f"(from {_PROFILE_PATH_ENV})."
            )
        validated_fallback = _validate_agent_keys(
            fallback_profile, known_packages, strict=strict_mode
        )
        return _merge_with_fallback_profile(validated_fallback)

    try:
        raw_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
        if raw_profile is None:
            raise ValueError(f"Prompt profile at {profile_path} is empty.")
        loaded_profile = PromptProfile.model_validate(raw_profile)
        validated_profile = _validate_agent_keys(
            loaded_profile, known_packages, strict=strict_mode
        )
        return _merge_with_fallback_profile(validated_profile)
    except Exception as exc:
        if strict_mode:
            raise RuntimeError(
                f"Failed to load prompt profile from {profile_path}: {exc}"
            ) from exc
        logger.warning(
            "Failed to load prompt profile from %s: %s. Falling back to built-in defaults.",
            profile_path,
            exc,
        )
        validated_fallback = _validate_agent_keys(
            fallback_profile, known_packages, strict=strict_mode
        )
        return _merge_with_fallback_profile(validated_fallback)


def resolve_prompt_profile_context(package_id: str) -> dict[str, str]:
    profile = get_prompt_profile()
    resolved: dict[str, str] = {
        "company_context": profile.global_config.company_context,
        "industry_domain": profile.global_config.industry_domain,
    }

    agent_overrides = profile.agents.get(package_id)
    if agent_overrides is not None:
        if agent_overrides.role is not None:
            resolved["role"] = agent_overrides.role
        if agent_overrides.expertise_field is not None:
            resolved["expertise_field"] = agent_overrides.expertise_field
        if agent_overrides.response_style is not None:
            resolved["response_style"] = agent_overrides.response_style

    return resolved


def clear_prompt_profile_cache() -> None:
    get_prompt_profile.cache_clear()
