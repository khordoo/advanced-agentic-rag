# Generation Model Settings

These settings control model selection and generation orchestration.

## Core Model IDs

- `GENERATION_MODEL_PROVIDER`
- `GENERATION_REASONING_MODEL`
- `GENERATION_VERIFIER_MODEL`
- `GENERATION_FAST_MODEL`

## Runtime Controls

- `GENERATION_ENABLE_INTENT_ROUTING`
- `GENERATION_ENABLE_GROUNDING_VERIFIER`
- `GENERATION_MAX_AGENT_ITERATIONS`
- `GENERATION_MAX_TOOL_CALLS`
- `GENERATION_MAX_SELF_CORRECTION_ATTEMPTS`
- `GENERATION_INTENT_ROUTER_LOOKBACK`
- `GENERATION_GENERAL_ASSISTANT_LOOKBACK`

## Practical Guidance

- Keep intent routing and grounding verifier enabled for higher response quality.
- Lower iteration/tool-call limits to reduce cost and latency.
- Use a cheaper fast model for lightweight steps and keep the reasoning model for complex passes.
