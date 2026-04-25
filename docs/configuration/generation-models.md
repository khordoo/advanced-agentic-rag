# Generation Model Settings

This system uses three model roles:
- a reasoning model,
- a verifier model,
- and a fast model.

Choosing the right models has a major impact on performance for structured data and reasoning tasks.

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

## Model Selection Guidelines

### 1. Reasoning Model (Primary Agent)

The reasoning model is responsible for:
- understanding the question,
- retrieving relevant context,
- and performing reasoning and calculations.

For complex tasks such as:
- financial data analysis,
- table interpretation,
- multi-step reasoning,
- numerical aggregation,

use a large model (approximately 100B+ parameters).

Smaller models can be acceptable for:
- simple Q&A,
- keyword-based retrieval,
- general knowledge queries,

but they often struggle with:
- combining values across rows,
- interpreting structured tables,
- avoiding logical errors (for example, double counting).

### 2. Verifier Model (Grounding Check)

The verifier checks that answers are:
- supported by provided documents,
- numerically and logically consistent.

This model should be reliable and precise, and capable of detecting subtle inconsistencies.

A model with similar or slightly lower capability than the reasoning model is usually sufficient, as long as it can handle:
- numerical validation,
- scope checking (for example, quarterly vs annual).

### 3. Fast Model (Lightweight Tasks)

The fast model is used for:
- routing,
- lightweight transformations,
- simple intermediate steps.

A smaller, low-latency model is appropriate here because it is not expected to perform heavy reasoning.

## Practical Recommendation

For best results:
- use a large reasoning model (about 100B+) for structured and numerical tasks,
- use a robust verifier model to validate outputs,
- use a small, fast model for auxiliary steps.

If your workload includes financial reports, structured tables, or analytical queries, prioritize model quality over speed. Smaller models can produce plausible but incorrect answers.

## Operational Tuning

- Keep `GENERATION_ENABLE_INTENT_ROUTING=true` and `GENERATION_ENABLE_GROUNDING_VERIFIER=true` for higher answer quality.
- Lower iteration and tool-call limits when you need tighter latency/cost bounds.
- Keep the fast model lightweight; reserve the reasoning model for complex reasoning passes.
