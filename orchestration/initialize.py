"""Pre-flight orchestration setup: work-pool provisioning and deployment registration.

This module is executed by the ingestion-worker container entrypoint
**before** the Prefect worker process starts.  Every operation is idempotent
so re-running it on container restart or stack recreation is a no-op.

Usage (inside container or locally):
    python -m orchestration.initialize
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time

import httpx
from prefect.client.orchestration import get_client

from orchestration.config import get_prefect_config
from orchestration.ingestion.flows import end_to_end_flow, factory_reset_flow

logger = logging.getLogger("orchestration.initialize")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
)
config = get_prefect_config()


# ---------------------------------------------------------------------------
# 1. Wait for Prefect API readiness
# ---------------------------------------------------------------------------
def wait_for_server() -> None:
    """Block until the Prefect API server responds on /api/health.

    Uses exponential backoff with a ceiling.  Exits the process with
    code 1 if the server is not reachable within *MAX_WAIT_SECONDS*.
    """
    health_url = f"{config.api_url}/health"
    logger.info("Waiting for Prefect API at %s …", health_url)

    backoff = config.initial_backoff
    deadline = time.monotonic() + config.max_wait_seconds

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code == 200:
                logger.info("Prefect API is healthy ✓")
                return
            logger.warning(
                "Prefect API returned %s — retrying in %.1fs",
                resp.status_code,
                backoff,
            )
        except httpx.ConnectError:
            logger.info("Prefect API not reachable yet — retrying in %.1fs", backoff)
        except httpx.HTTPError as exc:
            logger.warning("HTTP error probing Prefect API: %s", exc)

        time.sleep(backoff)
        backoff = min(backoff * 2, config.max_backoff)

    logger.critical(
        "Prefect API did not become healthy within %ds. Aborting.",
        config.max_wait_seconds,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# 2. Idempotent work-pool creation
# ---------------------------------------------------------------------------
async def ensure_work_pool() -> None:
    """Create the target work pool if it does not already exist.

    Uses the async ``PrefectClient`` for direct API interaction.
    Catches only ``ObjectNotFound`` (HTTP 404) to distinguish "pool
    doesn't exist yet" from real failures (503, disk full, network).
    """
    from prefect.exceptions import ObjectNotFound

    async with get_client() as client:
        try:
            pool = await client.read_work_pool(config.work_pool_name)
            logger.info(
                "Work pool '%s' already exists (type=%s) — skipping creation ✓",
                pool.name,
                pool.type,
            )
            return
        except ObjectNotFound:
            logger.info(
                "Work pool '%s' not found — creating (type=%s) …",
                config.work_pool_name,
                config.work_pool_type,
            )

        from prefect.client.schemas.actions import WorkPoolCreate

        await client.create_work_pool(
            work_pool=WorkPoolCreate(
                name=config.work_pool_name,
                type=config.work_pool_type,
            )
        )
        logger.info("Work pool '%s' created ✓", config.work_pool_name)


# ---------------------------------------------------------------------------
# 3. Automated deployment registration
# ---------------------------------------------------------------------------

# Deployment specifications: each tuple is (flow_object, deployment_suffix, tags, description).
_DEPLOYMENT_SPECS: list[tuple] = [
    (
        end_to_end_flow,
        "",
        ["ingestion", "safe", "oss"],
        "Safe RAG end-to-end ingestion run with migration preflight.",
    ),
    (
        factory_reset_flow,
        "-factory-reset",
        ["ingestion", "admin", "dangerous", "destructive", "oss"],
        (
            "DANGEROUS: Wipes vector backend and ingestion registries. "
            "Requires confirm_destructive_wipe=true."
        ),
    ),
]


async def register_deployments() -> None:
    """Register (upsert) the canonical deployments via the low-level client API.

    Uses ``client.create_flow`` to register flow metadata (idempotent — the
    server returns the existing UUID if the flow name is already registered),
    then ``client.create_deployment`` to upsert the deployment record.

    This bypasses ``flow.deploy()`` which requires an image or remote storage
    in Prefect 3.x — unnecessary because the worker container already has
    the code baked into its image.
    """
    logger.info("Registering deployments to pool '%s' …", config.work_pool_name)

    async with get_client() as client:
        for flow_obj, suffix, tags, description in _DEPLOYMENT_SPECS:
            # Step 1: Ensure the flow is registered and get its server-side UUID.
            flow_id = await client.create_flow_from_name(flow_obj.name)

            # Step 2: Build the entrypoint string Prefect uses to locate the flow.
            # Format: "module/path.py:flow_function_name"
            entrypoint = f"orchestration/ingestion/flows.py:{flow_obj.fn.__name__}"

            deployment_name = f"{config.deployment_name}{suffix}"

            # Step 3: Upsert the deployment.
            deployment_id = await client.create_deployment(
                flow_id=flow_id,
                name=deployment_name,
                work_pool_name=config.work_pool_name,
                tags=tags,
                description=description,
                entrypoint=entrypoint,
                path="/app",
            )
            logger.info(
                "Deployment '%s' → flow '%s' registered (id=%s) ✓",
                deployment_name,
                flow_obj.name,
                deployment_id,
            )

    logger.info("All deployments registered ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _async_main() -> None:
    """Run all async pre-flight steps in a single event loop."""
    await ensure_work_pool()
    await register_deployments()


def main() -> None:
    """Run the full pre-flight initialization sequence."""
    logger.info("═══ Orchestration Pre-Flight Initialization ═══")

    wait_for_server()
    asyncio.run(_async_main())

    logger.info("═══ Pre-flight complete — handing off to worker ═══")


if __name__ == "__main__":
    main()
