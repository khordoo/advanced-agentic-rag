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
import os
import sys
import time

import httpx
from prefect.client.orchestration import get_client

from orchestration.ingestion.flows import end_to_end_flow, factory_reset_flow

logger = logging.getLogger("orchestration.initialize")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration (read from env, matching docker-compose.yaml)
# ---------------------------------------------------------------------------
PREFECT_API_URL: str = os.getenv("PREFECT_API_URL", "http://localhost:4200/api")
WORK_POOL_NAME: str = os.getenv("PREFECT_INGESTION_WORK_POOL", "ingestion-pool")
WORK_POOL_TYPE: str = "process"

DEPLOYMENT_NAME: str = os.getenv(
    "PREFECT_DEPLOYMENT_NAME", "local-ingestion-runner"
)

# Server readiness probe
MAX_WAIT_SECONDS: int = int(os.getenv("PREFECT_INIT_MAX_WAIT", "120"))
INITIAL_BACKOFF: float = 1.0
MAX_BACKOFF: float = 10.0


# ---------------------------------------------------------------------------
# 1. Wait for Prefect API readiness
# ---------------------------------------------------------------------------
def wait_for_server() -> None:
    """Block until the Prefect API server responds on /api/health.

    Uses exponential backoff with a ceiling.  Exits the process with
    code 1 if the server is not reachable within *MAX_WAIT_SECONDS*.
    """
    health_url = f"{PREFECT_API_URL}/health"
    logger.info("Waiting for Prefect API at %s …", health_url)

    backoff = INITIAL_BACKOFF
    deadline = time.monotonic() + MAX_WAIT_SECONDS

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
        backoff = min(backoff * 2, MAX_BACKOFF)

    logger.critical(
        "Prefect API did not become healthy within %ds. Aborting.",
        MAX_WAIT_SECONDS,
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
            pool = await client.read_work_pool(WORK_POOL_NAME)
            logger.info(
                "Work pool '%s' already exists (type=%s) — skipping creation ✓",
                pool.name,
                pool.type,
            )
            return
        except ObjectNotFound:
            logger.info(
                "Work pool '%s' not found — creating (type=%s) …",
                WORK_POOL_NAME,
                WORK_POOL_TYPE,
            )

        from prefect.client.schemas.actions import WorkPoolCreate

        await client.create_work_pool(
            work_pool=WorkPoolCreate(
                name=WORK_POOL_NAME,
                type=WORK_POOL_TYPE,
            )
        )
        logger.info("Work pool '%s' created ✓", WORK_POOL_NAME)


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
    logger.info("Registering deployments to pool '%s' …", WORK_POOL_NAME)

    async with get_client() as client:
        for flow_obj, suffix, tags, description in _DEPLOYMENT_SPECS:
            # Step 1: Ensure the flow is registered and get its server-side UUID.
            flow_id = await client.create_flow_from_name(flow_obj.name)

            # Step 2: Build the entrypoint string Prefect uses to locate the flow.
            # Format: "module/path.py:flow_function_name"
            entrypoint = f"orchestration/ingestion/flows.py:{flow_obj.fn.__name__}"

            deployment_name = f"{DEPLOYMENT_NAME}{suffix}"

            # Step 3: Upsert the deployment.
            deployment_id = await client.create_deployment(
                flow_id=flow_id,
                name=deployment_name,
                work_pool_name=WORK_POOL_NAME,
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

