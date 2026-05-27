"""Prefect flows for ingestion orchestration."""

from __future__ import annotations

import argparse
import logging

from prefect import flow
from prefect.exceptions import MissingContextError
from prefect import get_run_logger

from orchestration.ingestion.tasks import discovery_task
from orchestration.ingestion.tasks import end_to_end_ingestion_task
from orchestration.ingestion.tasks import ingestion_task
from orchestration.ingestion.tasks import reset_ingestion_registries_task
from orchestration.ingestion.tasks import reset_vector_backend_task
from orchestration.ingestion.tasks import run_database_migrations_task


@flow(name="rag-discovery-flow", log_prints=True)
def discovery_flow() -> dict[str, int] | None:
    """Orchestrate discovery only."""
    _get_logger().info("Running discovery flow")
    return discovery_task()


@flow(name="rag-ingestion-flow", log_prints=True)
def ingestion_flow(*, max_jobs: int | None = None) -> int:
    """Orchestrate ingestion worker execution only."""
    _get_logger().info("Running ingestion flow")
    return ingestion_task(max_jobs=max_jobs)


@flow(name="rag-end-to-end-ingestion-flow", log_prints=True)
def end_to_end_flow(
    *,
    max_jobs: int | None = None,
) -> dict[str, object]:
    """Orchestrate discovery followed by ingestion."""
    logger = _get_logger()
    logger.info("Running end-to-end ingestion flow")
    run_database_migrations_task()
    result = end_to_end_ingestion_task(max_jobs=max_jobs)

    # ── Optional post-ingestion eval hook (non-blocking) ─────────────
    try:
        from evaluation.config import get_eval_config  # lazy

        eval_config = get_eval_config()
        if eval_config.auto_run_after_ingestion:
            logger.info("Post-ingestion eval hook triggered — running ground truth eval")
            from evaluation.orchestration.flows import eval_golden_flow  # lazy

            eval_golden_flow()
    except Exception as exc:
        # Eval failures NEVER break ingestion
        logger.warning("Post-ingestion eval hook failed (non-fatal): %s", exc)

    return result


@flow(name="rag-factory-reset-flow", log_prints=True)
def factory_reset_flow(*, confirm_destructive_wipe: bool = False) -> None:
    """Explicitly wipe vectors and ingestion registries for a clean rebuild."""
    logger = _get_logger()
    logger.warning("Running factory reset flow")
    if not confirm_destructive_wipe:
        raise ValueError(
            "Factory reset is destructive. Set confirm_destructive_wipe=true to proceed."
        )
    reset_vector_backend_task()
    reset_ingestion_registries_task()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ingestion orchestration flows")
    parser.add_argument(
        "flow_name",
        choices=["discovery", "ingestion", "end_to_end", "factory_reset"],
        default="end_to_end",
        nargs="?",
        help="Flow to execute",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Optional worker processing cap",
    )
    parser.add_argument(
        "--register-deployment",
        action="store_true",
        help="Register or update a Prefect deployment for UI-triggered runs",
    )
    parser.add_argument(
        "--register-all-deployments",
        action="store_true",
        help="Register both document ingestion and destructive reset deployments",
    )
    parser.add_argument(
        "--deployment-name",
        default="ingest-documents",
        help="Deployment name when --register-deployment is used",
    )
    parser.add_argument(
        "--work-pool",
        default="rag-work-pool",
        help="Prefect work pool name for deployment registration",
    )
    parser.add_argument(
        "--work-queue",
        default=None,
        help="Optional Prefect work queue for deployment registration",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.register_all_deployments:
        _register_deployment(
            target_flow=end_to_end_flow,
            deployment_name=args.deployment_name,
            work_pool=args.work_pool,
            work_queue=args.work_queue,
            tags=["ingestion", "safe", "oss"],
            description="Discover, process, and index source documents for retrieval.",
        )
        _register_deployment(
            target_flow=factory_reset_flow,
            deployment_name="reset-document-index",
            work_pool=args.work_pool,
            work_queue=args.work_queue,
            tags=["ingestion", "admin", "dangerous", "destructive", "oss"],
            description="Destructively clears indexed documents and ingestion state for a clean rebuild.",
        )
        return

    if args.register_deployment:
        target_flow = _resolve_flow(args.flow_name)
        if args.flow_name == "factory_reset":
            _register_deployment(
                target_flow=target_flow,
                deployment_name=args.deployment_name,
                work_pool=args.work_pool,
                work_queue=args.work_queue,
                tags=["ingestion", "admin", "dangerous", "destructive", "oss"],
                description="Destructively clears indexed documents and ingestion state for a clean rebuild.",
            )
        else:
            _register_deployment(
                target_flow=target_flow,
                deployment_name=args.deployment_name,
                work_pool=args.work_pool,
                work_queue=args.work_queue,
                tags=["ingestion", "safe", "oss"],
                description="Discover, process, and index source documents for retrieval.",
            )
        return

    if args.flow_name == "discovery":
        discovery_flow()
        return

    if args.flow_name == "ingestion":
        ingestion_flow(max_jobs=args.max_jobs)
        return

    if args.flow_name == "factory_reset":
        factory_reset_flow()
        return

    end_to_end_flow(max_jobs=args.max_jobs)


def _resolve_flow(flow_name: str):
    if flow_name == "discovery":
        return discovery_flow
    if flow_name == "ingestion":
        return ingestion_flow
    if flow_name == "factory_reset":
        return factory_reset_flow
    return end_to_end_flow


def _register_deployment(
    *,
    target_flow,
    deployment_name: str,
    work_pool: str,
    work_queue: str | None,
    tags: list[str],
    description: str,
) -> None:
    target_flow.deploy(
        name=deployment_name,
        work_pool_name=work_pool,
        work_queue_name=work_queue,
        build=False,
        push=False,
        tags=tags,
        description=description,
    )


def _get_logger() -> logging.Logger:
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


if __name__ == "__main__":
    main()
