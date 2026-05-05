from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository

def reconcile_source_assets(db_manager, source_system: str, run_id: int) -> int:
    """Handles the transactional logic for soft-deleting absent assets."""
    with db_manager.get_session() as session:
        with session.begin():
            repo = SourceAssetRepository(session)
            affected_count = repo.mark_deleted_absent_assets(
                source_system=source_system,
                current_run_id=run_id
            )
            return affected_count
