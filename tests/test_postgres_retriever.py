from datetime import UTC, datetime
from types import SimpleNamespace

from rag_core.retrieval.postgres_retriever import PostgresRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys


def test_weighted_rrf_missing_branch_scores_default_to_zero() -> None:
    dense_ranks = {1: 1, 2: 2}
    fts_ranks = {2: 1, 3: 2}

    fused = PostgresRetriever._fuse_rank_maps(
        dense_ranks=dense_ranks,
        fts_ranks=fts_ranks,
        dense_weight=0.5,
        text_weight=0.5,
        rrf_k=2,
    )

    assert fused[1] == 0.5 * (1.0 / 3.0)
    assert fused[3] == 0.5 * (1.0 / 4.0)
    assert fused[2] == (0.5 * (1.0 / 4.0)) + (0.5 * (1.0 / 3.0))


def test_weighted_rrf_prefers_top_single_when_k_is_small() -> None:
    fused = PostgresRetriever._fuse_rank_maps(
        dense_ranks={10: 1},
        fts_ranks={20: 15, 10: 200},
        dense_weight=0.5,
        text_weight=0.5,
        rrf_k=2,
    )

    assert fused[10] > fused[20]


def test_section_to_document_includes_asset_name_when_available() -> None:
    section = SimpleNamespace(
        id=13,
        asset_key="source_documents_report_ef7a4579.pdf",
        source_uri="/tmp/report.pdf",
        page_numbers=[2],
        text="section text",
        created_at=datetime(2026, 5, 15, 4, 41, 10, tzinfo=UTC),
        asset=SimpleNamespace(asset_name="report.pdf"),
    )

    doc = PostgresRetriever._section_to_document(section=section, score=0.42)

    assert doc.metadata[PayloadKeys.ASSET_KEY.value] == "source_documents_report_ef7a4579.pdf"
    assert doc.metadata[PayloadKeys.ASSET_NAME.value] == "report.pdf"
    assert doc.metadata[PayloadKeys.SOURCE.value] == "/tmp/report.pdf"
    assert doc.metadata[PayloadKeys.PAGE.value] == [2]
    assert doc.metadata["score"] == 0.42
    assert "created_at" in doc.metadata


def test_section_to_document_defaults_asset_name_to_empty_when_missing() -> None:
    section = SimpleNamespace(
        id=1,
        asset_key="asset-1",
        source_uri="/tmp/a.pdf",
        page_numbers=[1],
        text="text",
        created_at=None,
        asset=None,
    )

    doc = PostgresRetriever._section_to_document(section=section)

    assert doc.metadata[PayloadKeys.ASSET_NAME.value] == ""
