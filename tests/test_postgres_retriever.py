from rag_core.retrieval.postgres_retriever import PostgresRetriever


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
