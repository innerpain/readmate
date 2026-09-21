from src.retrieval.result_fusion import (
    Channel,
    best_dense_score,
    reciprocal_rank_fusion,
    select_with_quota,
)


def _dense(name, ids, scores=None):
    return Channel(name=name, chunk_ids=list(ids), dense_scores=dict(scores or {}))


def test_rrf_lifts_chunks_several_channels_agree_on():
    question = _dense("dense:0", ["a", "b", "c"], {"a": 0.8, "b": 0.7, "c": 0.6})
    sub_query = _dense("dense:1", ["c", "b", "a"], {"a": 0.5, "b": 0.6, "c": 0.65})
    fused = reciprocal_rank_fusion([question, sub_query])
    ids = [hit.chunk_id for hit in fused]
    # 'a' and 'c' are rank 1 in one list and rank 3 in the other, so they tie
    # above 'b' (rank 2 in both); the tie breaks on first appearance.
    assert ids[0] == "a"
    assert ids[2] == "b"
    assert all(len(hit.channels) == 2 for hit in fused)


def test_rrf_prefers_a_chunk_both_channels_rank_highly():
    first = _dense("dense:0", ["x", "y", "z"])
    second = _dense("dense:1", ["y", "z", "x"])
    fused = reciprocal_rank_fusion([first, second])
    assert fused[0].chunk_id == "y"


def test_quota_keeps_one_slot_for_each_query():
    # 'req2' only retrieved 'd2'; RRF alone would drop it behind three chunks
    # that both queries found.
    first = _dense("dense:0", ["a", "b", "c"])
    second = _dense("dense:1", ["a", "b", "c"])
    third = _dense("dense:2", ["d2"])
    fused = reciprocal_rank_fusion([first, second, third])
    selected = select_with_quota(fused, [first, second, third], top_k=4, per_channel_slots=1)
    ids = [hit.chunk_id for hit in selected]
    assert ids[0] == "a"  # first channel's best keeps its slot
    assert "d2" in ids    # the exclusive requirement survives
    assert len(ids) == 4


def test_quota_never_exceeds_top_k():
    channels = [_dense(f"dense:{index}", [f"c{index}"]) for index in range(6)]
    fused = reciprocal_rank_fusion(channels)
    selected = select_with_quota(fused, channels, top_k=3, per_channel_slots=1)
    assert len(selected) == 3


def test_a_channel_without_its_own_slot_never_displaces_a_reserved_slot():
    """D19: only dense channels remain, but the quota contract still holds --
    a channel that does not require a slot can never override one that does."""

    dense = _dense("dense:0", ["a", "b", "c"])
    extra = Channel(name="extra", chunk_ids=["k1"], requires_own_slot=False)
    fused = reciprocal_rank_fusion([dense, extra])
    # With a single slot the guaranteed dense slot wins; the extra channel can
    # only take a slot that quota did not reserve (it must not override it).
    assert [hit.chunk_id for hit in select_with_quota(fused, [dense, extra], top_k=1, per_channel_slots=1)] == ["a"]
    # With room, a hit that also fused in may still enter by fused order.
    assert "k1" in [hit.chunk_id for hit in select_with_quota(fused, [dense, extra], top_k=4)]


def test_best_dense_score_returns_none_for_hits_no_dense_channel_saw():
    dense = _dense("dense:0", ["a"], {"a": 0.44})
    extra = Channel(name="extra", chunk_ids=["b"])
    assert best_dense_score([dense, extra], "a") == 0.44
    assert best_dense_score([dense, extra], "b") is None


def test_rrf_takes_no_weights_argument():
    """D28: ``weights`` had no caller -- every call site passed ``k`` only --
    and RRF's whole point is that the channel scales never need calibrating."""

    import inspect

    assert "weights" not in inspect.signature(reciprocal_rank_fusion).parameters
    fused = reciprocal_rank_fusion([_dense("dense:0", ["a"])], k=10)
    assert fused[0].chunk_id == "a"


def test_empty_channels_fuse_to_nothing():
    assert reciprocal_rank_fusion([_dense("dense:0", [])]) == []
    assert select_with_quota([], [], top_k=5) == []
