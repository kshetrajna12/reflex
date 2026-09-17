from reflex.engine import CACHED, Segment, build_pack


def test_mask_isolates_branches():
    segs = [Segment([1, 2, 3], -1), Segment([4, 5], 0), Segment([6, 7, 8], 0)]
    pack = build_pack(segs)
    m = pack.attention_mask[0, 0]
    assert m.shape == (8, 8)
    # state is causal within itself
    assert m[0, 0] and not m[0, 1] and m[2, 0]
    # branch 1 sees state + itself causally, never branch 2
    assert m[3, :3].all() and m[3, 3] and not m[3, 4] and not m[3, 5:].any()
    assert m[4, 3] and m[4, 4]
    # branch 2 never sees branch 1
    assert not m[5, 3:5].any() and m[5, :3].all() and m[5, 5]
    # positions restart after the state
    assert pack.position_ids[0].tolist() == [0, 1, 2, 3, 4, 3, 4, 5]
    assert pack.last_index.tolist() == [2, 4, 7]


def test_mask_with_cached_state():
    segs = [Segment([4, 5], CACHED), Segment([6], CACHED)]
    pack = build_pack(segs, past_len=3, past_seg=CACHED)
    m = pack.attention_mask[0, 0]
    assert m.shape == (3, 6)
    assert m[0, :3].all() and m[0, 3] and not m[0, 4] and not m[0, 5]
    assert m[2, :3].all() and not m[2, 3:5].any() and m[2, 5]
    assert pack.position_ids[0].tolist() == [3, 4, 3]
