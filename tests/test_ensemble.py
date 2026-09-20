import numpy as np

from reflex.ensemble import disagreement, load_ensemble, merged_probs_for_keys, variant_formats
from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.readout import Calibration
from reflex.schema import ChoiceQuestion, NoulQuestion


def test_variants_share_the_state_prefix_and_differ_after_it():
    base = PromptFormat()
    fmts = variant_formats(base, load_ensemble("prompts/ensemble-v1.json"))
    assert len(fmts) == 4
    state = {"ticket": "hello"}
    assert len({f.prefix(state) for f in fmts}) == 1
    q = ChoiceQuestion(type="choice", instructions="Which?", criteria={"a": None, "b": None})
    texts = {build_branches("q", q, f, 1)[0].text for f in fmts}
    assert len(texts) >= 3


def test_noul_readout_variants_read_different_labels():
    base = PromptFormat()
    fmts = variant_formats(base, [{}, {"noul_readout": "yesno"}])
    q = NoulQuestion(type="noul", instructions="Is it?")
    a, b = (build_branches("q", q, f, 1)[0] for f in fmts)
    assert a.labels == ["A", "B"] and b.labels != ["A", "B"]
    assert a.keys == b.keys == [True, False]


def test_prefix_changing_variant_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        load_ensemble([{"state_heading": "# Facts"}])


def test_merge_and_disagreement():
    cal = Calibration()
    b1 = Branch("q", "choice", "", ["A", "B"], ["x", "y"])
    b2 = Branch("q", "choice", "", ["A", "B"], ["y", "x"])  # permuted order
    agree = [(b1, np.array([3.0, 0.0])), (b2, np.array([0.0, 3.0]))]  # both say x
    p = merged_probs_for_keys("choice", agree, cal, ["x", "y"])
    assert p[0] > 0.9 and abs(p.sum() - 1) < 1e-9
    assert disagreement(agree, cal, "choice") < 1e-6
    split = [(b1, np.array([3.0, 0.0])), (b2, np.array([3.0, 0.0]))]  # x then y
    p = merged_probs_for_keys("choice", split, cal, ["x", "y"])
    assert abs(p[0] - 0.5) < 1e-6
    assert disagreement(split, cal, "choice") > 0.4
