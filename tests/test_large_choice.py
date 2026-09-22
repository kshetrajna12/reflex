"""A choice bigger than the alphabet: pages in, one distribution out."""

import numpy as np
import pytest
from pydantic import ValidationError

from reflex.prompt import PAGE_SIZE, Branch, PromptFormat, build_branches
from reflex.readout import Calibration, merge_branches, to_answer
from reflex.schema import ChoiceQuestion


def choice(n: int) -> ChoiceQuestion:
    return ChoiceQuestion(
        type="choice", instructions="which one?", criteria={f"opt{i}": None for i in range(n)}
    )


def test_small_choice_is_one_page():
    """2..26 options: one branch, lettered A.., and no mention of pages anywhere."""
    for n in (2, 5, PAGE_SIZE):
        (br,) = build_branches("q", choice(n), PromptFormat(), 1)
        assert br.n_pages == 1
        assert br.labels == [chr(ord("A") + i) for i in range(n)]
        assert br.keys == [f"opt{i}" for i in range(n)]
        assert "shown separately" not in br.text


def test_sixty_options_make_three_pages():
    brs = build_branches("q", choice(60), PromptFormat(), 1)
    assert [len(b.keys) for b in brs] == [26, 26, 8]
    assert {b.group for b in brs} == {brs[0].group} and all(b.n_pages == 3 for b in brs)
    # every page relabels from A, and the union is the whole question, in order
    assert all(b.labels[0] == "A" for b in brs)
    assert [k for b in brs for k in b.keys] == [f"opt{i}" for i in range(60)]
    assert "Options 1-26 of 60" in brs[0].text
    assert "Options 53-60 of 60" in brs[2].text


def test_pages_union_to_one_distribution():
    brs = build_branches("q", choice(60), PromptFormat(), 1)
    rng = np.random.default_rng(0)
    results = [(b, rng.normal(size=len(b.keys))) for b in brs]
    probs = merge_branches("choice", results, Calibration())
    assert len(probs) == 60
    assert sum(probs.values()) == pytest.approx(1.0)


def test_a_favourite_on_the_last_page_wins():
    """Pages are pooled before the softmax, so a page carrying only also-rans does not
    get handed a third of the mass."""
    brs = build_branches("q", choice(60), PromptFormat(), 1)
    results = []
    for b in brs:
        row = np.full(len(b.keys), -1.0)
        if "opt55" in b.keys:
            row[b.keys.index("opt55")] = 9.0
        results.append((b, row))
    ans = to_answer("choice", merge_branches("choice", results, Calibration()))
    assert ans.choice == "opt55"
    assert ans.probabilities["opt55"] > 0.9


def test_permutations_page_differently_and_round_trip():
    brs = build_branches("q", choice(60), PromptFormat(), 2)
    groups: dict[int, list[Branch]] = {}
    for b in brs:
        groups.setdefault(b.group, []).append(b)
    assert len(groups) == 2
    orders = []
    for pages in groups.values():
        keys = [k for p in pages for k in p.keys]
        assert sorted(keys) == sorted(f"opt{i}" for i in range(60))  # nothing lost or doubled
        orders.append(keys)
    assert orders[0] != orders[1]  # the second order is a distinct shuffle
    # the cut is taken after the shuffle, so the second order's first page is not the first's
    assert orders[0][:26] != orders[1][:26]


def test_merge_keeps_both_orders_of_a_paged_choice():
    brs = build_branches("q", choice(60), PromptFormat(), 2)
    results = [(b, np.zeros(len(b.keys))) for b in brs]
    probs = merge_branches("choice", results, Calibration())
    assert len(probs) == 60
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1 / 60) for v in probs.values())


def test_compact_style_pages_too():
    brs = build_branches("q", choice(60), PromptFormat(style="compact"), 1)
    assert len(brs) == 3
    assert '"note"' in brs[0].text and "Options 1-26 of 60" in brs[0].text
    assert '"note"' not in build_branches("q", choice(4), PromptFormat(style="compact"), 1)[0].text


def test_schema_ceiling():
    assert len(choice(256).criteria) == 256
    with pytest.raises(ValidationError, match="at most 256 options per choice"):
        choice(257)


def test_restrict_normalises_a_page_but_not_a_lone_branch():
    """`Engine.restrict` hands pooled pages full-vocabulary log-probabilities, so no page
    can look preferred just because its prompt left more mass on the alphabet."""
    import torch

    from reflex.engine import Engine

    class Fake:
        def label_id(self, label):
            return ord(label) - ord("A")

    row = torch.tensor([3.0, 1.0, 0.0, -5.0, 7.0])
    lone = Branch("q", "choice", "", ["A", "B"], ["x", "y"], n_pages=1)
    paged = Branch("q", "choice", "", ["A", "B"], ["x", "y"], n_pages=2)
    assert Engine.restrict(Fake(), row, lone).tolist() == [3.0, 1.0]
    got = Engine.restrict(Fake(), row, paged)
    expect = row.log_softmax(-1)[:2]
    assert torch.allclose(got, expect)
    # the shift is common to the page, so the page's own distribution is untouched
    assert torch.allclose(got.softmax(-1), torch.tensor([3.0, 1.0]).softmax(-1))
