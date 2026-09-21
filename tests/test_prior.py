import itertools

import numpy as np
import pytest

from reflex.prior import PositionPrior, PriorFitter
from reflex.prompt import Branch
from reflex.readout import Calibration, merge_branches


def test_round_trip(tmp_path):
    prior = PositionPrior(
        table={"choice:3": [0.5, 0.3, 0.2], "noul:2": [0.6, 0.4]},
        mode="sub",
        strength=0.7,
        model="Qwen/Qwen3.5-4B",
        method="permuted",
        n_fit=400,
    )
    path = tmp_path / "prior.json"
    prior.save(path)
    back = PositionPrior.load(path)
    assert back.to_dict() == prior.to_dict()
    assert np.allclose(back.row("choice", 3), [0.5, 0.3, 0.2])


def test_missing_file_is_an_error_but_none_is_not(tmp_path):
    assert PositionPrior.load(None) is None
    with pytest.raises(FileNotFoundError):
        PositionPrior.load(tmp_path / "nope.json")


def test_uniform_prior_is_a_no_op():
    prior = PositionPrior(table={"choice:4": [0.25] * 4})
    p = np.array([0.5, 0.2, 0.2, 0.1])
    for mode in ("div", "sub"):
        prior.mode = mode
        assert np.allclose(prior.debias("choice", p), p)


def test_zero_strength_and_unfitted_primitive_are_no_ops():
    p = np.array([0.7, 0.3])
    assert np.allclose(PositionPrior({"noul:2": [0.9, 0.1]}, strength=0.0).debias("noul", p), p)
    assert np.allclose(PositionPrior({"noul:2": [0.9, 0.1]}).debias("choice", p), p)


def test_dividing_out_recovers_the_unbiased_distribution():
    """Observed = truth reweighted by a position prior; dividing it out gives truth back."""
    truth = np.array([0.6, 0.3, 0.1])
    bias = np.array([0.5, 0.35, 0.15])
    prior = PositionPrior(table={"choice:3": list(bias)}, mode="div")
    for order in itertools.permutations(range(3)):
        obs = truth[list(order)] * bias
        obs = obs / obs.sum()
        back = prior.debias("choice", obs)
        assert np.allclose(back, truth[list(order)], atol=1e-9)


def test_subtractive_mode_recovers_an_additively_biased_distribution():
    truth = np.array([0.5, 0.3, 0.2])
    bias = np.array([0.5, 0.3, 0.2])  # sums to 1; excess over uniform is bias - 1/3
    obs = truth + (bias - 1 / 3)
    prior = PositionPrior(table={"choice:3": list(bias)}, mode="sub")
    assert np.allclose(prior.debias("choice", obs / obs.sum()), truth, atol=1e-9)


def test_backoff_borrows_the_nearest_option_count():
    prior = PositionPrior(table={"choice:4": [0.4, 0.3, 0.2, 0.1]})
    three = prior.row("choice", 3)
    assert np.allclose(three, np.array([0.4, 0.3, 0.2]) / 0.9)
    five = prior.row("choice", 5)
    assert np.allclose(five, np.array([0.4, 0.3, 0.2, 0.1, 0.1]) / 1.1)
    assert prior.row("choice", 1) is None


def test_fitter_recovers_a_content_free_prior():
    fitter = PriorFitter()
    for _ in range(50):
        fitter.add_content_free("choice", np.array([2.0, 1.0, 1.0]))
    prior = fitter.finish(min_questions=10, method="content-free")
    assert np.allclose(prior.row("choice", 3), [0.5, 0.25, 0.25])
    assert prior.n_fit == 50


def test_fitter_recovers_a_planted_prior_from_permutations():
    truth = np.array([0.55, 0.25, 0.20])
    bias = np.array([0.45, 0.35, 0.20])
    fitter = PriorFitter()
    for _ in range(30):
        branches = []
        for order in itertools.permutations(range(3)):
            obs = truth[list(order)] + (bias - 1 / 3)
            branches.append((list(order), obs / obs.sum()))
        fitter.add_permuted("choice", branches)
    prior = fitter.finish(min_questions=10, method="permuted")
    assert np.allclose(prior.row("choice", 3), bias, atol=1e-9)


def test_thin_cells_are_dropped():
    fitter = PriorFitter()
    fitter.add_content_free("choice", np.array([0.9, 0.1]))
    assert fitter.finish(min_questions=20).table == {}


def test_merge_branches_applies_the_prior():
    """One branch, a prior that favours A: the merged answer moves off A."""
    cal = Calibration(prior=PositionPrior(table={"choice:2": [0.8, 0.2]}, mode="div"))
    br = Branch("q", "choice", "", ["A", "B"], ["x", "y"])
    logits = np.log([0.6, 0.4])
    plain = merge_branches("choice", [(br, logits)], Calibration())
    assert np.isclose(plain["x"], 0.6)
    debiased = merge_branches("choice", [(br, logits)], cal)
    assert np.isclose(debiased["x"], (0.6 / 1.6) / (0.6 / 1.6 + 0.4 / 0.4))
    assert debiased["x"] < plain["x"]


def test_calibration_round_trips_a_prior(tmp_path):
    cal = Calibration(prior=PositionPrior(table={"noul:2": [0.7, 0.3]}))
    path = tmp_path / "calibration.json"
    cal.save(path)
    back = Calibration.load(path)
    assert back.prior is not None
    assert np.allclose(back.prior.row("noul", 2), [0.7, 0.3])
    assert Calibration.load(tmp_path / "absent.json").prior is None
