"""Probability maps retain the model's normalized distribution at wide widths."""

import math

import pytest

from reflex.readout import to_answer
from reflex.schema import ScoreQuestion


def test_wide_choice_probability_map_sums_to_one():
    probabilities = {f"option_{i}": 1 / 255 for i in range(255)}
    answer = to_answer("choice", probabilities)

    assert set(answer.probabilities) == set(probabilities)
    assert math.isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-12)
    assert all(value > 0 for value in answer.probabilities.values())


def test_choice_winner_uses_unrounded_probabilities():
    answer = to_answer("choice", {"first": 0.4999999, "second": 0.5000001})

    assert answer.choice == "second"
    assert answer.probabilities["second"] > answer.probabilities["first"]


def test_score_probability_map_sums_to_one():
    question = ScoreQuestion(
        type="score", instructions="Rate this synthetic state", criteria=[str(i) for i in range(7)]
    )
    answer = to_answer("score", {i: 1 / 7 for i in range(7)}, question)

    assert set(answer.probabilities) == {str(i) for i in range(7)}
    assert math.isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-12)
    assert answer.score == 3


@pytest.mark.parametrize(
    "probabilities",
    [
        {"first": math.nan, "second": 1.0},
        {"first": math.inf, "second": 0.0},
        {"first": -0.1, "second": 1.1},
        {"first": 0.4, "second": 0.4},
    ],
)
def test_invalid_probability_maps_are_rejected(probabilities):
    with pytest.raises(ValueError, match="answer probabilities"):
        to_answer("choice", probabilities)
