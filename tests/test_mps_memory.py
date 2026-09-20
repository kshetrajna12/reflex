"""MPS memory defaults must stay valid next to whatever the user already set. Runs on CPU."""

import pytest

from reflex.mps import HIGH_WATERMARK, LOW_WATERMARK, memory_watermarks


def test_defaults_when_nothing_is_set():
    assert memory_watermarks({}) == {HIGH_WATERMARK: "0.7", LOW_WATERMARK: "0.6"}


def test_a_user_low_above_our_high_raises_the_high_default():
    # torch rejects low=1.0 with high=0.7 ("invalid low watermark ratio 1")
    assert memory_watermarks({LOW_WATERMARK: "1.0"}) == {HIGH_WATERMARK: "1"}
    assert memory_watermarks({LOW_WATERMARK: "0.3"}) == {HIGH_WATERMARK: "0.7"}


def test_a_user_high_below_our_low_lowers_the_low_default():
    assert memory_watermarks({HIGH_WATERMARK: "0.5"}) == {LOW_WATERMARK: "0.5"}
    assert memory_watermarks({HIGH_WATERMARK: "1.4"}) == {LOW_WATERMARK: "0.6"}


def test_no_limit_is_respected():
    assert memory_watermarks({HIGH_WATERMARK: "0.0"}) == {LOW_WATERMARK: "0.6"}
    assert memory_watermarks({HIGH_WATERMARK: "0", LOW_WATERMARK: "1.4"}) == {}


def test_a_valid_user_pair_is_left_alone():
    assert memory_watermarks({HIGH_WATERMARK: "1.0", LOW_WATERMARK: "1.0"}) == {}


@pytest.mark.parametrize(
    "env",
    [
        {HIGH_WATERMARK: "0.5", LOW_WATERMARK: "0.6"},  # low above high
        {HIGH_WATERMARK: "2.5"},  # torch caps high at 2
        {LOW_WATERMARK: "2.5"},  # no valid high exists for it
        {LOW_WATERMARK: "-0.1"},
        {HIGH_WATERMARK: "lots"},
    ],
)
def test_an_invalid_pair_fails_with_a_clear_message(env):
    with pytest.raises(ValueError, match="PYTORCH_MPS"):
        memory_watermarks(env)
