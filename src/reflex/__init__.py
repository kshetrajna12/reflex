"""reflex: a System One decision model built on a pretrained causal LM.

Give it a block of *state* and a set of typed *questions*; it returns calibrated
probability distributions over the answers you supplied, never free text.

    from reflex import Engine
    engine = Engine.load("Qwen/Qwen3-8B")
    resp = engine.answer(SystemOneRequest(state=..., questions={...}))
"""

from reflex.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

__all__ = [
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Engine",
    "NoulAnswer",
    "NoulQuestion",
    "ScoreAnswer",
    "ScoreQuestion",
    "SystemOneRequest",
    "SystemOneResponse",
]


def __getattr__(name):  # lazy: importing torch is slow and optional for schema users
    if name == "Engine":
        from reflex.engine import Engine

        return Engine
    raise AttributeError(name)
