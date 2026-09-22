"""Request / response contract.

Mirrors the shape of TypeSafe's `POST /v1/systemone` so client code written against
Jev works unchanged against a local reflex server.

Three primitives:
  * noul   – "is this true?"            -> P(yes)
  * choice – "which of these options?"  -> distribution over option keys
  * score  – "which level?"             -> distribution over ordered levels + expected value
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# `instructions` and criteria descriptions may be a string, or structured JSON that we
# render for the model (e.g. {"what": ..., "examples": [...]}).
Text = str | dict[str, Any] | list[Any]

# Letters label at most 26 options per branch, so a bigger choice is shown as several
# pages of 26 and read as one distribution (see reflex.prompt). 256 is the ceiling.
MAX_CHOICE_OPTIONS = 256
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")
    true: Text | None = None
    false: Text | None = None


class NoulQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    instructions: Text
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    instructions: Text
    criteria: dict[str, Text | None]

    @field_validator("criteria")
    @classmethod
    def _check_options(cls, v: dict[str, Text | None]) -> dict[str, Text | None]:
        if len(v) < 2:
            raise ValueError("a choice needs at least two options")
        if len(v) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"at most {MAX_CHOICE_OPTIONS} options per choice")
        for k in v:
            if not k or not k.strip():
                raise ValueError("option keys must be non-empty")
        return v


class ScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["score"]
    instructions: Text
    criteria: list[Text]  # ordered low -> high

    @field_validator("criteria")
    @classmethod
    def _check_levels(cls, v: list[Text]) -> list[Text]:
        if not (MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS):
            raise ValueError(f"a score takes {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        return v


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "reflex-latest"
    state: Text
    questions: dict[str, Question]
    # reflex extensions (ignored by the real API)
    permutations: int | None = Field(
        default=None,
        ge=1,
        le=8,
        description="Average choice/score readout over N shuffled option orders to reduce "
        "position bias. Costs N branches per question. Default: the server's setting.",
    )

    @model_validator(mode="after")
    def _non_empty(self) -> SystemOneRequest:
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0  # there is no generation; kept for API compatibility
    state_tokens: int = 0
    question_tokens: int = 0
    state_cache_hit: bool = False
    images: int = 0


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
