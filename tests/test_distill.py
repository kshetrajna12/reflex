from reflex.distill.cli import _split
from reflex.distill.label import to_training_rows
from reflex.distill.questions import BANK, GENERAL, _extract_json_array, clean, validate
from reflex.prompt import PromptFormat
from reflex.train.data import examples_from_rows


def test_bank_questions_are_valid_requests():
    state = {"text": "hello"}
    for domain, qs in BANK.items():
        assert validate(state, qs).keys() == qs.keys(), domain
    assert validate(state, GENERAL).keys() == GENERAL.keys()


def test_llm_output_is_parsed_and_filtered():
    text = (
        'Sure! ```json\n[{"id": "Needs Reply?", "type": "noul", "instructions": "Does it need a reply?"},'
        ' {"id": "bad", "type": "choice", "instructions": "x", "criteria": {"only": null}},'
        ' {"id": "lvl", "type": "score", "instructions": "how bad", "criteria": ["a", "b", "c"]}]\n```'
    )
    arr = _extract_json_array(text)
    assert len(arr) == 3
    qs = validate({"t": 1}, {f"q{i}": clean(q) for i, q in enumerate(arr)})
    assert set(qs) == {"q0", "q2"}  # the one-option choice is rejected


def test_split_is_stable_and_uniform():
    vals = [_split(f"id{i}", 0) for i in range(2000)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert 0.15 < sum(v < 0.2 for v in vals) / 2000 < 0.25
    assert _split("abc", 0) == _split("abc", 0) != _split("abc", 1)


def test_teacher_rows_become_soft_training_examples():
    row = {
        "id": "x",
        "domain": "review",
        "state": {"review": "meh"},
        "questions": {"stars": BANK["review"]["stars"], "recommend": BANK["review"]["recommend"]},
        "labels": {"stars": {"0": 0.1, "1": 0.2, "2": 0.4, "3": 0.2, "4": 0.1}, "recommend": 0.3},
    }
    rows = list(to_training_rows([row]))
    assert rows[0]["source"] == "distill/review"
    ex = examples_from_rows(rows, PromptFormat(chat=False, no_think=False, system_prompt=None))
    assert len(ex) == 2
    assert abs(ex[0].target.sum() - 1) < 1e-9 and abs(ex[1].target.sum() - 1) < 1e-9
