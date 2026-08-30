"""Regression tests for the metric fixes — each test pins a specific bug
case; removing the matching fix makes its test fail by construction."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("lighteval")

from src.metrics.metrics import (
    ConservativeExactMatch,
    METEOR,
    RelaxedExtractMatch,
    RelaxedExtractMatchMultiLabel,
)
from src.metrics.utils import clean_model_output
from src.tasks.scio import process_options


def _doc(golds, choices=None):
    return SimpleNamespace(get_golds=lambda: list(golds), choices=list(choices or golds))


def _resp(*texts):
    return SimpleNamespace(final_text=list(texts))


# clean_model_output: marker extraction must precede first-line truncation.


def test_clean_extracts_think_before_first_line():
    assert clean_model_output("<think>\nreasoning\n</think>\nB") == "B"


def test_clean_extracts_assistantfinal_before_first_line():
    assert clean_model_output("assistantfinal\nB") == "B"


def test_clean_qwen_dangling_close_think():
    assert clean_model_output("</think>\nAnswer: B") == "Answer: B"


# RelaxedExtractMatchMultiLabel: lookaround-based label parser.


def test_multilabel_comma_separated_full_set():
    assert RelaxedExtractMatchMultiLabel()._parse_option_labels("A,B,C,D") == {"A", "B", "C", "D"}


def test_multilabel_slovak_prefix_stripped():
    assert RelaxedExtractMatchMultiLabel()._parse_option_labels("Odpoveď: A, B") == {"A", "B"}


def test_multilabel_does_not_match_lowercase_word():
    assert RelaxedExtractMatchMultiLabel()._parse_option_labels("Bratislava") == set()


def test_multilabel_accepts_semicolons():
    assert RelaxedExtractMatchMultiLabel()._parse_option_labels("A;B;C") == {"A", "B", "C"}


def test_multilabel_reads_multiline_body():
    parser = RelaxedExtractMatchMultiLabel()
    assert parser._parse_option_labels("Reason: both look plausible\nOdpoveď: A, B") == {"A", "B"}


# ConservativeExactMatch.


def test_conservative_norm_applies_normalize_to_gold():
    metric = ConservativeExactMatch(
        strip_strings=True,
        normalize_gold=lambda s: s.lower(),
        normalize_pred=lambda s: s.lower(),
    )
    assert metric.compute(_doc(golds=["Áno"], choices=["Áno"]), _resp("áno")) == 1.0


def test_conservative_longest_choice_wins_over_substring():
    metric = ConservativeExactMatch()
    doc = _doc(golds=["New York"], choices=["New York", "York"])
    assert metric.compute(doc, _resp("The answer is New York")) == 1.0


def test_conservative_does_not_short_circuit_on_first_pred():
    metric = ConservativeExactMatch()
    doc = _doc(golds=["B"], choices=["A", "B"])
    assert metric.compute(doc, _resp("A and B both", "B")) == 1.0


def test_conservative_matches_choices_with_punctuation():
    metric = ConservativeExactMatch()
    assert metric.compute(_doc(golds=["25%"], choices=["25%", "50%"]), _resp("25%")) == 1.0
    assert metric.compute(_doc(golds=["C++"], choices=["C++", "C"]), _resp("C++")) == 1.0


def test_conservative_overlapping_choices_are_ambiguous():
    metric = ConservativeExactMatch()
    doc = _doc(golds=["ab cd"], choices=["ab cd", "cd ef"])
    assert metric.compute(doc, _resp("ab cd ef")) == 0.0


def test_conservative_reads_multiline_body():
    metric = ConservativeExactMatch()
    doc = _doc(golds=["A"], choices=["A", "B"])
    assert metric.compute(doc, _resp("Reasoning: choose the listed answer\nAnswer: A")) == 1.0


# RelaxedExtractMatch float return + fallback marker extraction.


def test_relaxed_returns_float_not_truncated_int():
    metric = RelaxedExtractMatch(
        type_exact_match="full",
        aggregation_function=lambda xs: sum(xs) / len(xs),
    )
    score = metric.compute(_doc(golds=["B"]), _resp("B", "A"))
    assert score == pytest.approx(0.5)
    assert isinstance(score, float)


def test_fallback_prefers_answer_marker_over_first_colon():
    metric = RelaxedExtractMatch(type_exact_match="full", fallback_mode="first_match")
    assert metric.compute(_doc(golds=["B"]), _resp("Reason: not B\nAnswer: B")) == 1.0


def test_fallback_marker_requires_explicit_delimiter():
    metric = RelaxedExtractMatch(type_exact_match="full", fallback_mode="first_match")
    assert metric.compute(_doc(golds=["B"]), _resp("Answer the question carefully")) == 0.0


def test_fallback_marker_word_boundary_on_keyword():
    # "answer-bar" must not trip the "answer" keyword.
    metric = RelaxedExtractMatch(type_exact_match="full", fallback_mode="first_match")
    assert metric.compute(_doc(golds=["bar baz"]), _resp("foo answer-bar baz")) == 0.0


# METEOR best-of-references per prediction.


def test_meteor_best_over_references():
    import nltk

    for resource, path in [("wordnet", "corpora/wordnet"), ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pytest.skip(f"nltk {resource} unavailable")

    metric = METEOR()
    one_ref = metric.compute(_doc(golds=["Bratislava"]), _resp("Bratislava"))
    two_refs = metric.compute(_doc(golds=["Bratislava", "Prešporok"]), _resp("Bratislava"))
    assert two_refs == pytest.approx(one_ref)


# scio.process_options: only strip a recognizable label prefix.


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("(A) Foo", "Foo"),
        ("A) Foo", "Foo"),
        ("(a) Foo", "Foo"),
        ("A. Foo", "Foo"),
        ("A: Foo", "Foo"),
        ("(A)Foo", "Foo"),
        # Unprefixed options pass through verbatim.
        ("Tree", "Tree"),
        ("Apple", "Apple"),
        ("American Civil War", "American Civil War"),
        # Hyphenated content is not a label.
        ("L-arginine", "L-arginine"),
        ("A-grade", "A-grade"),
        ("I-95", "I-95"),
    ],
)
def test_scio_process_options(raw, expected):
    assert process_options([raw]) == [expected]


# ENABLE_THINKING_MODE is resolved at import; we reload to exercise truthy parsing.


def test_thinking_mode_env_var_resolution(monkeypatch):
    import importlib
    import src.metrics.utils as utils_mod

    for truthy in ("1", "true", "YES", "On"):
        monkeypatch.setenv("ENABLE_THINKING_MODE", truthy)
        assert importlib.reload(utils_mod).ENABLE_THINKING_MODE is True, truthy

    monkeypatch.delenv("ENABLE_THINKING_MODE", raising=False)
    importlib.reload(utils_mod)
