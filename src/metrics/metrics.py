import re
from typing import Callable, Literal

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation, ROUGE
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.models.model_output import ModelResponse
from lighteval.metrics.utils.metric_utils import CorpusLevelMetric
from lighteval.tasks.requests import Doc
from lighteval.metrics.metrics_corpus import CorpusLevelTranslationMetric
from lighteval.metrics.sample_preparator import GenerativeCorpusMetricInput, Preparator
from lighteval.utils.utils import as_list
from src.metrics.utils import clean_model_output


class RelaxedExtractMatch(SampleLevelComputation):
    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
        fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
        type_exact_match: Literal["full", "relaxed", "prefix", "suffix"] = "full",
    ):
        self.aggregation_function = aggregation_function
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        self.strip_strings = strip_strings
        self.type_exact_match = type_exact_match
        self.fallback_mode = fallback_mode

    def compute(self, doc: Doc, model_response: ModelResponse) -> int:
        golds = doc.get_golds()

        results = []
        for gold in golds:
            for pred in model_response.final_text:
                results.append(self.compute_one_item(gold=gold, pred=pred))

        # Handle empty results case
        if not results:
            return 0

        final_score = self.aggregation_function(results)

        if (
            final_score is None or final_score == 0
        ):  # if the main method fails, we can try to extract the answer using the fallback method
            return self.fallback_compute(golds=golds, model_response=model_response)

        return int(final_score) if final_score is not None else 0

    # Word boundary in front of the keyword prevents matches inside
    # "answerable"/"answer-bar"; the literal `:` (not `[:\-]?`) prevents
    # matches in prose like "Answer the question". `assistantfinal` is
    # already handled by strip_wrappers, so it's not in this list.
    _ANSWER_MARKER = re.compile(
        r"(?<![A-Za-z])(?:answer|odpoveď|odpoved|riešenie|riesenie|final answer)\s*:\s*(.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    _GENERIC_MARKER = re.compile(r":\s*([^:\n]+?)\s*$", re.MULTILINE)

    def fallback_compute(self, golds: list[str], model_response: ModelResponse) -> int:
        if self.fallback_mode != "first_match":
            return 0

        fallback_results = []
        for text in model_response.final_text:
            # strip_wrappers (not clean_model_output) preserves multi-line
            # bodies so the marker search can find an answer on a later line.
            cleaned = clean_model_output(text)
            marker_matches = list(self._ANSWER_MARKER.finditer(cleaned))
            if marker_matches:
                extracted = marker_matches[-1].group(1).strip()
            else:
                generic_matches = list(self._GENERIC_MARKER.finditer(cleaned))
                extracted = generic_matches[-1].group(1).strip() if generic_matches else None
            if extracted:
                for gold in golds:
                    fallback_results.append(self.compute_one_item(gold=gold, pred=extracted))

        if not fallback_results:
            return 0
        return int(self.aggregation_function(fallback_results))

    def _parse_option_labels(self, text: str) -> set[str]:
        """Extract option labels (e.g., A, B, C, D) from prediction text.

        Handles formats like:
        - "A B C"
        - "A, B, C"
        - "A. B. C."
        - "(A) (B) (C)"
        - "Options: A, B, C"
        - "A;B;C"
        """
        # strip_wrappers preserves multi-line bodies so e.g.
        # "Reason: ...\nOdpoveď: A, B" still exposes the label line.
        text = clean_model_output(text)
        text = re.sub(
            r"(?i)(options?|answers?|choices?|odpoveď|odpovede|možnosť|možnosti):\s*",
            "",
            text,
        )
        # Lookbehind/lookahead (not consume) so adjacent labels in compact
        # lists like "A,B,C,D" don't share their separator.
        options = re.findall(r"(?:^|(?<=[\s,.;:()/\-]))([A-Z])(?=[\s,.;:()/\-]|$)", text)
        return set(options)

    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:
        if not pred:
            return 0

        pred = clean_model_output(pred)

        if self.strip_strings:
            gold = gold.strip()
            pred = pred.strip()

        if self.normalize_gold:
            gold = self.normalize_gold(gold)
        if self.normalize_pred:
            pred = self.normalize_pred(pred)

        if self.type_exact_match == "full":
            return 1 if gold == pred else 0
        elif self.type_exact_match == "relaxed":
            # Word-boundary match: gold must appear as a complete word in pred
            pattern = r"\b" + re.escape(gold) + r"\b"
            return 1 if re.search(pattern, pred) else 0
        elif self.type_exact_match == "prefix":
            return 1 if pred.startswith(gold) else 0
        elif self.type_exact_match == "suffix":
            return 1 if pred.endswith(gold) else 0
        else:
            raise ValueError(f"Invalid type_exact_match: {self.type_exact_match}")


class RelaxedExtractMatchMultiLabel(RelaxedExtractMatch):
    def __init__(self, aggregation_function: Callable[[list[float]], float] = max):
        """Multi-label exact match that compares sets of extracted option labels.

        Only `aggregation_function` is used; compute() does not apply normalization,
        type matching, or string stripping from parent class — it always does exact
        set matching on parsed option labels (A, B, C, etc.).

        Args:
            aggregation_function: How to combine per-prediction scores. Defaults to max.
        """
        self.aggregation_function = aggregation_function

    def compute(self, doc: Doc, model_response: ModelResponse) -> int:
        golds = doc.get_golds()
        pred_scores = []

        for pred in model_response.final_text:
            pred_set = self._parse_option_labels(pred)
            gold_set = set(golds)
            # Exact set match: all golds present AND no extras.
            pred_scores.append(1.0 if pred_set == gold_set else 0.0)

        if not pred_scores:
            return 0

        return int(self.aggregation_function(pred_scores))


class METEOR(SampleLevelComputation):
    def __init__(
        self,
        normalize_gold: Callable | None = None,
        normalize_pred: Callable | None = None,
        aggregation_function: Callable | None = None,
    ):
        self.normalize_gold = normalize_gold
        self.normalize_pred = normalize_pred
        if aggregation_function is None:
            self.aggregation_function = np.mean
        else:
            self.aggregation_function = aggregation_function

    def compute(self, doc: Doc, model_response: ModelResponse) -> float:
        from nltk import word_tokenize
        from nltk.translate import meteor_score

        golds = doc.get_golds()
        preds = model_response.final_text

        preds = [clean_model_output(pred) for pred in preds]

        if self.normalize_gold:
            golds = [self.normalize_gold(gold) for gold in golds]
        if self.normalize_pred:
            preds = [self.normalize_pred(pred) for pred in preds]

        # Best-of-references per pred, then aggregate over preds. A flat
        # average over the cross-product would dilute a correct match
        # against one of N references.
        per_pred_scores = []
        for pred in preds:
            pred_tokens = word_tokenize(pred)
            per_gold = [
                meteor_score.single_meteor_score(word_tokenize(gold), pred_tokens)
                for gold in golds
            ]
            if per_gold:
                per_pred_scores.append(max(per_gold))

        if not per_pred_scores:
            return 0.0

        return float(self.aggregation_function(per_pred_scores))


class ConservativeExactMatch(RelaxedExtractMatch):
    """Exact match that scores 0 when more than one choice appears in the
    prediction. Per-prediction scores are combined via `aggregation_function`
    (defaults to `max`)."""

    def _normalize(self, value: str) -> str:
        if self.strip_strings:
            value = value.strip()
        if self.normalize_gold:
            value = self.normalize_gold(value)
        return value

    def compute(self, doc: Doc, model_response: ModelResponse) -> int:
        # Normalize golds the same way choices are normalized, otherwise a
        # config like `normalize_gold=lower` lowercases the choice but
        # compares against an un-normalized gold ("áno" vs "Áno").
        gold_set = {self._normalize(str(g)) for g in doc.get_golds()}

        # Dedup choices while preserving original order (deterministic across
        # PYTHONHASHSEEDs); sort longest-first so e.g. "New York" claims its
        # span before "York" can match inside it.
        seen: set[str] = set()
        normalized_choices: list[str] = []
        for choice in doc.choices:
            c = self._normalize(str(choice))
            if c and c not in seen:
                seen.add(c)
                normalized_choices.append(c)
        ordered_choices = sorted(
            enumerate(normalized_choices), key=lambda iv: (-len(iv[1]), iv[0])
        )

        pred_scores: list[float] = []
        for pred in model_response.final_text:
            pred_cleaned = clean_model_output(pred)
            if self.strip_strings:
                pred_cleaned = pred_cleaned.strip()
            if self.normalize_pred:
                pred_cleaned = self.normalize_pred(pred_cleaned)

            # Lookarounds (not `\b`) so choices that begin/end with non-word
            # characters like "25%" or "C++" can still match themselves.
            matches: list[tuple[int, int, str]] = []
            for _, choice_str in ordered_choices:
                pattern = r"(?<!\w)" + re.escape(choice_str) + r"(?!\w)"
                for m in re.finditer(pattern, pred_cleaned):
                    matches.append((m.start(), m.end(), choice_str))

            # Drop a match only when its span is strictly contained inside a
            # longer match. Distinct overlapping spans (e.g. "ab cd" and
            # "cd ef" inside "ab cd ef") remain as separate matches.
            kept = [
                c
                for s, e, c in matches
                if not any(
                    s2 <= s and e <= e2 and (e2 - s2 > e - s) for s2, e2, _ in matches
                )
            ]
            matching_choices = list(dict.fromkeys(kept))

            if len(matching_choices) > 1:
                pred_scores.append(0)
            elif len(matching_choices) == 1 and matching_choices[0] in gold_set:
                pred_scores.append(1)
            else:
                pred_scores.append(0)

        if not pred_scores:
            return 0
        return int(self.aggregation_function(pred_scores))
    

class GenerativePreparator(Preparator):
    @staticmethod
    def prepare(doc: Doc, model_response: ModelResponse, **kwargs):
        """Prepares an individual generative example to the format expected by metrics computed at the corpus level (aggregated).

        Args:
            doc (Doc): The document containing gold references.
            model_response (ModelResponse): The model's response containing predictions.
            **kwargs: Additional keyword arguments.

        Returns:
            GenerativeCorpusMetricInput: Stores the golds and predictions as such
        """
        golds = as_list(doc.get_golds())
        predictions = [clean_model_output(pred) for pred in model_response.final_text]
        return GenerativeCorpusMetricInput(golds=golds, preds=predictions)

    def __str__(self):
        attrs = vars(self)
        attr_strs = []
        for k, v in attrs.items():
            if callable(v):
                val_str = v.__name__
            else:
                val_str = str(v)
            attr_strs.append(f"{k}={val_str}")
        return f"{self.__class__.__name__}({', '.join(attr_strs)})"


normalized_exact_match = SampleLevelMetric(
    metric_name="em_norm",
    sample_level_fn=RelaxedExtractMatch(
        strip_strings=True,
        normalize_gold=lambda x: x.lower(),
        normalize_pred=lambda x: x.lower(),
        type_exact_match="full",
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


relaxed_normalized_exact_match = SampleLevelMetric(
    metric_name="em_norm_relaxed",
    sample_level_fn=RelaxedExtractMatch(
        strip_strings=True,
        normalize_gold=lambda x: x.lower(),
        normalize_pred=lambda x: x.lower(),
        type_exact_match="relaxed",
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


relaxed_exact_match = SampleLevelMetric(
    metric_name="em_relaxed",
    sample_level_fn=RelaxedExtractMatch(type_exact_match="relaxed"),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


em_relaxed_multi_label = SampleLevelMetric(
    metric_name="em_relaxed_multi_label",
    sample_level_fn=RelaxedExtractMatchMultiLabel(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


meteor = SampleLevelMetric(
    metric_name="meteor",
    sample_level_fn=METEOR(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

conservative_exact_match = SampleLevelMetric(
    metric_name="em_conservative",
    sample_level_fn=ConservativeExactMatch(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

conservative_norm_exact_match = SampleLevelMetric(
    metric_name="em_conservative_norm",
    sample_level_fn=ConservativeExactMatch(
        strip_strings=True,
        normalize_gold=lambda x: x.lower(),
        normalize_pred=lambda x: x.lower(),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

rouge1 = SampleLevelMetric(
    metric_name="rouge1",
    sample_level_fn=ROUGE(
        "rouge1",
        normalize_pred=lambda x: clean_model_output(x),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
) 

rouge2 = SampleLevelMetric(
    metric_name="rouge2",
    sample_level_fn=ROUGE(
        "rouge2",
        normalize_pred=lambda x: clean_model_output(x),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

rougeL = SampleLevelMetric(
    metric_name="rougeL",
    sample_level_fn=ROUGE(
        "rougeL",
        normalize_pred=lambda x: clean_model_output(x),
    ),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

chrf = CorpusLevelMetric(
    metric_name="chrf",
    sample_level_fn=GenerativePreparator(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=CorpusLevelTranslationMetric("chrf"),
    higher_is_better=True,
)

chrf_plus = CorpusLevelMetric(
    metric_name="chrf++",
    sample_level_fn=GenerativePreparator(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=CorpusLevelTranslationMetric("chrf++"),
    higher_is_better=True,
)