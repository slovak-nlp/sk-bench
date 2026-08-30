from typing import Callable, Literal

import numpy as np

from lighteval.metrics.metrics_sample import ExactMatches, F1_score
from lighteval.metrics.normalizations import get_multilingual_normalizer
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.tasks.requests import SamplingMethod
from lighteval.utils.language import Language
from src.metrics.utils import clean_model_output


def normalize_answer(language: Language, lower: bool = True) -> Callable[[str], str]:
    def _inner_normalize_answer(text: str) -> str:
        text = clean_model_output(text)

        normalizer = get_multilingual_normalizer(language, lower=lower)
        return normalizer(text)

    return _inner_normalize_answer


class MultilingualCustomExactMatchMetric(SampleLevelMetric):
    def __init__(
        self,
        language: Language,
        match_type: Literal["prefix", "suffix", "full"] = "full",
        aggregation_function: Callable[[list[float]], float] = max,
    ):
        """Creates a language-aware exact match metric, which returns the exact match score
        Args:
            language: The language of the samples.
            match_type: The type of match to use
                - "prefix": Prefixes must match
                - "suffix": Suffixes must match
                - "full": Full strings must match
            aggregation_function: Aggregation samples to use when multiple golds are present.
        """
        super().__init__(
            metric_name=f"exact_match_{language.value}_{match_type}",
            sample_level_fn=ExactMatches(
                normalize_gold=get_multilingual_normalizer(language),
                normalize_pred=normalize_answer(language),
                aggregation_function=aggregation_function,
                type_exact_match=match_type,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )


class MultilingualCustomF1ScoreMetric(SampleLevelMetric):
    def __init__(self, language: Language, aggregation_function: Callable[[list[float]], float] = max):
        """Creates a language-aware F1 score metric, which returns the F1 score.

        Args:
            language: The language of the samples.
            aggregation_function: Aggregation samples to use when multiple golds are present.
        """
        super().__init__(
            metric_name=f"f1_{language.value}",
            sample_level_fn=F1_score(
                normalize_gold=normalize_answer(language),
                normalize_pred=normalize_answer(language),
                aggregation_function=aggregation_function,
            ),
            category=SamplingMethod.GENERATIVE,
            corpus_level_fn=np.mean,
            higher_is_better=True,
        )
