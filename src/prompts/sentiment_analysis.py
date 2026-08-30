from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_sentiment_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating sentiment analysis prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for text
        query_template: Template for the query with placeholder for text
        choices: List of choice texts

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        text = transformed["text"]
        gold_idx = transformed["gold_idx"]

        query = query_template.format(text=text)

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction_template,
        )

    return prompt_fn


SENTIMENT_BINARY_PROMPT_CONFIGS = {
    "prompt1_sentiment_binary_en": {
        "query_template": "Text: {text}\nI consider the sentiment to be:",
        "instruction_template": 'Determine the sentiment of the following text. Answer only "negative" or "positive".',
        "choices": ["Negative", "Positive"],
    },
    "prompt1_sentiment_binary_sk": {
        "query_template": "Text: {text}\nSentiment považujem za:",
        "instruction_template": "Urči sentiment nasledujúceho textu. Odpovedz iba slovom „negatívny“ alebo „pozitívny“.",
        "choices": ["Negatívny", "Pozitívny"],
    },
    "prompt2_sentiment_binary_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Determine whether the sentiment of the following text is negative or positive. Answer only "negative" or "positive".',
        "choices": ["Negative", "Positive"],
    },
    "prompt2_sentiment_binary_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Rozhodni, či je sentiment nasledujúceho textu negatívny alebo pozitívny. Odpovedz iba slovom „negatívny“ alebo „pozitívny“.",
        "choices": ["Negatívny", "Pozitívny"],
    },
    "prompt3_sentiment_binary_en": {
        "query_template": "Text: {text}\nSentiment:",
        "instruction_template": "Classify sentiment into the following categories: negative/positive.",
        "choices": ["Negative", "Positive"],
    },
    "prompt3_sentiment_binary_sk": {
        "query_template": "Text: {text}\nSentiment:",
        "instruction_template": "Klasifikuj sentiment do kategórií: negatívny/pozitívny.",
        "choices": ["Negatívny", "Pozitívny"],
    },
}

SENTIMENT_3_CLASS_PROMPT_CONFIGS = {
    "prompt1_sentiment_3_class_en": {
        "query_template": "Text: {text}\nI consider the sentiment to be:",
        "instruction_template": 'Determine the sentiment of the following text. Answer only "negative", "neutral", or "positive".',
        "choices": ["Negative", "Neutral", "Positive"],
    },
    "prompt1_sentiment_3_class_sk": {
        "query_template": "Text: {text}\nSentiment považujem za:",
        "instruction_template": "Urči sentiment nasledujúceho textu. Odpovedz iba slovom „negatívny“, „neutrálny“ alebo „pozitívny“.",
        "choices": ["Negatívny", "Neutrálny", "Pozitívny"],
    },
    "prompt2_sentiment_3_class_en": {
        "query_template": "Text: {text}\nAnswer:",
        "instruction_template": 'Determine whether the sentiment of the following text is negative, neutral, or positive. Answer only with "negative", "neutral", or "positive".',
        "choices": ["Negative", "Neutral", "Positive"],
    },
    "prompt2_sentiment_3_class_sk": {
        "query_template": "Text: {text}\nOdpoveď:",
        "instruction_template": "Rozhodni, či sentiment nasledujúceho textu je negatívny, neutrálny alebo pozitívny. Odpovedz iba slovom „negatívny“, „neutrálny“ alebo „pozitívny“.",
        "choices": ["Negatívny", "Neutrálny", "Pozitívny"],
    },
    "prompt3_sentiment_3_class_en": {
        "query_template": "Text: {text}\nSentiment:",
        "instruction_template": "Classify sentiment into the following categories: negative/neutral/positive.",
        "choices": ["Negative", "Neutral", "Positive"],
    },
    "prompt3_sentiment_3_class_sk": {
        "query_template": "Text: {text}\nSentiment:",
        "instruction_template": "Klasifikuj sentiment do kategórií: negatívny/neutrálny/pozitívny.",
        "choices": ["Negatívny", "Neutrálny", "Pozitívny"],
    },
}


for name, config in SENTIMENT_BINARY_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_sentiment_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_sentiment_binary)(prompt_fn)

for name, config in SENTIMENT_3_CLASS_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_sentiment_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_sentiment_3class)(prompt_fn)
