from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_gradation_prompt_base(
    transform: Callable,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Gradation prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        query_template: Template for the query with placeholder for question
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        lemma = transformed["lemma"]
        degree_label = transformed["degree_label"]
        answer = transformed["answer"]

        query = query_template.format(lemma=lemma, degree_label=degree_label)

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=None,
        )

    return prompt_fn


GRADATION_PROMPT_CONFIGS = {
    "prompt1_gradation_sk": {
        "query_template": "Uveď {degree_label} prídavného mena/príslovky „{lemma}“. Odpovedz iba jedným slovom.",
    },
    "prompt2_gradation_sk": {
        "query_template": "Vytvor {degree_label} prídavného mena/príslovky „{lemma}“. Odpovedz iba jedným slovom.",
    },
    "prompt3_gradation_sk": {
        "query_template": "Úloha: pre prídavné meno/príslovku „{lemma}“ uveď {degree_label}. V odpovedi napíš iba jedno slovo.",
    },
}

for name, config in GRADATION_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_gradation_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_gradation)(prompt_fn)
