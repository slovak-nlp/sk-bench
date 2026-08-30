from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_declension_prompt_base(
    transform: Callable,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Declension prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        query_template: Template for the query with placeholder for question
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        lemma = transformed["lemma"]
        target_description = transformed["target_description"]
        answer = transformed["answer"]

        # add v or vo before target_description if it starts with a "v"
        if target_description.startswith("v"):
            target_description = "vo " + target_description
        else:
            target_description = "v " + target_description

        query = query_template.format(lemma=lemma, target_description=target_description)

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=None,
        )

    return prompt_fn


DECLENSION_PROMPT_CONFIGS = {
    "prompt1_declension_sk": {
        "query_template": "Uveď tvar podstatného mena „{lemma}“ {target_description}. Odpovedz iba jedným slovom.",
    },
    "prompt2_declension_sk": {
        "query_template": "Vyskloňuj podstatné meno „{lemma}“ {target_description}. Odpovedz iba jedným slovom.",
    },
    "prompt3_declension_sk": {
        "query_template": "Úloha: pre podstatné meno „{lemma}“ napíš tvar {target_description}. V odpovedi uveď iba jedno slovo.",
    },
}

for name, config in DECLENSION_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_declension_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_declension)(prompt_fn)
