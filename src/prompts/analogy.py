from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_analogy_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Analogies prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by analogy prompts. It should return a dict with keys 'concept1', 'concept2', 'concept3', 'gold_idx', and optionally 'choices'.
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        concept1 = transformed["concept1"]
        concept2 = transformed["concept2"]
        concept3 = transformed["concept3"]
        gold_idx = transformed["gold_idx"]
        choices = transformed.get("choices", [])

        query = query_template.format(concept1=concept1, concept2=concept2, concept3=concept3)

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction_template,
        )

    return prompt_fn


ANALOGY_PROMPT_CONFIGS = {
    "prompt1_analogy_en": {
        "query_template": "{concept1} - {concept2}\n{concept3} - ?\n\nAnswer:",
        "instruction_template": "Fill in the word so that it fits analogically with the preceding three words. Output only the missing word, nothing else.",
    },
    "prompt1_analogy_sk": {
        "query_template": "{concept1} - {concept2}\n{concept3} - ?\n\nOdpoveď:",
        "instruction_template": "Doplň slovo tak, aby sa analogicky hodilo k predchádzajúcim trom slovám. Napíš iba chýbajúce slovo, nič iné.",
    },
    "prompt2_analogy_en": {
        "query_template": "{concept1} ↔ {concept2}\n{concept3} ↔ ?\n\nAnswer:",
        "instruction_template": "Complete the analogy. Given the relationship between the first pair of words, apply the same relationship to find the missing word. Output only the missing word, nothing else.",
    },
    "prompt2_analogy_sk": {
        "query_template": "{concept1} ↔ {concept2}\n{concept3} ↔ ?\n\nOdpoveď:",
        "instruction_template": "Doplň analógiu. Na základe vzťahu medzi prvým párom slov uplatni rovnaký vzťah a nájdi chýbajúce slovo. Napíš iba chýbajúce slovo, nič iné.",
    },
    "prompt3_analogy_en": {
        "query_template": "{concept1} - {concept2}\n{concept3} - ?\n\nAnswer:",
        "instruction_template": "You are given a word analogy in the following format:\nA - B\nC - ?\nYour task is to identify the relationship between A and B, then apply that same relationship to C to produce the missing word. Output only the missing word, nothing else.",
    },
    "prompt3_analogy_sk": {
        "query_template": "{concept1} - {concept2}\n{concept3} - ?\n\nOdpoveď:",
        "instruction_template": "Dostaneš slovnú analógiu v nasledujúcom formáte:\nA - B\nC - ?\nTvojou úlohou je zistiť vzťah medzi slovami A a B a rovnaký vzťah uplatniť na slovo C, čím získaš chýbajúce slovo. Napíš iba chýbajúce slovo, nič iné.",
    },
}

for name, config in ANALOGY_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_analogy_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_analogy)(prompt_fn)
