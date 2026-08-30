from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_copa_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating CoPA prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by mqa_prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question
        choices: List of choices to be included in the prompt

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        premise = transformed["premise"]
        choices_text = transformed["choices"]
        gold_idx = transformed["gold_idx"]

        query = query_template.format(premise=premise, choice1=choices_text[0], choice2=choices_text[1])
        instruction = instruction_template if instruction_template else None

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction,
        )

    return prompt_fn


COPA_PROMPT_CONFIGS = {
    "prompt1_copa_en": {
        "query_template": 'Given the premise "{premise}", and that we are looking for the cause of this premise, which hypothesis seems more plausible?\n\nHypothesis 1: {choice1}.\nHypothesis 2: {choice2}.\n\nAnswer only with "1" or "2".\n\nAnswer:',
        "instruction_template": None,
        "choices": ["1", "2"],
    },
    "prompt1_copa_sk": {
        "query_template": "Vzhľadom na predpoklad „{premise}“, a skutočnosť, že hľadáme príčinu tohto predpokladu, ktorá hypotéza sa zdá byť pravdepodobnejšia?\n\nHypotéza 1: {choice1}\nHypotéza 2: {choice2}\n\nOdpovedz len „1“ alebo „2“.\n\nOdpoveď:",
        "instruction_template": None,
        "choices": ["1", "2"],
    },
    "prompt2_copa_en": {
        "query_template": 'Given the premise "{premise}", and that we are wondering what happened as a result of this premise, which hypothesis seems more plausible?\n\nHypothesis 1: {choice1}.\nHypothesis 2: {choice2}.\n\nAnswer only with "1" or "2".\n\nAnswer:',
        "instruction_template": None,
        "choices": ["1", "2"],
    },
    "prompt2_copa_sk": {
        "query_template": "Na základe predpokladu „{premise}“ a na to, že nás zaujíma, čo sa stalo v dôsledku tohto predpokladu, ktorá hypotéza sa javí ako pravdepodobnejšia?\n\nHypotéza 1: {choice1}\nHypotéza 2: {choice2}\n\nOdpovedz len „1“ alebo „2“.\n\nOdpoveď:",
        "instruction_template": None,
        "choices": ["1", "2"],
    },
    "prompt3_copa_en": {
        "query_template": 'A text description of the situation "{premise}" and two text fragments of the description "{choice1}" and "{choice2}" are given. Decide which of the two fragments is a consequence of the described situation? Answer with one number "1" or "2", without any additional text.\n\nAnswer:',
        "instruction_template": None,
        "choices": ["1", "2"],
    },
    "prompt3_copa_sk": {
        "query_template": "Je uvedený textový popis situácie „{premise}“ a dva úryvky textu s popisom „{choice1}“ a „{choice2}“. Rozhodni, ktorý z týchto dvoch úryvkov je dôsledkom opísanej situácie. Odpovedz jedným číslom „1“ alebo „2“, bez ďalšieho textu.\n\nOdpoveď:",
        "instruction_template": None,
        "choices": ["1", "2"],
    },
}

for name, config in COPA_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_copa_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_copa)(prompt_fn)
