from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_chiby_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Chiby prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by mqa_prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question
        choices: List of possible answer choices
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        sentence = transformed["sentence"]
        gold_idx = transformed["gold_idx"]

        query = query_template.format(sentence=sentence)
        instruction = instruction_template if instruction_template else None

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction,
        )

    return prompt_fn


CHIBY_PROMPT_CONFIGS = {
    "prompt1_chiby_en": {
        "query_template": "Sentence: {sentence}\nAnswer:",
        "instruction_template": 'Check whether the sentence is grammatically correct or incorrect.\nA sentence is not written correctly if it contains spelling errors (such as missing diacritics or incorrect capitalisation), morphological errors (such as incorrect declension), or lexical errors (such as Czechisms). It must be written grammatically correctly.\n\nIgnore extra spaces, do not change commas to semicolons, and do not use an em dash (—). Pay attention to the correct use of the en dash (–) and the hyphen (-).\n\nIf the sentence is incorrect, write "no", if the sentence is correct, write "yes".',
        "choices": ["No", "Yes"],
    },
    "prompt1_chiby_sk": {
        "query_template": "Veta: {sentence}\nOdpoveď:",
        "instruction_template": "Skontroluj, či je daná veta gramaticky správna alebo nesprávna.\nVeta nie je napísaná správne pokiaľ obsahuje pravopisné chyby (napríklad chýbajúca diakritika, písanie veľkých a malých písmen), morfologické chyby (napríklad skloňovanie) alebo lexikálne chyby (napríklad čechizmy). Musí byť napísaná gramaticky správne.\n\nIgnoruj medzery navyše, nemeň čiarku za bodkočiarku a ani nepoužívaj dlhú pomlčku (—). Daj pozor na správne využitie pomlčky (–) a spojovníka (-).\n\nAk je veta nesprávna napíš „nie“ a ak je veta správna napíš „áno“.",
        "choices": ["Nie", "Áno"],
    },
    "prompt2_chiby_en": {
        "query_template": "Sentence: {sentence}\nAnswer:",
        "instruction_template": 'Your task is to indicate whether each sentence is grammatically correct (answer "yes") or incorrect (answer "no").\n\nIgnore extra spaces and do not change commas to semicolons. Do not use an em dash (—). Use only an en dash (–) or a hyphen (-).',
        "choices": ["No", "Yes"],
    },
    "prompt2_chiby_sk": {
        "query_template": "Veta: {sentence}\nOdpoveď:",
        "instruction_template": "Tvojou úlohou je napísať, či je veta gramaticky správna (odpovedz slovom „áno“) alebo nesprávna (odpovedz slovom „nie“).\n\nIgnoruj medzery navyše a nemeň čiarku za bodkočiarku. Nepoužívaj dlhú pomlčku (—). Používaj iba pomlčku (–) a spojovník (-).",
        "choices": ["Nie", "Áno"],
    },
    "prompt3_chiby_en": {
        "query_template": "Sentence: {sentence}\nAnswer:",
        "instruction_template": 'A sentence is grammatically correct if it contains no errors (such as spelling, morphological, stylistic, lexical or other errors).\n\nDo not consider extra spaces to be errors, and do not swap comma for a semicolon.\nDo not use the em dash (—). Use only the en dash (–) or the hyphen (-).\n\nIf the sentence is correct, write "yes", if it is incorrect, write "no".',
        "choices": ["No", "Yes"],
    },
    "prompt3_chiby_sk": {
        "query_template": "Veta: {sentence}\nOdpoveď:",
        "instruction_template": "Veta je gramaticky správna ak neobsahuje žiadne typy chýb (napríklad pravopisné, morfologické, štylistické, lexikálne a iné).\n\nDo chýb nerátaj medzery navyše a nezamieňaj čiarku s bodkočiarkou.\nNepoužívaj dlhú pomlčku (—). Používaj iba pomlčku (–) a spojovník (-).\n\nPokiaľ je veta správna, napíš „áno“ a pokiaľ je nesprávna, napíš „nie“.",
        "choices": ["Nie", "Áno"],
    },
}

for name, config in CHIBY_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_chiby_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_chiby)(prompt_fn)
