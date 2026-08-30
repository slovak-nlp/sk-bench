from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_identity_form_prompt_base(
    transform: Callable,
    query_template: str,
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating Identity Form prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        query_template: Template for the query with placeholder for question
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        lemma = transformed["lemma"]
        feature_description_sk = transformed["feature_description_sk"]
        form = transformed["form"]
        answer = transformed["answer"]

        query = query_template.format(lemma=lemma, feature_description_sk=feature_description_sk, form=form)

        instruction = """Slovenské gramatické kategórie a ich hodnoty:
  - pád: nominatív, genitív, datív, akuzatív, vokatív, lokál, inštrumentál
  - číslo: jednotné, množné
  - rod: mužský životný, mužský neživotný, ženský, stredný
  - stupeň: prvý, druhý, tretí
  - osoba: prvá, druhá, tretia
  - čas alebo spôsob: neurčitok, prítomný, minulý, budúci, rozkazovací, prechodník"""

        return Doc(
            task_name=task_name,
            query=query,
            choices=[answer],
            gold_index=0,
            instruction=instruction,
        )

    return prompt_fn


IDENTITY_FORM_PROMPT_CONFIGS = {
    "prompt1_identity_form_sk": {
        "query_template": "Uveď {feature_description_sk} tvaru „{form}“ (lemma „{lemma}“). Odpovedz len jednou zvolenou hodnotou gramatickej kategórie.",
    },
    "prompt2_identity_form_sk": {
        "query_template": "Urč {feature_description_sk} tvaru „{form}“ (lemma „{lemma}“). Odpovedz len jednou zvolenou hodnotou gramatickej kategórie.",
    },
    "prompt3_identity_form_sk": {
        "query_template": "Úloha: pre tvar „{form}“ (lemma „{lemma}“) uveď {feature_description_sk}. Odpovedz len jednou zvolenou hodnotou gramatickej kategórie.",
    },
}

for name, config in IDENTITY_FORM_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_identity_form_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_identity_form)(prompt_fn)
