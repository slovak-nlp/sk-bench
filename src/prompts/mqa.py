from functools import partial
from string import ascii_uppercase
from typing import Callable, Literal

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_mqa_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    language: Literal["en", "sk"],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating MQA prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by mqa_prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholder for question
        language: The language of the prompts
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        question = transformed["question"]
        choices = transformed["choices"]
        gold_idx = transformed["gold_idx"]
        context = transformed.get("context", None)

        # Generate choices dynamically based on actual number of choices
        letter_prefixes = [f"{letter}) " for letter in ascii_uppercase[: len(choices)]]
        choice_text = "\n".join(f"{prefix}{choice}" for prefix, choice in zip(letter_prefixes, choices))
        letter_string = ", ".join(ascii_uppercase[: len(choices)])

        query = ""
        context_instruction = ""
        if context and context.strip() != "":
            if language == "en":
                query += f"Context: {context}\n\n"
                context_instruction = " only based on the given context"
            elif language == "sk":
                query += f"Kontext: {context}\n\n"
                context_instruction = " len na základe uvedeného kontextu"

        query += query_template.format(question=question, choices=choice_text)

        output_choices = [f"{letter}" for letter in ascii_uppercase[: len(choices)]]
        instruction = (
            instruction_template.format(letter_string=letter_string, context_instruction=context_instruction)
            if instruction_template
            else None
        )

        return Doc(
            task_name=task_name,
            query=query,
            choices=output_choices,
            gold_index=gold_idx,
            instruction=instruction,
        )

    return prompt_fn


MQA_SINGLE_CORRECT_PROMPT_CONFIGS = {
    "prompt1_mqa_single_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Choose the correct answer {letter_string} for the given question{context_instruction}. There is only one correct answer for each question. Answer only with the letter of the correct option.",
    },
    "prompt1_mqa_single_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Vyber správnu odpoveď {letter_string} pre zadanú otázku{context_instruction}. Pre každú otázku existuje len jedna správna odpoveď. Odpovedz len písmenom správnej možnosti.",
    },
    "prompt2_mqa_single_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Task: Select the correct answer {letter_string} for the following question{context_instruction}. Answer only with the letter of the correct option.",
    },
    "prompt2_mqa_single_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Úloha: Vyber správnu odpoveď {letter_string} na nasledujúcu otázku{context_instruction}. Odpovedz len písmenom správnej možnosti.",
    },
    "prompt3_mqa_single_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "I will present you with questions and possible answers {letter_string}, and your task is to select the correct answer{context_instruction}. Answer only with the letter of the correct option.",
    },
    "prompt3_mqa_single_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Predložím ti otázky a možné odpovede {letter_string} a tvojou úlohou je vybrať správnu odpoveď{context_instruction}. Odpovedz len písmenom správnej možnosti.",
    },
}

MQA_MULTIPLE_CORRECT_PROMPT_CONFIGS = {
    "prompt1_mqa_multiple_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Choose all the correct answers {letter_string} for the given question{context_instruction}. There may be one or more correct answers for each question. Answer by listing the letters of the correct answers separated by commas.",
    },
    "prompt1_mqa_multiple_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Vyber všetky správne odpovede {letter_string} pre zadanú otázku{context_instruction}. Pre každú otázku môže byť jedna alebo viac správnych odpovedí. Odpovedz zoznamom písmen správnych odpovedí oddelených čiarkami.",
    },
    "prompt2_mqa_multiple_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "Task: Select all the correct answers {letter_string} for the following question{context_instruction}. Answer by listing the letters of the correct answers separated by commas.",
    },
    "prompt2_mqa_multiple_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Úloha: Vyber všetky správne odpovede {letter_string} na nasledujúcu otázku{context_instruction}. Odpovedz zoznamom písmen správnych odpovedí oddelených čiarkami.",
    },
    "prompt3_mqa_multiple_correct_en": {
        "query_template": "Question: {question}\nChoices:\n{choices}\n\nAnswer:",
        "instruction_template": "I will present you with questions and possible answers {letter_string}, and your task is to select all the correct answers{context_instruction}. Answer by listing the letters of the correct answers separated by commas.",
    },
    "prompt3_mqa_multiple_correct_sk": {
        "query_template": "Otázka: {question}\nMožnosti:\n{choices}\n\nOdpoveď:",
        "instruction_template": "Predložím ti otázky a možné odpovede {letter_string} a tvojou úlohou je vybrať všetky správne odpovede{context_instruction}. Odpovedz zoznamom písmen správnych odpovedí oddelených čiarkami.",
    },
}


for name, config in MQA_SINGLE_CORRECT_PROMPT_CONFIGS.items():
    language = "sk" if "_sk" in name else "en"
    prompt_fn = partial(_get_mqa_prompt_base, **config, language=language)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_multi_choice)(prompt_fn)

for name, config in MQA_MULTIPLE_CORRECT_PROMPT_CONFIGS.items():
    language = "sk" if "_sk" in name else "en"
    prompt_fn = partial(_get_mqa_prompt_base, **config, language=language)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_multi_select)(prompt_fn)
