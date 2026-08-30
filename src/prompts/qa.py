from functools import partial
from typing import Callable, Literal

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_qa_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    language: Literal["en", "sk"],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating QA prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholders for context and question
        language: The language of the prompt, either "en" or "sk"
    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        context = transformed.get("context", None)
        question = transformed["question"]
        answers = transformed["answers"]
        unknown = transformed.get("unknown", False)

        query = ""
        context_instruction = ""
        if context and context.strip() != "":
            if language == "en":
                query += f"Context: {context}\n\n"
                context_instruction = " only based on the given context"
            elif language == "sk":
                query += f"Kontext: {context}\n\n"
                context_instruction = " len na základe uvedeného kontextu"

        unknown_suffix_instruction = ""
        unknown_suffix_query = ""
        if unknown:
            # Handle no-answer case by adding a special "neviem/unknown" choice and adjusting gold index accordingly
            empty_answer = "neviem" if language == "sk" else "unknown"
            if len(answers["text"]) == 0:
                choices = [empty_answer]
                gold_index = [0]
            else:
                choices = [ans if len(ans) > 0 else empty_answer for ans in answers["text"]]
                gold_index = list(range(len(choices)))

            if language == "sk":
                unknown_suffix = "Odpovedz „neviem“ v prípade, ak sa otázka nedá zodpovedať len na základe kontextu."
            if language == "en":
                unknown_suffix = 'Answer "unknown" if the question cannot be answered only based on the context.'

            if instruction_template:
                unknown_suffix_instruction = " " + unknown_suffix
            else:
                unknown_suffix_query = "\n" + unknown_suffix
        else:
            choices = [ans for ans in answers["text"] if len(ans) > 0]
            gold_index = [i for i, ans in enumerate(choices) if ans in answers["text"]]

        instruction = (
            instruction_template.format(
                context_instruction=context_instruction, unknown_suffix=unknown_suffix_instruction
            )
            if instruction_template
            else None
        )
        query += query_template.format(question=question, unknown_suffix=unknown_suffix_query)

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_index,
            instruction=instruction,
        )

    return prompt_fn


# Register prompt variants dynamically
QA_PROMPT_CONFIGS = {
    "prompt1_qa_en": {
        "query_template": "Question: {question}\nAnswer:",
        "instruction_template": "Answer the following question directly{context_instruction}, without explanation, in Slovak.{unknown_suffix}",
    },
    "prompt1_qa_sk": {
        "query_template": "Otázka: {question}\nOdpoveď:",
        "instruction_template": "Odpovedz na nasledujúcu otázku priamo{context_instruction}, bez vysvetľovania, v slovenčine.{unknown_suffix}",
    },
    "prompt2_qa_en": {
        "query_template": "Question: {question}\nAnswer:",
        "instruction_template": "Task: Find the answer to the question{context_instruction}. The answer has to be in Slovak.{unknown_suffix}",
    },
    "prompt2_qa_sk": {
        "query_template": "Otázka: {question}\nOdpoveď:",
        "instruction_template": "Úloha: Nájdi odpoveď na nasledujúcu otázku{context_instruction}. Odpoveď musí byť v slovenčine.{unknown_suffix}",
    },
    "prompt3_qa_en": {
        "query_template": "Answer the following question in Slovak: {question}{unknown_suffix}\n\nAnswer:",
        "instruction_template": None,
    },
    "prompt3_qa_sk": {
        "query_template": "Odpovedz na nasledujúcu otázku v slovenčine: {question}{unknown_suffix}\n\nOdpoveď:",
        "instruction_template": None,
    },
}

for name, config in QA_PROMPT_CONFIGS.items():
    language = "sk" if "_sk" in name else "en"
    prompt_fn = partial(_get_qa_prompt_base, **config, language=language)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_question_answering)(prompt_fn)
