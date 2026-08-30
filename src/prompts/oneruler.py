from functools import partial
from typing import Callable

from typing_extensions import Literal

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


SLOVAK_NIAH_PROMPT = {
    "task": "Prečítaj si a zapamätaj nasledujúci text. Neskôr sa ťa naň spýtam.\n\n<Text>\n{context}\n</Text>\n\n",
    "needle_words": 'Pre heslo "{key}" je špeciálne magické slovo: {value}',
    "needle_numbers": 'Pre heslo "{key}" je špeciálne magické číslo: {value}',
    "question_single_numbers": '<Otázka> Aké špeciálne magické čísla spojené so slovom "{query1}" sú uvedené v poskytnutom texte?',
    "question_single_words": '<Otázka> Aké špeciálne magické slová spojené so slovom "{query1}" sú uvedené v poskytnutom texte?',
    "question_multi_numbers": '<Otázka> Aké špeciálne magické čísla spojené so slovami "{query1}" a "{query2}" sú uvedené v poskytnutom texte?',
    "question_multi_words": '<Otázka> Aké špeciálne magické slová spojené so slovami "{query1}" a "{query2}" sú uvedené v poskytnutom texte?',
    "please_list": "Uveď prosím všetky, ktoré zodpovedajú.",
    "if_no_numbers": 'Ak takéto čísla neexistujú, odpovedz prosím "žiadne".</Otázka>\n\n\n',
    "if_no_words": 'Ak takéto slová neexistujú, odpovedz prosím "žiadne".</Otázka>\n\n\n',
    "answer_prefix": "Odpoveď uveď v nasledujúcom formáte:\n",
    "answer_words": "<Odpoveď> Tu uveď všetky slová </Odpoveď>",
    "answer_numbers": "<Odpoveď> Tu uveď všetky čísla </Odpoveď>",
    "none": "žiadne",
}


ENGLISH_NIAH_PROMPT = {
    "task": "Please read and memorize the text below. I will ask you about it later.\n\n<text>\n{context}\n</text>\n\n",
    "needle_words": 'The special magic word for "{key}" is: {value} ',
    "needle_numbers": 'The special magic number for "{key}" is: {value} ',
    "question_single_numbers": '<Question> What special magic numbers associated with "{query1}" are mentioned in the provided text?',
    "question_single_words": '<Question> What special magic words associated with "{query1}" are mentioned in the provided text?',
    "question_multi_numbers": '<Question> What special magic numbers associated with "{query1}" and "{query2}" are mentioned in the provided text?',
    "question_multi_words": '<Question> What special magic words associated with "{query1}" and "{query2}" are mentioned in the provided text? ',
    "please_list": "Please list all that apply.",
    "if_no_numbers": 'If no such numbers exist, please answer "none".</Question>\n\n\n',
    "if_no_words": 'If no such words exist, please answer "none".</Question>\n\n\n',
    "answer_prefix": "Please provide your answer in the following format:\n",
    "answer_words": "<Answer> List all words here </Answer>",
    "answer_numbers": "<Answer> List all numbers here </Answer>",
    "none": "none",
}


SLOVAK_CWE_PROMPT = """Prečítaj si a zapamätaj si nasledujúci zoznam slov. V tomto zozname sa niektoré slová opakujú častejšie ako iné. Zapamätaj si tie, ktoré sa vyskytujú najčastejšie.

<Zoznam> {context} </Zoznam>

<Otázka> Akých je 10 najčastejších slov v zozname? </Otázka>

Uveď svoju odpoveď v nasledujúcom formáte:
<Odpoveď> Tu uveď slová </Odpoveď>"""


ENGLISH_CWE_PROMPT = """Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.

<List> {context} </List>

<Question> What are the 10 most common words in the above list? </Question>

Please provide your answer in the following format:
<Answer> List the words here </Answer>"""


def format_cwe_prompt(context: str, language: Literal["slovak", "english"] = "slovak") -> str:
    if language == "slovak":
        return SLOVAK_CWE_PROMPT.format(context=context)
    else:
        return ENGLISH_CWE_PROMPT.format(context=context)


def format_niah_prompt(
    context: str, query1: str, query2: str, task: str, language: Literal["slovak", "english"] = "slovak"
) -> str:
    if language == "slovak":
        prompt = SLOVAK_NIAH_PROMPT
    else:
        prompt = ENGLISH_NIAH_PROMPT

    if task == "niah_multiquery":
        question_template = prompt["question_multi_numbers"]
        question_text = question_template.format(query1=query1, query2=query2)
    else:
        question_template = prompt["question_single_numbers"]
        question_text = question_template.format(query1=query1)

    full_input = (
        prompt["task"].format(context=context)
        + question_text
        + " "
        + prompt["please_list"]
        + " "
        + prompt["if_no_numbers"]
        + prompt["answer_prefix"]
        + prompt["answer_numbers"]
    )
    return full_input


def _get_oneruler_prompt_base(
    transform: Callable,
    language: Literal["slovak", "english"] = "slovak",
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating NLI prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        language: The language of the prompt, either "slovak" or "english"

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        outputs = transformed["outputs"]
        context = transformed["context"]
        query1 = transformed["query1"]
        query2 = transformed["query2"]
        task = transformed["task"]
        lang = transformed["language"]
        gold_idx = transformed["gold_idx"]

        if task == "cwe":
            query = format_cwe_prompt(context=context, language=language)
        else:
            query = format_niah_prompt(context=context, query1=query1, query2=query2, task=task, language=language)

        if "niah_none" in task and language == "english":
            outputs = ["none"]

        return Doc(
            task_name=task_name,
            query=query,
            choices=outputs,
            gold_index=gold_idx,
            specific={"task": line["task"], "language": lang},
        )

    return prompt_fn


PROMPT_REGISTRY.register("oneruler_slovak", category=PromptCategory.p_oneruler)(
    partial(_get_oneruler_prompt_base, language="slovak")
)
PROMPT_REGISTRY.register("oneruler_english", category=PromptCategory.p_oneruler)(
    partial(_get_oneruler_prompt_base, language="english")
)
