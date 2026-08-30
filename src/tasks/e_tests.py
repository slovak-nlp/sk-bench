r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{etesty,
  title  = {{e-testy-sk}: {S}lovak High-School Examination Question Bank},
  author = {{Slovak NLP Community}},
  year   = {2025},
  howpublished = {Hugging Face dataset (URL withheld for anonymous review)},
}
Citation of the sk-bench benchmark (this repository):

@inproceedings{suppa2026skbench,
title={sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak},
author={{\v{S}}uppa, Marek and Vykopal, Ivan and Ridzik, Andrej and Sopkovi{\v{c}}, Kristi{\'a}n and K{\v{n}}a{\v{z}}ekov{\'a}, Nat{\'a}lia and Kop{\v{c}}an, Jaroslav and Bl{\v{s}}t{\'a}k, Miroslav and Ondrejov{\'a}, Vikt{\'o}ria and Hl{\'a}dek, Daniel and Gregor, Michal and Tamajka, Martin and {\v{S}}imko, Mari{\'a}n},
booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
year={2026},
url={https://openreview.net/forum?id=LLfU6QSSEF}
}
"""

import json

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language
from src.metrics import conservative_exact_match, em_relaxed_multi_label, relaxed_exact_match
from src.metrics.qa_metrics import MultilingualCustomExactMatchMetric, MultilingualCustomF1ScoreMetric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


SUBSETS = [
    "biologia",
    "obcianska-nauka",
    "informatika",
    "geografia",
    "matematika",
    "chemia",
    "dejepis",
    "fyzika",
    "slovensky-jazyk",
]


def get_gold_indices(options, single_choice=False):
    """Get the indices of the correct options from the list of options."""
    if isinstance(options, str):
        options = json.loads(options)

    gold_indices = []
    for i, option in enumerate(options):
        if option["correct"]:
            gold_indices.append(i)

    if single_choice:
        if len(gold_indices) != 1:
            raise ValueError("Expected exactly one correct option for single choice question")
        return gold_indices[0]

    return gold_indices


def create_e_tests_single_choice_task(prompt_name: str, subset: str):
    """Create an E-Tests single choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("e_tests", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": [option["text"] for option in json.loads(line["options"])],
                "gold_idx": get_gold_indices(json.loads(line["options"]), single_choice=True),
            }
        ),
        hf_repo="slovak-nlp/e-testy-sk",
        hf_subset="default",
        hf_revision="1a5e798e0473241097c4a26c511426167522adf5",
        hf_filter=lambda record: record["question_type"] == "single_choice"
        and not record["has_image"]
        and len(record["correct_answers"]) == 1
        and record["subject"] == subset,
        hf_avail_splits=["train"],
        evaluation_splits=("train",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


def create_e_tests_multiple_choice_task(prompt_name: str, subset: str):
    """Create an E-Tests multiple choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("e_tests", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": [option["text"] for option in json.loads(line["options"])],
                "gold_idx": get_gold_indices(json.loads(line["options"]), single_choice=False),
            }
        ),
        hf_repo="slovak-nlp/e-testy-sk",
        hf_subset="default",
        hf_revision="1a5e798e0473241097c4a26c511426167522adf5",
        hf_filter=lambda record: record["question_type"] == "multiple_choice"
        and not record["has_image"]
        and record["options"] is not None
        and record["subject"] == subset,
        hf_avail_splits=["train"],
        evaluation_splits=("train",),
        generation_size=32,
        metrics=[
            em_relaxed_multi_label,
        ],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


def create_e_tests_text_input_task(prompt_name: str, subset: str):
    """Create an E-Tests text input task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("e_tests", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "answers": {
                    "text": line["correct_answers"],
                },
            }
        ),
        hf_repo="slovak-nlp/e-testy-sk",
        hf_subset="default",
        hf_revision="1a5e798e0473241097c4a26c511426167522adf5",
        hf_filter=lambda record: record["question_type"] == "text_input"
        and not record["has_image"]
        and record["subject"] == subset,
        hf_avail_splits=["train"],
        evaluation_splits=("train",),
        generation_size=256,
        metrics=[
            MultilingualCustomExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualCustomF1ScoreMetric(Language.SLOVAK),
            Metrics.rouge1,
            Metrics.rouge2,
            Metrics.rougeL,
        ],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_question_answering)
    return task


TASKS_TABLE = (
    [
        create_e_tests_single_choice_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
        for subset in SUBSETS
    ]
    + [
        create_e_tests_multiple_choice_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_select)
        for subset in SUBSETS
    ]
    + [
        create_e_tests_text_input_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_question_answering)
        for subset in SUBSETS
        if subset != "matematika"
    ]
)
