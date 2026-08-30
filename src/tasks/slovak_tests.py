r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{slovak_tests,
  title        = {{Slovak School Exams}: {S}lovak School-Examination Question
                  Bank},
  author       = {{Slovak NLP Community}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/slovak-nlp/slovak-school-exams-test-preprocessed}
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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.metrics.qa_metrics import MultilingualCustomExactMatchMetric, MultilingualCustomF1ScoreMetric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


SUBSETS = ["vyssie_studium", "monitor9", "maturita", "prijmacie_skusky"]


def filtering(line):
    """Filter out records where the answer is not in the options."""
    if not line.get("options") or not line.get("answer"):
        return False
    options_text = line["options"]

    return line["answer"] in options_text


def create_slovak_tests_multiple_choice_task(prompt_name: str, subset: str):
    """Create a Slovak multiple choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slovak_tests", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["questions"],
                "context": line["reference"],
                "choices": line["options"],
                "gold_idx": line["options"].index(line["answer"]),
            }
        ),
        hf_repo="slovak-nlp/slovak-school-exams-test-preprocessed",
        hf_subset="default",
        hf_revision="1d51f8205ee2341d11f05273d9c6d4ba3532a1df",
        hf_filter=lambda record: record["options"] is not None
        and len(record["options"]) > 0
        and filtering(record)
        and record["category_test"] == subset,
        hf_avail_splits=["dev", "test"],
        few_shots_split="dev",
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


def create_slovak_tests_text_input_task(prompt_name: str, subset: str):
    """Create a Slovak text input task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slovak_tests", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["questions"],
                "context": line["reference"],
                "answers": {
                    "text": [line["answer"]],
                },
            }
        ),
        hf_repo="slovak-nlp/slovak-school-exams-test-preprocessed",
        hf_revision="1d51f8205ee2341d11f05273d9c6d4ba3532a1df",
        hf_filter=lambda record: (record["options"] is None or len(record["options"]) == 0)
        and record["category_test"] == subset,
        hf_subset="default",
        hf_avail_splits=["dev", "test"],
        few_shots_split="dev",
        evaluation_splits=("test",),
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


TASKS_TABLE = [
    create_slovak_tests_multiple_choice_task(prompt_name, subset)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
    for subset in SUBSETS
] + [
    create_slovak_tests_text_input_task(prompt_name, subset)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_question_answering)
    for subset in SUBSETS
    if subset != "vyssie_studium"
]
