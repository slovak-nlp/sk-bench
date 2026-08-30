r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{slovak_trivia,
  title        = {{Slovak Trivia}: {S}lovak Open-Domain Trivia
                  Question-Answering Dataset},
  author       = {{Slovak NLP Community}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/slovak-nlp/slovak_trivia}
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
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def get_gold_indices(options, single_choice=False):
    """Get the indices of the correct options from the list of options."""
    if isinstance(options, str):
        options = json.loads(options)

    gold_indices = []
    for i, option in enumerate(options):
        if option["is_correct"]:
            gold_indices.append(i)

    if single_choice:
        if len(gold_indices) != 1:
            raise ValueError("Expected exactly one correct option for single choice question")
        return gold_indices[0]

    return gold_indices


def create_slovak_trivia_task(prompt_name: str):
    """Create a Slovak Trivia single choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slovak_trivia", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": [option["text"] for option in line["options"]],
                "gold_idx": get_gold_indices(line["options"], single_choice=True),
            }
        ),
        hf_repo="slovak-nlp/slovak_trivia",
        hf_subset="default",
        hf_revision="2f27ff5a3c266a379c164d3cb43fc9cf8e45b37a",
        hf_avail_splits=["test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


TASKS_TABLE = [
    create_slovak_trivia_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
