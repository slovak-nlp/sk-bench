r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{scio,
  title        = {{Scio testy}: {S}lovak Secondary-School Admission Test
                  Question Bank},
  author       = {{Slovak NLP Community}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/slovak-nlp/scio-testy}
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

import re

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


SUBSETS = ["aj", "mat", "bio", "che", "vsp"]


def filtering(line, subset):
    return line["source_file"].lower().startswith(subset)



def process_options(options):
    return [re.sub(r"^\([A-Z]\) ", "", option) for option in options]


def create_scio_task(prompt_name: str, subset: str):
    """Create a Scio multiple choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("scio", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "context": line["reference"] if line.get("reference") not in (None, "", "None") else None,
                "choices": process_options(line["options"]),
                "gold_idx": line["options"].index(line["answer"]),
            }
        ),
        hf_repo="slovak-nlp/scio-testy",
        hf_subset="default",
        hf_revision="78c67eeae5d48b92590945a1c589b7730bfce8df",
        hf_filter=lambda record: filtering(record, subset),
        hf_avail_splits=["dev", "test"],
        few_shots_split="dev",
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


TASKS_TABLE = [
    create_scio_task(prompt_name, subset)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
    for subset in SUBSETS
]
