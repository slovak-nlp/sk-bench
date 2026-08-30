r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{autoskola,
  title  = {{Autoškola}: {S}lovak Driving-Licence Examination Item Bank},
  author = {{Slovak NLP Community}},
  year   = {2025},
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
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_driving_test_multiple_choice_task(prompt_name: str):
    """Create a Driving Test single choice task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("driving_test", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": line["options"],
                "gold_idx": line["options"].index(line["answer"]),
            }
        ),
        hf_repo="slovak-nlp/driving-test-dataset-slovak",
        hf_subset="default",
        hf_revision="37064890fbc00bfca59c035d38f52d265949361e",
        hf_filter=lambda record: record["image"] is None and record["answer"] in record["options"],
        hf_avail_splits=["dev", "test"],
        evaluation_splits=("test",),
        few_shots_split="dev",
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


TASKS_TABLE = [
    create_driving_test_multiple_choice_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
