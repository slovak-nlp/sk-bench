r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@article{toxic_sk,
author = {Sokolov\'{a}, Zuzana and Harahus, Maro\v{s} and Hl\'{a}dek, Daniel and Sta\v{s}, J\'{a}n},
year = {2025},
month = {11},
pages = {279-289},
title = {Annotated Slovak Datasets for Toxicity, Hate Speech, and Sentiment Analysis},
volume = {76},
journal = {Journal of Linguistics/Jazykovedn\'{y} \v{c}asopis},
doi = {10.2478/jazcas-2025-0025}
}
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_toxic_sk_task(prompt_name: str):
    """Create a ToxicSK task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("toxic_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {"text": line["text"], "gold_idx": line["label"]}
        ),
        hf_repo="TUKE-KEMT/toxic-sk",
        hf_subset="default",
        hf_revision="2fb416e18785c128e7ec934c07a14c803a88cdb3",
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_toxicity)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_toxic_sk_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_toxicity)
]
