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


def create_senti_sk_task_binary(prompt_name: str):
    """Create a SentiSK task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("senti_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "text": line["text"],
                # if the label is "negative", we consider it negative sentiment (gold_idx=0), if it's "positive", we consider it positive sentiment (gold_idx=1)
                "gold_idx": 0 if line["label"] == "negative" else 1,
            }
        ),
        hf_repo="TUKE-KEMT/senti-sk",
        hf_subset="default",
        hf_revision="3c8a92168ff2b692c653de65c97bda573127d15d",
        # filter out neutral class
        hf_filter=lambda line: line["label"] != "neutral",
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_sentiment)
    return task


def create_senti_sk_task_3class(prompt_name: str):
    """Create a SentiSK task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("senti_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "text": line["text"],
                # if the label is "negative", we consider it negative sentiment (gold_idx=0), if it's "positive", we consider it positive sentiment (gold_idx=2), if it's "neutral", we consider it neutral sentiment (gold_idx=1)
                "gold_idx": 0 if line["label"] == "negative" else 2 if line["label"] == "positive" else 1,
            }
        ),
        hf_repo="TUKE-KEMT/senti-sk",
        hf_subset="default",
        hf_revision="3c8a92168ff2b692c653de65c97bda573127d15d",
        hf_avail_splits=["train", "test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_sentiment)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_senti_sk_task_binary(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_sentiment_binary)
] + [
    create_senti_sk_task_3class(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_sentiment_3class)
]
