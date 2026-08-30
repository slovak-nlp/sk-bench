"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{nli_sk,
  title  = {{NLI-SK-annotated}: A Native {S}lovak Natural Language
            Inference Dataset},
  author = {{Natália Kňažeková and Marek Šuppa}},
  year   = {2024},
  howpublished = {Hugging Face dataset},
  note         = {Accessed: 2026-02-20},
  url = {https://huggingface.co/datasets/natalia-nk/NLI-SK-annotated}
}
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_nli_sk_task(prompt_name: str):
    """Create a NLI SK task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("nli_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "premise": line["Premise"],
                "hypothesis": line["Hypothesis"],
                "gold_idx": {"Neutral": 0, "Entailment": 1, "Contradiction": 2}[line["Label"]],
            }
        ),
        hf_repo="natalia-nk/NLI-SK-annotated",
        hf_filter=lambda line: line["Label"]
        in [
            "Entailment",
            "Contradiction",
            "Neutral",
        ],  # Filter out Other category
        hf_subset="default",
        hf_revision="79914f425b59d8b9fabb6d38c37f6d81f9723f46",
        hf_avail_splits=["test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_nli)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_nli_sk_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_nli_3class)
]
