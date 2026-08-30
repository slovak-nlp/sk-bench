r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@article{financial_exams,
  title={Financial Question-answering Dataset for Slovak Language Model Evaluation},
  author={Hl{\'a}dek, Daniel and Sopkovi{\v{c}}, Kristi{\'a}n and Sta{\v{s}}, J{\'a}n and Sokolov{\'a}, Zuzana and Pleva, Mat{\'u}{\v{s}}},
  journal = {Journal of Linguistics/Jazykovedn\'{y} \v{c}asopis},
  volume={76},
  number={1},
  pages={247--257},
  year={2025},
  doi={10.2478/jazcas-2025-0022},
  publisher={De Gruyter Brill Sp. z oo, Paradigm Publishing Services}
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


def create_financial_exam_task(prompt_name: str):
    """Create a financial exam task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("financial_exams", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["prompt"],
                "choices": line["answers"],
                "gold_idx": line["label"],
            }
        ),
        hf_repo="TUKE-KEMT/slovak-financial-exam",
        hf_subset="default",
        hf_revision="7cdbfe635c3ed0d8b4df2ee04429d8afa651b33c",
        hf_avail_splits=["test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_completion)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_financial_exam_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_completion)
]
