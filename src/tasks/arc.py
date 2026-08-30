"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{arc2018,
      title={Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge}, 
      author={Peter Clark and Isaac Cowhey and Oren Etzioni and Tushar Khot and Ashish Sabharwal and Carissa Schoenick and Oyvind Tafjord},
      year={2018},
      eprint={1803.05457},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/1803.05457}, 
}
"""

from string import ascii_uppercase

from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


LETTER_INDICES = ascii_uppercase


def record_to_sample(record):
    query = record["question"].strip()
    target = record["answerKey"]
    choices = record["choices"]["text"]

    return Sample(input=query, target=target, choices=choices)


def create_arc_task(prompt_name: str):
    """Create an ARC task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("arc", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": line["choices"]["text"],
                "gold_idx": int(line["answerKey"]) - 1
                if line["answerKey"].isdigit()
                else LETTER_INDICES.index(line["answerKey"]),
            }
        ),
        hf_repo="jon-tow/okapi_arc_challenge",
        hf_subset="sk",
        hf_revision="56aa31aedeb55f0635cc7c3c13f7f86b276f9a4e",
        evaluation_splits=("test",),
        few_shots_split="train",
        generation_size=32,
        sample_fields=record_to_sample,
        solver=[multiple_choice(cache=True)],
        scorer=choice(),
        metrics=[
            relaxed_exact_match,
            Metrics.exact_match,
            conservative_exact_match,
        ],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_arc_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
