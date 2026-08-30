r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{skj1,
  title  = {{SKJ1}: {S}lovak Grammar Correction Multi-Choice Set},
  author = {{Marek Dobe\v{s}, Peter Bedn\'{a}r}},
  year   = {2026},
  howpublished = {Hugging Face dataset},
  note         = {Accessed: 2026-04-20},
  url = {https://huggingface.co/datasets/slovak-nlp/SKJ1}
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


def create_skj1_task(prompt_name: str):
    """Create a SKJ1 task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("skj1", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "sentence": line["chybna_veta"],
                "choices": [line[option] for option in ["A", "B", "C", "D", "E"]],
                "gold_idx": ["A", "B", "C", "D", "E"].index(line["spravna_moznost"]),
            }
        ),
        hf_repo="slovak-nlp/SKJ1",
        hf_subset="default",
        hf_revision="e615c60b415389b629703f82d3a79810dbde6734",
        hf_avail_splits=["train"],
        evaluation_splits=("train",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


TASKS_TABLE = [
    create_skj1_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_skj1)
]
