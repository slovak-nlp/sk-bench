"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{truthfulqa2022,
  author       = {Stephanie Lin and
                  Jacob Hilton and
                  Owain Evans},
  editor       = {Smaranda Muresan and
                  Preslav Nakov and
                  Aline Villavicencio},
  title        = {{TruthfulQA:} Measuring How Models Mimic Human Falsehoods},
  booktitle    = {Proceedings of the 60th Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2022, Dublin, Ireland,
                  May 22-27, 2022},
  pages        = {3214--3252},
  publisher    = {Association for Computational Linguistics},
  year         = {2022},
  url          = {https://doi.org/10.18653/v1/2022.acl-long.229},
  doi          = {10.18653/V1/2022.ACL-LONG.229},
  timestamp    = {Mon, 01 Aug 2022 16:27:53 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/LinHE22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
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


def create_truthfulqa_task(prompt_name: str):
    """Create a TruthfulQA task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("truthfulqa", prompt_name, "mc1"),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": line["mc1_targets_choices"],
                "gold_idx": line["mc1_targets_labels"].index(1),
            }
        ),
        hf_repo="alexandrainst/m_truthfulqa",
        hf_subset="sk",
        hf_revision="f0445d470f1925882b990f5f247fdcf288972f60",
        evaluation_splits=("val",),
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_truthfulqa_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
