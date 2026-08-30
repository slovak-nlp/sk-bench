"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{belebele2024,
  author       = {Lucas Bandarkar and
                  Davis Liang and
                  Benjamin Muller and
                  Mikel Artetxe and
                  Satya Narayan Shukla and
                  Donald Husa and
                  Naman Goyal and
                  Abhinandan Krishnan and
                  Luke Zettlemoyer and
                  Madian Khabsa},
  editor       = {Lun{-}Wei Ku and
                  Andre Martins and
                  Vivek Srikumar},
  title        = {The {B}elebele Benchmark: a Parallel Reading Comprehension Dataset
                  in 122 Language Variants},
  booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational
                  Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand,
                  August 11-16, 2024},
  pages        = {749--775},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://doi.org/10.18653/v1/2024.acl-long.44},
  doi          = {10.18653/V1/2024.ACL-LONG.44},
  timestamp    = {Wed, 28 Aug 2024 16:44:03 +0200},
  biburl       = {https://dblp.org/rec/conf/acl/BandarkarLMASHG24.bib},
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


def create_belebele_task(prompt_name: str):
    """Create a Belebele task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("belebele", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "context": line["flores_passage"],
                "choices": [line[f"mc_answer{i}"] for i in range(1, 5)],
                "gold_idx": int(line["correct_answer_num"]) - 1,
            },
        ),
        hf_repo="facebook/belebele",
        hf_subset="slk_Latn",
        hf_revision="7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
        evaluation_splits=("test",),
        hf_avail_splits=["test"],
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_belebele_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
