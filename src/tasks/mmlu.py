"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{mmlu2021,
  title         = {Measuring Massive Multitask Language Understanding},
  author        = {Dan Hendrycks and Collin Burns and Steven Basart and
                   Andy Zou and Mantas Mazeika and Dawn Song and
                   Jacob Steinhardt},
  year          = {2020},
  eprint        = {2009.03300},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CY},
  url           = {https://arxiv.org/abs/2009.03300v3}
}
@inproceedings{okapi2023,
  author       = {Viet Dac Lai and
                  Chien Van Nguyen and
                  Nghia Trung Ngo and
                  Thuat Nguyen and
                  Franck Dernoncourt and
                  Ryan A. Rossi and
                  Thien Huu Nguyen},
  editor       = {Yansong Feng and
                  Els Lefever},
  title        = {Okapi: Instruction-tuned Large Language Models in Multiple Languages
                  with Reinforcement Learning from Human Feedback},
  booktitle    = {Proceedings of the 2023 Conference on Empirical Methods in Natural
                  Language Processing, {EMNLP} 2023 - System Demonstrations, Singapore,
                  December 6-10, 2023},
  pages        = {318--327},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.emnlp-demo.28},
  doi          = {10.18653/V1/2023.EMNLP-DEMO.28},
  timestamp    = {Wed, 13 Mar 2024 09:27:43 +0100},
  biburl       = {https://dblp.org/rec/conf/emnlp/LaiNNNDRN23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""

from functools import partial
from string import ascii_uppercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.multilingual.tasks.mlmm_mmlu import MMLU_SUBSETS
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def mmlu_filter(subset, line):
    return line["id"].split("/")[0] == subset


def create_mmlu_task(subset: str, prompt_name: str):
    """Create an MMLU task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("mmlu", prompt_name, subset),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": line["choices"],
                "gold_idx": ascii_uppercase.index(line["answer"])
                if isinstance(line["answer"], str)
                else line["answer"],
            }
        ),
        hf_repo="lighteval/okapi_mmlu",
        hf_subset="sk",
        hf_revision="b3e738ae24d5d96c4b425d4893f377451196cd2e",
        hf_filter=partial(mmlu_filter, subset),
        evaluation_splits=("test",),
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_mmlu_task(subset, prompt_name)
    for subset in MMLU_SUBSETS
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
