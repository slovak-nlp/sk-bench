r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{ifeval_sk,
  title  = {{IFEval-SK}: A {S}lovak Adaptation of {M-IFEval} with a
            Re-implemented {S}lovak Instruction Registry},
  author = {{Slovak NLP Community}},
  year   = {2024},
  howpublished = {Hugging Face dataset (URL withheld for anonymous review);
                  adapted from the M-IFEval multilingual instruction-following
                  project},
}
@misc{ifeval2023,
  title         = {Instruction-Following Evaluation for Large Language Models},
  author        = {Jeffrey Zhou and Tianjian Lu and Swaroop Mishra and
                   Siddhartha Brahma and Sujoy Basu and Yi Luan and
                   Denny Zhou and Le Hou},
  year          = {2023},
  eprint        = {2311.07911},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CL},
  url           = {https://arxiv.org/abs/2311.07911v1}
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

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.imports import requires
from src.metrics import ifeval_sk_metrics, ifeval_sk_scorer
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


@requires("langdetect")
def record_to_sample(record):
    metadata = {"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]}
    return Sample(
        input=record["prompt"],
        metadata=metadata,
    )


def create_ifeval_task(prompt_name: str):
    """Create an ifeval_sk task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("ifeval_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name),
        hf_repo="slovak-nlp/ifeval-sk",
        hf_subset="default",
        hf_revision="2114c70fdcd22e8e3e5ea11f6b77067740959bf4",
        metrics=[ifeval_sk_metrics],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=4096,
        stop_sequence=[],
        version="0.1",
        sample_fields=record_to_sample,
        solver=[generate(cache=True)],
        scorer=ifeval_sk_scorer(),
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_ifeval)
    return task


TASKS_TABLE = [
    create_ifeval_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_ifeval)
]

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.imports import requires
from src.metrics import ifeval_sk_metrics, ifeval_sk_scorer
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


@requires("langdetect")
def record_to_sample(record):
    metadata = {"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"]}
    return Sample(
        input=record["prompt"],
        metadata=metadata,
    )


def create_ifeval_task(prompt_name: str):
    """Create an ifeval_sk task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("ifeval_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name),
        hf_repo="slovak-nlp/ifeval-sk",
        hf_subset="default",
        hf_revision="2114c70fdcd22e8e3e5ea11f6b77067740959bf4",
        metrics=[ifeval_sk_metrics],
        hf_avail_splits=["train"],
        evaluation_splits=["train"],
        few_shots_split="train",
        few_shots_select="random_sampling",
        generation_size=4096,
        stop_sequence=[],
        version="0.1",
        sample_fields=record_to_sample,
        solver=[generate(cache=True)],
        scorer=ifeval_sk_scorer(),
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_ifeval)
    return task


TASKS_TABLE = [
    create_ifeval_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_ifeval)
]
