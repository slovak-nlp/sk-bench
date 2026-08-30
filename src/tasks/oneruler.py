r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{oneruler2025,
  author       = {Yekyung Kim and Jenna Russell and Marzena Karpinska and
                  Mohit Iyyer},
  title        = {One Ruler to Measure Them All: Benchmarking Multilingual
                  Long-Context Language Models},
  year         = {2025},
  eprint       = {2503.01996},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2503.01996}
}
@misc{onerulersk,
  title  = {{OneRuler-SK}: Long-Context {S}lovak Reading Comprehension Suite},
  author = {{Slovak NLP Community}},
  year   = {2025},
  howpublished = {Hugging Face dataset (URL withheld for anonymous review)},
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

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import oneruler_metric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


SUBSETS = [
    "cwe_4096",
    "cwe_8192",
    "cwe_16384",
    "niah_multikey_4096",
    "niah_multikey_8192",
    "niah_multikey_16384",
    "niah_multiquery_4096",
    "niah_multiquery_8192",
    "niah_multiquery_16384",
    "niah_multivalue_4096",
    "niah_multivalue_8192",
    "niah_multivalue_16384",
    "niah_none_4096",
    "niah_none_8192",
    "niah_none_16384",
    "niah_single_4096",
    "niah_single_8192",
    "niah_single_16384",
]


def create_oneruler_task(prompt_name: str, subtask: str):
    """Create an Slovak Oneruler task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("oneruler_sk", prompt_name, subtask),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "outputs": line["outputs"],
                "context": line["context"],
                "query1": line["query1"],
                "query2": line["query2"],
                "gold_idx": list(range(len(line["outputs"]))),
                "task": line["task"],
                "language": "sk" if "slovak" in prompt_name else "en",
            }
        ),
        hf_repo="slovak-nlp/oneruler-sk-processed",
        hf_subset=subtask,
        hf_revision="c3a160bc7ce2dada86803f26ee699ca75ee26e41",
        hf_avail_splits=["test"],
        evaluation_splits=("test",),
        generation_size=512,
        metrics=[oneruler_metric],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_oneruler)
    return task


TASKS_TABLE = [
    create_oneruler_task(prompt_name, subtask)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_oneruler)
    for subtask in SUBSETS
]
