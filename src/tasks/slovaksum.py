r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{slovaksum2024,
  author       = {Viktoria Ondrejov{\'{a}} and
                  Marek Suppa},
  editor       = {Nicoletta Calzolari and
                  Min{-}Yen Kan and
                  V{\'{e}}ronique Hoste and
                  Alessandro Lenci and
                  Sakriani Sakti and
                  Nianwen Xue},
  title        = {{SlovakSum:} {A} Large Scale {S}lovak Summarization Dataset},
  booktitle    = {Proceedings of the 2024 Joint International Conference on Computational
                  Linguistics, Language Resources and Evaluation, {LREC/COLING} 2024,
                  20-25 May, 2024, Torino, Italy},
  pages        = {14916--14922},
  publisher    = {{ELRA} and {ICCL}},
  year         = {2024},
  url          = {https://aclanthology.org/2024.lrec-main.1298},
  timestamp    = {Mon, 03 Jun 2024 14:33:48 +0200},
  biburl       = {https://dblp.org/rec/conf/coling/OndrejovaS24.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
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

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import meteor, rouge1, rouge2, rougeL, chrf, chrf_plus
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_slovaksum_task(prompt_name: str):
    """Create a slovaksum task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slovaksum", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "text": line["text"],
                "summary": line["sum"],
            }
        ),
        hf_repo="slovak-nlp/slovaksum-downsampled",
        hf_subset="default",
        hf_revision="2e4cdabe972ac040114701ea9e885e0a648a33fb",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        generation_size=256,
        metrics=[
            rouge1,
            rouge2,
            rougeL,
            chrf,
            chrf_plus,
            meteor,
        ],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_summarization)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_slovaksum_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_summarization)
]
