r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{smesum2020,
  author       = {Marek Suppa and
                  Jergus Adamec},
  editor       = {Nicoletta Calzolari and
                  Fr{\'{e}}d{\'{e}}ric B{\'{e}}chet and
                  Philippe Blache and
                  Khalid Choukri and
                  Christopher Cieri and
                  Thierry Declerck and
                  Sara Goggi and
                  Hitoshi Isahara and
                  Bente Maegaard and
                  Joseph Mariani and
                  H{\'{e}}l{\`{e}}ne Mazo and
                  Asuncion Moreno and
                  Jan Odijk and
                  Stelios Piperidis},
  title        = {A Summarization Dataset of {S}lovak News Articles},
  booktitle    = {Proceedings of The 12th Language Resources and Evaluation Conference,
                  {LREC} 2020, Marseille, France, May 11-16, 2020},
  pages        = {6725--6730},
  publisher    = {European Language Resources Association},
  year         = {2020},
  url          = {https://aclanthology.org/2020.lrec-1.830/},
  timestamp    = {Sat, 30 Sep 2023 09:46:30 +0200},
  biburl       = {https://dblp.org/rec/conf/lrec/SuppaA20.bib},
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


def create_smesum_task(prompt_name: str):
    """Create a smesum task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("smesum", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "text": line["document"],
                "summary": line["title"],
            }
        ),
        hf_repo="slovak-nlp/smesum-downsampled",
        hf_subset="default",
        hf_revision="cb427bdc96e444637508fb982915f706c19d81d4",
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
    create_smesum_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_summarization)
]
