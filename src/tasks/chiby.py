r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@article{Garabik2026,
author  = {Radovan Garabík},
title   = {Slovenská Wikipédia ako zdroj dát pre korpus chýb},
journal = {Kultúra slova},
year    = {2026},
note    = {In print},
url     = {https://www.sav.sk/journals/uploads/0518105173-79-Slovenska-Wikipedia-ako-zdroj-dat-pre-korpus-chyb.pdf}
}

@misc{chiby,
  title  = {{Chiby}: A {S}lovak Grammaticality Classification Dataset},
  author = {{Slovak NLP Community}},
  year   = {2024},
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
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_chiby_task(prompt_name: str):
    """Create a Chiby task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("chiby", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {"sentence": line["sentence"], "gold_idx": line["label"]}
        ),
        hf_repo="slovak-nlp/chiby-classification",
        hf_subset="default",
        hf_revision="ce8533398057ace6da817758335633067648ca93",
        hf_avail_splits=["test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_chiby_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_chiby)
]
