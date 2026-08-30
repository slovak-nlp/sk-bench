r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{slohry,
  title        = {{SloHry}: {S}lovak Adjective--Noun Phrase-Completion
                  Benchmark},
  author       = {{Slovak NLP Community}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/slovak-nlp/slohry}
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


SUBSETS = ["high", "mid", "low"]


def create_declension_task(prompt_name: str, subset: str):
    """Create a declension task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slohry", prompt_name, f"declension_{subset}"),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "lemma": line["lemma"],
                "target_description": line["target_description"],
                "answer": line["answer"],
            }
        ),
        hf_repo="slovak-nlp/slohry",
        hf_subset="sklonovanie",
        hf_revision="3b47acd8a549c65ec6ab34aacd538f3ebcbfa32f",
        hf_filter=lambda x: x["freq_bucket"] == subset,
        hf_avail_splits=["val", "test"],
        evaluation_splits=("test",),
        few_shots_split="val",
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


def create_gradation_task(prompt_name: str, subset: str):
    """Create a gradation task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slohry", prompt_name, f"gradation_{subset}"),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "lemma": line["lemma"],
                "degree_label": line["degree_label"],
                "answer": line["answer"],
            }
        ),
        hf_repo="slovak-nlp/slohry",
        hf_subset="stupnovanie",
        hf_revision="3b47acd8a549c65ec6ab34aacd538f3ebcbfa32f",
        hf_filter=lambda x: x["freq_bucket"] == subset,
        hf_avail_splits=["val", "test"],
        evaluation_splits=("test",),
        few_shots_split="val",
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


def create_identity_form_task(prompt_name: str, subset: str):
    """Create an identity form task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slohry", prompt_name, f"identity_form_{subset}"),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "lemma": line["lemma"],
                "form": line["form"],
                "feature_description_sk": line["feature_description_sk"],
                "answer": line["answer"],
            }
        ),
        hf_repo="slovak-nlp/slohry",
        hf_subset="urcenie_tvaru",
        hf_revision="3b47acd8a549c65ec6ab34aacd538f3ebcbfa32f",
        hf_filter=lambda x: x["freq_bucket"] == subset,
        hf_avail_splits=["val", "test"],
        evaluation_splits=("test",),
        few_shots_split="val",
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
        version="1",
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


def create_inflection_task(prompt_name: str, subset: str):
    """Create an inflection task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slohry", prompt_name, f"inflection_{subset}"),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "lemma": line["lemma"],
                "target_description": line["target_description"],
                "answer": line["answer"],
            }
        ),
        hf_repo="slovak-nlp/slohry",
        hf_subset="casovanie",
        hf_revision="3b47acd8a549c65ec6ab34aacd538f3ebcbfa32f",
        hf_filter=lambda x: x["freq_bucket"] == subset,
        hf_avail_splits=["val", "test"],
        evaluation_splits=("test",),
        few_shots_split="val",
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_grammar)
    return task


TASKS_TABLE = (
    [
        create_declension_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_declension)
        for subset in SUBSETS
    ]
    + [
        create_gradation_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_gradation)
        for subset in SUBSETS
    ]
    + [
        create_identity_form_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_identity_form)
        for subset in SUBSETS
    ]
    + [
        create_inflection_task(prompt_name, subset)
        for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_inflection)
        for subset in SUBSETS
    ]
)
