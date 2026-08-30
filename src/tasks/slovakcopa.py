"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{ondrejova-suppa-2024-llms,
    title = {Can {LLM}s Handle Low-Resource Dialects? A Case Study on Translation and Common Sense Reasoning in {Š}ari{š}},
    author = {Ondrejová, Viktória and Šuppa, Marek},
    editor = {Scherrer, Yves and Jauhiainen, Tommi and Ljubešić, Nikola and Zampieri, Marcos and Nakov, Preslav and Tiedemann, Jörg},
    booktitle = {Proceedings of the Eleventh Workshop on NLP for Similar Languages, Varieties, and Dialects (VarDial 2024)},
    month = jun,
    year = {2024},
    address = {Mexico City, Mexico},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2024.vardial-1.11/},
    doi = {10.18653/v1/2024.vardial-1.11},
    pages = {130--139}
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


def create_slovakcopa_task(prompt_name: str):
    """Create an SlovakCoPA task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("slovakcopa", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "premise": line["premise_sk"],
                "choices": [line["choice1_sk"], line["choice2_sk"]],
                "gold_idx": line["label"],
            }
        ),
        hf_repo="NaiveNeuron/SlovakCOPA",
        hf_subset="default",
        hf_revision="5c61e4cd9343f9b4fab623bdc50bb895cc170771",
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_copa)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_slovakcopa_task(prompt_name) for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_copa)
]
