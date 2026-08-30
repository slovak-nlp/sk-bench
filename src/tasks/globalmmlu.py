r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{globalmmlu2024,
    title = "Global {MMLU}: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation",
    author = "Singh, Shivalika  and
      Romanou, Angelika  and
      Fourrier, Cl{\'e}mentine  and
      Adelani, David Ifeoluwa  and
      Ngui, Jian Gang  and
      Vila-Suero, Daniel  and
      Limkonchotiwat, Peerat  and
      Marchisio, Kelly  and
      Leong, Wei Qi  and
      Susanto, Yosephine  and
      Ng, Raymond  and
      Longpre, Shayne  and
      Ruder, Sebastian  and
      Ko, Wei-Yin  and
      Bosselut, Antoine  and
      Oh, Alice  and
      Martins, Andre  and
      Choshen, Leshem  and
      Ippolito, Daphne  and
      Ferrante, Enzo  and
      Fadaee, Marzieh  and
      Ermis, Beyza  and
      Hooker, Sara",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.919/",
    doi = "10.18653/v1/2025.acl-long.919",
    pages = "18761--18799",
    ISBN = "979-8-89176-251-0",
    abstract = "Reliable multilingual evaluation is difficult, and culturally appropriate evaluation is even harder to achieve.A common practice to fill this gap is to machine-translate English evaluation sets. However, translation introduces language bias and carries over cultural and regional assumptions from the original questions {--} often testing knowledge irrelevant to the target audience. In this work, we highlight the extent and impact of these biases and present a multilingual evaluation framework that aims to mitigate them through improved translations and annotation practices.Through a large-scale study involving professional and community translators and annotators, we show that state-of-the-art models excel primarily by learning Western-centric concepts. Notably, we find that model rankings on the full MMLU change when evaluated on a subset of questions explicitly marked as culturally sensitive.We release Global MMLU, a multilingual extension of MMLU across 42 languages, featuring improved translation quality, expanded language coverage, and designated subsets labeled as culturally sensitive and culturally agnostic to enable a more comprehensive and equitable benchmark for evaluating language models across diverse linguistic and cultural contexts."
}
"""

from string import ascii_lowercase

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_exact_match, relaxed_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_globalmmlu_task(prompt_name: str):
    """Create an Global-MMLU-Lite task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("globalmmlu-lite", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "choices": [line[f"option_{option}"] for option in ascii_lowercase[:4]],
                "gold_idx": ascii_lowercase.index(line["answer"].lower()),
            }
        ),
        hf_repo="CohereLabs/Global-MMLU-Lite",
        hf_subset="sk",
        hf_revision="600af1bff2d447c7fc3aee22f11f607560d5fcb7",
        evaluation_splits=("test",),
        few_shots_split="dev",
        few_shots_select=None,
        generation_size=32,
        metrics=[relaxed_exact_match, Metrics.exact_match, conservative_exact_match],
        stop_sequence=["\n"],
        version="1"
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_multi_choice)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_globalmmlu_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_multi_choice)
]
