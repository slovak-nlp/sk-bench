r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{csfd,
    title = "Resources and Few-shot Learners for In-context Learning in {S}lavic Languages",
    author = "{\v{S}}tef{\'a}nik, Michal  and
      Kadl{\v{c}}{\'i}k, Marek  and
      Gramacki, Piotr  and
      Sojka, Petr",
    editor = "Piskorski, Jakub  and
      Marci{\'n}czuk, Micha{\l}  and
      Nakov, Preslav  and
      Ogrodniczuk, Maciej  and
      Pollak, Senja  and
      P{\v{r}}ib{\'a}{\v{n}}, Pavel  and
      Rybak, Piotr  and
      Steinberger, Josef  and
      Yangarber, Roman",
    booktitle = "Proceedings of the 9th Workshop on Slavic Natural Language Processing 2023 (SlavicNLP 2023)",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.bsnlp-1.12/",
    doi = "10.18653/v1/2023.bsnlp-1.12",
    pages = "94--105",
    abstract = "Despite the rapid recent progress in creating accurate and compact in-context learners, most recent work focuses on in-context learning (ICL) for tasks in English. However, the ability to interact with users of languages outside English presents a great potential for broadening the applicability of language technologies to non-English speakers. In this work, we collect the infrastructure necessary for training and evaluation of ICL in a selection of Slavic languages: Czech, Polish, and Russian. We link a diverse set of datasets and cast these into a unified instructional format through a set of transformations and newly-crafted templates written purely in target languages. Using the newly-curated dataset, we evaluate a set of the most recent in-context learners and compare their results to the supervised baselines. Finally, we train, evaluate and publish a set of in-context learning models that we train on the collected resources and compare their performance to previous work. We find that ICL models tuned in English are also able to learn some tasks from non-English contexts, but multilingual instruction fine-tuning consistently improves the ICL ability. We also find that the massive multitask training can be outperformed by single-task training in the target language, uncovering the potential for specializing in-context learners to the language(s) of their application."
}
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_csfd_task(prompt_name: str):
    """Create a CSFD task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("movie_review", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "text": line["text"],
                "gold_idx": 0 if line["label"] == "negative" else 1,
            }
        ),
        hf_repo="mteb/slovak_movie_review_sentiment",
        hf_subset="default",
        hf_revision="29a7405aabcfd4860a51ae6f65a5650d63108f26",
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=("test",),
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_sentiment)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_csfd_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_sentiment_binary)
]
