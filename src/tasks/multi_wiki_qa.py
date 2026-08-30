"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{multiwikiqa2025,
      title={MultiWikiQA: A Reading Comprehension Benchmark in 300+ Languages}, 
      author={Dan Saattrup Smart},
      year={2026},
      eprint={2509.04111},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04111}, 
}
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language
from src.metrics.qa_metrics import MultilingualCustomExactMatchMetric, MultilingualCustomF1ScoreMetric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_multi_wiki_qa_task(prompt_name: str):
    """Create a Multi-WikiQA task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("multi_wiki_qa", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "question": line["question"],
                "context": line["context"],
                "answers": line["answers"],
            }
        ),
        hf_repo="alexandrainst/multi-wiki-qa",
        hf_subset="sk",
        hf_revision="9bdb3350487e399bd1cd873e1395290d636db5ce",
        evaluation_splits=("train",),
        generation_size=126,
        metrics=[
            MultilingualCustomExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualCustomF1ScoreMetric(Language.SLOVAK),
        ],
        stop_sequence=["\n"],
        hf_filter=lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_question_answering)
    return task


# Generate all tasks (single category)
TASKS_TABLE = [
    create_multi_wiki_qa_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_question_answering)
]
