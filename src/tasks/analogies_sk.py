r"""Task definitions.

If you use this task in your work, please cite the corresponding
paper(s):

@misc{blstak2025slovakconceptualdictionary,
      title={Slovak Conceptual Dictionary}, 
      author={Miroslav Blšták},
      year={2025},
      eprint={2512.00579},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.00579}, 
}
@misc{word_analogies_sk,
  title        = {{Word Analogies SK}: {S}lovak Word-Analogy Evaluation Set
                  from the Slovak Conceptual Dictionary},
  author       = {{Slovak NLP Community}},
  year         = {2025},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/slovak-nlp/word-analogies-sk},
  note         = {Derived from the Slovak Conceptual Dictionary
                  (arXiv:2512.00579)}
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
from src.metrics import normalized_exact_match, relaxed_normalized_exact_match
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


def create_analogies_task(prompt_name: str):
    """Create a word analogies task and register it with the task registry."""
    task = LightevalTaskConfig(
        name=make_task_name("analogies_sk", prompt_name),
        prompt_function=PROMPT_REGISTRY.get(prompt_name)(
            lambda line: {
                "concept1": line["concept1"],
                "concept2": line["concept2"],
                "concept3": line["concept3"],
                "choices": [line["answer"]],
                "gold_idx": 0,
            }
        ),
        hf_repo="slovak-nlp/word-analogies-sk",
        hf_subset="default",
        hf_revision="d6564d49c07b29ea93028f55a57c25eb231762f9",
        hf_avail_splits=["validation", "test"],
        evaluation_splits=("test",),
        few_shots_split="validation",
        generation_size=32,
        metrics=[relaxed_normalized_exact_match, normalized_exact_match],
        stop_sequence=["\n"],
    )
    TASK_REGISTRY.register(task, category=TaskCategory.c_analogy)
    return task


TASKS_TABLE = [
    create_analogies_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_analogy)
]
