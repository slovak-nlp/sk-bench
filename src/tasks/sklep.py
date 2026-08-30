r"""Task definitions for the slovak-nlp/sklep benchmark dataset.

Available subsets: 'hate-speech', 'sentiment-analysis', 'ner-wikigoldsk',
'ner-under', 'pos', 'question-answering', 'rte', 'nli', 'sts'

If you use this task in your work, please cite the corresponding
paper(s):

@inproceedings{sklep2025,
  author       = {Marek Suppa and
                  Andrej Ridzik and
                  Daniel Hl{\'{a}}dek and
                  Tom{\'{a}}{\v{s}} Jav{\r{u}}rek and
                  Viktoria Ondrejov{\'{a}} and
                  Krist{\'{\i}}na S{\'{a}}sikov{\'{a}} and
                  Marek T{\"{u}}rk and
                  Mari{\'{a}}n {\v{S}}imko},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {{skLEP:} {A} {S}lovak General Language Understanding Benchmark},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL}
                  2025, Vienna, Austria, July 27 - August 1, 2025},
  pages        = {26716--26743},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.findings-acl.1371/},
  timestamp    = {Sun, 09 Nov 2025 22:49:13 +0100},
  biburl       = {https://dblp.org/rec/conf/acl/SuppaRHJOSTS25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

The skLEP benchmark reuses earlier datasets; when using the
corresponding subsets, please also cite their sources:

Subset `sentiment-analysis` (CSFD Slovak movie reviews, via the
`slovak_sa` dataset):

@inproceedings{pecar-simko-2019-sentiment,
  author       = {Samuel Pecar and
                  Mari{\'{a}}n Simko and
                  M{\'{a}}ria Bielikov{\'{a}}},
  editor       = {Tomaz Erjavec and
                  Michal Marcinczuk and
                  Preslav Nakov and
                  Jakub Piskorski and
                  Lidia Pivovarova and
                  Jan Snajder and
                  Josef Steinberger and
                  Roman Yangarber},
  title        = {Improving Sentiment Classification in {S}lovak Language},
  booktitle    = {Proceedings of the 7th Workshop on Balto-Slavic Natural Language
                  Processing, BSNLP@ACL 2019, Florence, Italy, August 2, 2019},
  pages        = {114--119},
  publisher    = {Association for Computational Linguistics},
  year         = {2019},
  url          = {https://doi.org/10.18653/v1/W19-3716},
  doi          = {10.18653/V1/W19-3716},
  timestamp    = {Wed, 07 Dec 2022 23:13:53 +0100},
  biburl       = {https://dblp.org/rec/conf/acl-bsnlp/PecarSB19.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
Subsets `ner-wikigoldsk` and `ner-under`/`ner` (WikiGoldSK and the
WikiGoldSK silver corpus annotated following UniversalNER/UD_Slovak-SNK):

@inproceedings{wikigoldsk2023,
    title = "{W}iki{G}old{SK}: Annotated Dataset, Baselines and Few-Shot Learning Experiments for {S}lovak Named Entity Recognition",
    author = "Suba, David  and
      Suppa, Marek  and
      Kubik, Jozef  and
      Hamerlik, Endre  and
      Takac, Martin",
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
    url = "https://aclanthology.org/2023.bsnlp-1.16/",
    doi = "10.18653/v1/2023.bsnlp-1.16",
    pages = "138--145",
    abstract = "Named Entity Recognition (NER) is a fundamental NLP tasks with a wide range of practical applications. The performance of state-of-the-art NER methods depends on high quality manually anotated datasets which still do not exist for some languages. In this work we aim to remedy this situation in Slovak by introducing WikiGoldSK, the first sizable human labelled Slovak NER dataset. We benchmark it by evaluating state-of-the-art multilingual Pretrained Language Models and comparing it to the existing silver-standard Slovak NER dataset. We also conduct few-shot experiments and show that training on a sliver-standard dataset yields better results. To enable future work that can be based on Slovak NER, we release the dataset, code, as well as the trained models publicly under permissible licensing terms."
}
Subset `question-answering` (skquad):

@article{skquad2023,
  author       = {Daniel Hl{\'{a}}dek and
                  J{\'{a}}n Sta{\v{s}} and
                  Jozef Juh{\'{a}}r and
                  Tom{\'{a}}{\v{s}} Koct{\'{u}}r},
  title        = {{S}lovak Dataset for Multilingual Question Answering},
  journal      = {{IEEE} Access},
  volume       = {11},
  pages        = {32869--32881},
  year         = {2023},
  doi          = {10.1109/ACCESS.2023.3262308},
  url          = {https://doi.org/10.1109/ACCESS.2023.3262308}
}
Subset `nli` (SK-XNLI, translated from XNLI):

@inproceedings{xnli2018,
  author       = {Alexis Conneau and
                  Ruty Rinott and
                  Guillaume Lample and
                  Adina Williams and
                  Samuel R. Bowman and
                  Holger Schwenk and
                  Veselin Stoyanov},
  editor       = {Ellen Riloff and
                  David Chiang and
                  Julia Hockenmaier and
                  Jun'ichi Tsujii},
  title        = {{XNLI:} Evaluating Cross-lingual Sentence Representations},
  booktitle    = {Proceedings of the 2018 Conference on Empirical Methods in Natural
                  Language Processing, Brussels, Belgium, October 31 - November 4,
                  2018},
  pages        = {2475--2485},
  publisher    = {Association for Computational Linguistics},
  year         = {2018},
  url          = {https://doi.org/10.18653/v1/d18-1269},
  doi          = {10.18653/V1/D18-1269},
  timestamp    = {Fri, 06 Aug 2021 00:41:32 +0200},
  biburl       = {https://dblp.org/rec/conf/emnlp/ConneauRLWBSS18.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

Subset `rte` (translated PASCAL RTE challenge data):

@inproceedings{dagan2005pascal,
  author    = {Ido Dagan and Oren Glickman and Bernardo Magnini},
  title     = {The {PASCAL} Recognising Textual Entailment Challenge},
  booktitle = {Machine Learning Challenges, Evaluating Predictive Uncertainty,
               Visual Object Classification and Recognizing Textual Entailment,
               First {PASCAL} Machine Learning Challenges Workshop, {MLCW} 2005,
               Southampton, UK, April 11-13, 2005, Revised Selected Papers},
  series    = {Lecture Notes in Computer Science},
  volume    = {3944},
  pages     = {177--190},
  publisher = {Springer},
  year      = {2005},
  doi       = {10.1007/11736790_9}
}

Subset `sts` (translated STS Benchmark):

@inproceedings{cer2017semeval,
  author    = {Daniel Cer and Mona Diab and Eneko Agirre and
               I{\~{n}}igo Lopez{-}Gazpio and Lucia Specia},
  title     = {{SemEval}-2017 Task 1: Semantic Textual Similarity Multilingual
               and Crosslingual Focused Evaluation},
  booktitle = {Proceedings of the 11th International Workshop on Semantic
               Evaluation, SemEval@ACL 2017, Vancouver, Canada, August 3-4, 2017},
  pages     = {1--14},
  publisher = {Association for Computational Linguistics},
  year      = {2017},
  doi       = {10.18653/V1/S17-2001}
}

Subset `hate-speech` (Slovak hate-speech Facebook comments; TUKE-KEMT
`hate_speech_slovak` dataset):

@article{toxic_sk,
  author  = {Sokolov{\'a}, Zuzana and Harahus, Maro{\v{s}} and Hl{\'a}dek, Daniel and Sta{\v{s}}, J{\'a}n},
  title   = {Annotated Slovak Datasets for Toxicity, Hate Speech, and Sentiment Analysis},
  journal = {Journal of Linguistics/Jazykovedn{\'y} {\v{c}}asopis},
  volume  = {76},
  pages   = {279--289},
  year    = {2025},
  doi     = {10.2478/jazcas-2025-0025}
}

Subsets `pos` and the silver part of the NER data (UD_Slovak-SNK /
UNER_Slovak-SNK, Universal Dependencies):

@misc{ud_slovak_snk,
  author       = {Daniel Zeman and Martin Popel and Milan Straka and
                  Jan Haji{\v{c}} and Joakim Nivre and Filip Ginter and
                  others},
  title        = {{UD} {S}lovak-{SNK} ({Universal} {Dependencies} treebank)},
  year         = {2017},
  howpublished = {\url{https://github.com/UniversalDependencies/UD_Slovak-SNK}}
}

@inproceedings{piskorski-etal-2024-uner,
  author    = {Stephen Mayhew and Terra Blevins and Shuheng Liu and
               Marek {\v{S}}uppa and Hila Gonen and Joseph Marvin Imperial and
               B{\"o}rje F. Karlsson and others},
  title     = {{Universal NER}: {A} Gold-Standard Multilingual Named Entity
               Recognition Benchmark},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter
               of the Association for Computational Linguistics: Human Language
               Technologies (Volume 1: Long Papers), {NAACL} 2024},
  pages     = {4322--4337},
  publisher = {Association for Computational Linguistics},
  year      = {2024},
  url       = {https://aclanthology.org/2024.naacl-long.243/}
}
"""

from typing import Callable

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.metrics.qa_metrics import MultilingualCustomExactMatchMetric, MultilingualCustomF1ScoreMetric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


# Constants
HF_REPO = "slovak-nlp/sklep"


TASK_CONFIGS = [
    # Classification tasks with label column
    {
        "name": "sentiment-analysis",
        "task_category": TaskCategory.c_sentiment,
        "prompts": [
            PromptCategory.p_sentiment_binary,
        ],
        "columns": {"text": "text", "label": "label"},
        "label_map": {0: 0, 1: 1},  # 0=negative, 1=positive
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    {
        "name": "hate-speech",
        "task_category": TaskCategory.c_hate_speech,
        "prompts": [
            PromptCategory.p_hate_speech,
        ],
        "columns": {"text": "text", "label": "label"},
        "label_map": {0: 0, 1: 1},  # 0=no hate, 1=hate
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # NLI tasks with premise/hypothesis
    {
        "name": "nli",
        "task_category": TaskCategory.c_nli,
        "prompts": [
            PromptCategory.p_nli_3class,
        ],
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        # 1=neutral, 0=entailment, 2=contradiction
        "label_map": {1: 0, 0: 1, 2: 2},
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # RTE tasks with text1/text2
    {
        "name": "rte",
        "task_category": TaskCategory.c_nli,
        "prompts": [
            PromptCategory.p_nli_binary,
        ],
        "columns": {"premise": "text1", "hypothesis": "text2", "label": "label"},
        "label_map": {1: 1, 0: 0},  # 1=non-entailment, 0=entailment
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # QA tasks
    {
        "name": "question-answering",
        "task_category": TaskCategory.c_question_answering,
        "prompts": [
            PromptCategory.p_question_answering,
        ],
        "columns": {"context": "context", "question": "question", "answers": "answers"},
        "generation_size": 126,
        "metrics": [
            MultilingualCustomExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualCustomF1ScoreMetric(Language.SLOVAK),
        ],
        # "hf_filter": lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    },
]


def make_transform(columns_map: dict, label_map: dict | None = None) -> Callable:
    """Generate a transform function from column mapping.

    Args:
        columns_map: Dict mapping expected_key -> dataset_column_name
        label_map: Optional dict mapping raw_label_value -> gold_idx.
                   If None, assumes label is already an integer index.

    Returns:
        Function that transforms a row to expected format
    """

    def transform(line: dict) -> dict:
        result = {expected_key: line[dataset_col] for expected_key, dataset_col in columns_map.items()}
        if "label" in result and label_map is not None:
            result["gold_idx"] = label_map.get(result["label"], result["label"])

        # Support no answer for QA
        if "question" in result:
            result["unknown"] = True

        return result

    return transform


def create_sklep_task(config: dict, prompt_name: str) -> LightevalTaskConfig:
    """Create a single sklep task from config."""
    subset = config["name"]

    transform = make_transform(config["columns"], config.get("label_map"))
    prompt_fn = PROMPT_REGISTRY.get(prompt_name)(transform)

    return LightevalTaskConfig(
        name=make_task_name("sklep", prompt_name, subset),
        prompt_function=prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=subset,
        hf_revision="10549a8c63542e6a0db4fcf5fcdc29b3e1b8c4e9",
        metrics=config["metrics"],
        generation_size=config["generation_size"],
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        stop_sequence=["\n"],
        hf_filter=config.get("hf_filter"),
        version=0,
    )


TASKS_TABLE = []

for config in TASK_CONFIGS:  # type: ignore
    task_category = config["task_category"]
    for prompt in config["prompts"]:  # type: ignore
        if isinstance(prompt, PromptCategory):
            for prompt_name in PROMPT_REGISTRY.get_category_prompts(prompt):  # type: ignore
                task = create_sklep_task(config, prompt_name)
                TASKS_TABLE.append(task)
                TASK_REGISTRY.register(task, category=task_category)  # type: ignore
        else:
            task = create_sklep_task(config, prompt)  # prompt needs to be name of the registered prompt function
            TASKS_TABLE.append(task)
            TASK_REGISTRY.register(task, category=task_category)  # type: ignore

from typing import Callable

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.utils.language import Language
from src.metrics import conservative_norm_exact_match, normalized_exact_match, relaxed_normalized_exact_match
from src.metrics.qa_metrics import MultilingualCustomExactMatchMetric, MultilingualCustomF1ScoreMetric
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY
from src.utils import make_task_name


# Constants
HF_REPO = "slovak-nlp/sklep"


TASK_CONFIGS = [
    # Classification tasks with label column
    {
        "name": "sentiment-analysis",
        "task_category": TaskCategory.c_sentiment,
        "prompts": [
            PromptCategory.p_sentiment_binary,
        ],
        "columns": {"text": "text", "label": "label"},
        "label_map": {0: 0, 1: 1},  # 0=negative, 1=positive
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    {
        "name": "hate-speech",
        "task_category": TaskCategory.c_hate_speech,
        "prompts": [
            PromptCategory.p_hate_speech,
        ],
        "columns": {"text": "text", "label": "label"},
        "label_map": {0: 0, 1: 1},  # 0=no hate, 1=hate
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # NLI tasks with premise/hypothesis
    {
        "name": "nli",
        "task_category": TaskCategory.c_nli,
        "prompts": [
            PromptCategory.p_nli_3class,
        ],
        "columns": {"premise": "premise", "hypothesis": "hypothesis", "label": "label"},
        # 1=neutral, 0=entailment, 2=contradiction
        "label_map": {1: 0, 0: 1, 2: 2},
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # RTE tasks with text1/text2
    {
        "name": "rte",
        "task_category": TaskCategory.c_nli,
        "prompts": [
            PromptCategory.p_nli_binary,
        ],
        "columns": {"premise": "text1", "hypothesis": "text2", "label": "label"},
        "label_map": {1: 1, 0: 0},  # 1=non-entailment, 0=entailment
        "generation_size": 32,
        "metrics": [relaxed_normalized_exact_match, normalized_exact_match, conservative_norm_exact_match],
    },
    # QA tasks
    {
        "name": "question-answering",
        "task_category": TaskCategory.c_question_answering,
        "prompts": [
            PromptCategory.p_question_answering,
        ],
        "columns": {"context": "context", "question": "question", "answers": "answers"},
        "generation_size": 126,
        "metrics": [
            MultilingualCustomExactMatchMetric(Language.SLOVAK, "prefix"),
            MultilingualCustomF1ScoreMetric(Language.SLOVAK),
        ],
        # "hf_filter": lambda line: any(ans for ans in line["answers"]["text"] if len(ans) > 0),
    },
]


def make_transform(columns_map: dict, label_map: dict | None = None) -> Callable:
    """Generate a transform function from column mapping.

    Args:
        columns_map: Dict mapping expected_key -> dataset_column_name
        label_map: Optional dict mapping raw_label_value -> gold_idx.
                   If None, assumes label is already an integer index.

    Returns:
        Function that transforms a row to expected format
    """

    def transform(line: dict) -> dict:
        result = {expected_key: line[dataset_col] for expected_key, dataset_col in columns_map.items()}
        if "label" in result and label_map is not None:
            result["gold_idx"] = label_map.get(result["label"], result["label"])

        # Support no answer for QA
        if "question" in result:
            result["unknown"] = True

        return result

    return transform


def create_sklep_task(config: dict, prompt_name: str) -> LightevalTaskConfig:
    """Create a single sklep task from config."""
    subset = config["name"]

    transform = make_transform(config["columns"], config.get("label_map"))
    prompt_fn = PROMPT_REGISTRY.get(prompt_name)(transform)

    return LightevalTaskConfig(
        name=make_task_name("sklep", prompt_name, subset),
        prompt_function=prompt_fn,
        hf_repo=HF_REPO,
        hf_subset=subset,
        hf_revision="10549a8c63542e6a0db4fcf5fcdc29b3e1b8c4e9",
        metrics=config["metrics"],
        generation_size=config["generation_size"],
        hf_avail_splits=["train", "validation", "test"],
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        stop_sequence=["\n"],
        hf_filter=config.get("hf_filter"),
        version=0,
    )


TASKS_TABLE = []

for config in TASK_CONFIGS:  # type: ignore
    task_category = config["task_category"]
    for prompt in config["prompts"]:  # type: ignore
        if isinstance(prompt, PromptCategory):
            for prompt_name in PROMPT_REGISTRY.get_category_prompts(prompt):  # type: ignore
                task = create_sklep_task(config, prompt_name)
                TASKS_TABLE.append(task)
                TASK_REGISTRY.register(task, category=task_category)  # type: ignore
        else:
            task = create_sklep_task(config, prompt)  # prompt needs to be name of the registered prompt function
            TASKS_TABLE.append(task)
            TASK_REGISTRY.register(task, category=task_category)  # type: ignore
