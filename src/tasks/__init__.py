import src.prompts  # ensure prompts are registered before importing task modules
from src.tasks.analogies_sk import TASKS_TABLE as ANALOGIES_SK_TASKS_TABLE
from src.tasks.arc import TASKS_TABLE as ARC_TASKS_TABLE
from src.tasks.belebele import TASKS_TABLE as BELEBELE_TASKS_TABLE
from src.tasks.categories import TaskCategory
from src.tasks.chiby import TASKS_TABLE as CHIBY_TASKS_TABLE
from src.tasks.csfd import TASKS_TABLE as CSFD_TASKS_TABLE
from src.tasks.driving_tests import TASKS_TABLE as DRIVING_TESTS_TASKS_TABLE
from src.tasks.e_tests import TASKS_TABLE as E_TESTY_TASKS_TABLE
from src.tasks.financial_exams import TASKS_TABLE as FINANCIAL_EXAMS_TASKS_TABLE
from src.tasks.ifeval import TASKS_TABLE as IFEVALSK_TASKS_TABLE
from src.tasks.mmlu import TASKS_TABLE as MMLU_TASKS_TABLE
from src.tasks.multi_wiki_qa import TASKS_TABLE as MULTI_WIKI_QA_TASKS_TABLE
from src.tasks.nli_sk import TASKS_TABLE as NLI_SK_TASKS_TABLE
from src.tasks.oneruler import TASKS_TABLE as ONERULER_TASKS_TABLE
from src.tasks.scio import TASKS_TABLE as SCIO_TASKS_TABLE
from src.tasks.senti_sk import TASKS_TABLE as SENTI_SK_TASKS_TABLE
from src.tasks.skj1 import TASKS_TABLE as SKJ1_TASKS_TABLE
from src.tasks.sklep import TASKS_TABLE as SKLEP_TASKS_TABLE
from src.tasks.slohry import TASKS_TABLE as SLOHRY_TASKS_TABLE
from src.tasks.slovak_tests import TASKS_TABLE as SLOVAK_TESTS_TASKS_TABLE
from src.tasks.slovak_trivia import TASKS_TABLE as SLOVAK_TRIVIA_TASKS_TABLE
from src.tasks.slovakcopa import TASKS_TABLE as SLOVAKCOPA_TASKS_TABLE
from src.tasks.slovaksum import TASKS_TABLE as SLOVAKSUM_TASKS_TABLE
from src.tasks.smesum import TASKS_TABLE as SMESUM_TASKS_TABLE
from src.tasks.toxic_sk import TASKS_TABLE as TOXIC_SK_TASKS_TABLE
from src.tasks.truthfulqa import TASKS_TABLE as TRUTHFULQA_TASKS_TABLE
from src.tasks.globalmmlu import TASKS_TABLE as GLOBAL_MMLU_TASKS_TABLE


TASKS_TABLE = [
    *ANALOGIES_SK_TASKS_TABLE,
    *ARC_TASKS_TABLE,
    *BELEBELE_TASKS_TABLE,
    *CHIBY_TASKS_TABLE,
    *CSFD_TASKS_TABLE,
    *DRIVING_TESTS_TASKS_TABLE,
    *E_TESTY_TASKS_TABLE,
    *FINANCIAL_EXAMS_TASKS_TABLE,
    *IFEVALSK_TASKS_TABLE,
    *MMLU_TASKS_TABLE,
    *MULTI_WIKI_QA_TASKS_TABLE,
    *NLI_SK_TASKS_TABLE,
    *ONERULER_TASKS_TABLE,
    *SCIO_TASKS_TABLE,
    *SENTI_SK_TASKS_TABLE,
    *SKJ1_TASKS_TABLE,
    *SKLEP_TASKS_TABLE,
    *SLOHRY_TASKS_TABLE,
    *SLOVAK_TESTS_TASKS_TABLE,
    *SLOVAK_TRIVIA_TASKS_TABLE,
    *SLOVAKCOPA_TASKS_TABLE,
    *SLOVAKSUM_TASKS_TABLE,
    *SMESUM_TASKS_TABLE,
    *TOXIC_SK_TASKS_TABLE,
    *TRUTHFULQA_TASKS_TABLE,
    *GLOBAL_MMLU_TASKS_TABLE,
]


__all__ = [
    "TASKS_TABLE",
    "TaskCategory",
    "src",
]
