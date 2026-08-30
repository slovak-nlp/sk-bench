import numpy as np
from inspect_ai.scorer import Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetricGrouping
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from src.metrics.ifeval_sk import sk_instructions_registry
from src.metrics.utils import clean_model_output


submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


REASONING_TAG_PAIRS = [
    ("<think>", "</think>"),
]


def _preprocess_response(response: str) -> list[str]:
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    return all_responses


class IFEvalSKMetrics(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> dict:
        response = model_response.final_text[0]

        response = clean_model_output(response)
        instruction_list = doc.specific["instructions_id_list"]
        all_kwargs = doc.specific["kwargs"]
        prompt = doc.query

        all_responses = _preprocess_response(response)

        is_following_list_strict = []
        is_following_list_loose = []

        for index, instruction_id in enumerate(instruction_list):
            # remove sk: from the instruction id to get the instruction class name
            if instruction_id.startswith("sk:"):
                instruction_id = instruction_id.replace("sk:", "")
            instruction_cls = sk_instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
            instruction.build_description(**task_kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            # Strict
            if response.strip() and instruction.check_following(response):
                is_following_list_strict.append(True)
            else:
                is_following_list_strict.append(False)

            # Loose
            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list_loose.append(is_following)

        return {
            "prompt_level_strict_acc": int(all(is_following_list_strict)),
            "inst_level_strict_acc": is_following_list_strict,
            "prompt_level_loose_acc": int(all(is_following_list_loose)),
            "inst_level_loose_acc": is_following_list_loose,
        }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc


ifeval_sk_metrics = SampleLevelMetricGrouping(
    metric_name=submetric_names,
    higher_is_better=dict.fromkeys(submetric_names, True),
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=IFEvalSKMetrics(),
    corpus_level_fn={
        "prompt_level_strict_acc": np.mean,
        "inst_level_strict_acc": agg_inst_level_acc,
        "prompt_level_loose_acc": np.mean,
        "inst_level_loose_acc": agg_inst_level_acc,
    },
)


@scorer(
    metrics={
        "prompt_level_strict_acc": [accuracy(), stderr()],
        "prompt_level_loose_acc": [accuracy(), stderr()],
    }
)
def ifeval_sk_scorer():
    async def score(state: TaskState, target: Target):
        response = state.output.completion
        instruction_list = state.metadata["instruction_id_list"]
        all_kwargs = state.metadata["kwargs"]
        prompt = state.input
        response = clean_model_output(response)
        all_responses = _preprocess_response(response)

        is_following_list_strict = []
        is_following_list_loose = []
        for index, instruction_id in enumerate(instruction_list):
            if instruction_id.startswith("sk:"):
                instruction_id = instruction_id.replace("sk:", "")
            instruction_cls = sk_instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
            instruction.build_description(**task_kwargs)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)
            # Strict
            if response.strip() and instruction.check_following(response):
                is_following_list_strict.append(True)
            else:
                is_following_list_strict.append(False)
            # Loose
            is_following = False
            for r in all_responses:
                if r.strip() and instruction.check_following(r):
                    is_following = True
                    break

            is_following_list_loose.append(is_following)
        return Score(
            value={
                "prompt_level_strict_acc": int(all(is_following_list_strict)),
                "prompt_level_loose_acc": int(all(is_following_list_loose)),
            },
            explanation=str(instruction_list),
        )

    return score
