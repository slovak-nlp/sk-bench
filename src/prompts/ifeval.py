from lighteval.tasks.requests import Doc
from lighteval.utils.imports import requires
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


@requires("langdetect")
@PROMPT_REGISTRY.register("ifeval", category=PromptCategory.p_ifeval)
def ifeval_prompt_fn(line, task_name: str | None = None) -> Doc:
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={"instructions_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )
