def make_task_name(dataset: str, prompt_name: str, subset: str | None = None) -> str:
    """
    Create a standardized task name.

    Format: {dataset}.{prompt_name}[:{subset}]
    - Uses '.' to separate dataset from prompt
    - Uses ':' to separate from subset (only if subset is not 'default' or empty)

    Args:
        dataset: Dataset name (e.g., 'smesum', 'sklep', 'arc')
        prompt_name: Prompt variant name (e.g., 'summarization_sk')
        subset: Optional subset name (e.g., 'sentiment-analysis')

    Returns:
        Standardized task name
    """
    if subset and subset != "default":
        return f"{dataset}.{prompt_name}:{subset}"
    return f"{dataset}.{prompt_name}"


def create_temp_tasks_file(task_configs: list[tuple[list[str], int | None]], temp_file_path: str):
    with open(temp_file_path, "w") as f:
        for task_names, few_shots in task_configs:
            for task_name in task_names:
                if few_shots is not None:
                    f.write(f"{task_name}|{few_shots}\n")
                else:
                    f.write(f"{task_name}\n")
