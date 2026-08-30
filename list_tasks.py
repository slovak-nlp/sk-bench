from src.tasks.task_registry import TASK_REGISTRY


tasks_list = TASK_REGISTRY.list_tasks()
for task in tasks_list:
    print(task)
