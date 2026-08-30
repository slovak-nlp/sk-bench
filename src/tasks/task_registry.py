"""
Task Registry for tracking tasks by category.
Tasks are registered manually with their category.
"""

from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.tasks.categories import TaskCategory


class TaskRegistry:
    def __init__(self):
        self.registry: dict[str, LightevalTaskConfig] = {}
        self.categories: dict[TaskCategory, set[str]] = {}

    def register(self, task: LightevalTaskConfig, category: TaskCategory):
        """Register a task with a specific category."""
        task_name = task.name
        self.registry[task_name] = task

        if category not in self.categories:
            self.categories[category] = set()
        self.categories[category].add(task_name)

    def get(self, task_name: str):
        """Get task config by name."""
        return self.registry.get(task_name)

    def list_tasks(self):
        """List all registered task names."""
        return list(self.registry.keys())

    def get_category_tasks(self, category: str) -> list[str]:
        """Get all task names belonging to a category."""
        category_enum = TaskCategory(category)  # Convert string to TaskCategory enum
        return list(self.categories.get(category_enum, set()))

    def get_categories(self):
        """Get all registered categories as list of strings."""
        return [category.value for category in self.categories.keys()]

    def register_tasks_table(self, tasks_table: list[LightevalTaskConfig], category: TaskCategory):
        """Register all tasks from a TASKS_TABLE with a specific category."""
        for task in tasks_table:
            self.register(task, category)


# Initialize the global singleton instance
TASK_REGISTRY = TaskRegistry()
