from typing import Callable

from src.prompts.categories import PromptCategory


class PromptRegistry:
    def __init__(self, prompts: dict | None = None):
        self.registry: dict[str, Callable] = {}
        self.categories: dict[PromptCategory, set[str]] = {}  # category -> set of prompt names
        if prompts:
            self.register_multiple(prompts)

    # Modified register method to act as a decorator
    def register(self, name: str, prompt_function=None, category: PromptCategory | None = None):
        # If prompt_function is provided, use standard registration
        if prompt_function is not None:
            self.registry[name] = prompt_function
            if category:
                if category not in self.categories:
                    self.categories[category] = set()
                self.categories[category].add(name)

        # If prompt_function is None, return a decorator
        else:

            def decorator(func):
                self.registry[name] = func
                if category:
                    if category not in self.categories:
                        self.categories[category] = set()
                    self.categories[category].add(name)
                return func

            return decorator

    def get(self, name: str):
        return self.registry.get(name)

    def list_prompts(self):
        return list(self.registry.keys())

    def get_category_prompts(self, category: PromptCategory) -> list[str]:
        """Get all prompt names belonging to a category."""
        return list(self.categories.get(category, set()))

    def register_multiple(self, prompts: dict):
        for name, prompt_function in prompts.items():
            self.register(name, prompt_function)


# Initialize the global singleton instance here
PROMPT_REGISTRY = PromptRegistry()
