# Contributing to SK-Bench

## Adding a New Custom Task

This guide explains how to add a new custom task to the SK-Bench benchmark.

### File Structure

```
src/tasks/
├── your_new_task.py          # Task definitions

src/prompts/
├── your_prompt.py            # Prompt definitions (optional)
```

---

## Creating a New Prompt

### Step 1: Create Prompt File

Create `src/prompts/your_prompt.py`. For creating multiple prompt variants from a base function:

```python
from functools import partial
from typing import Callable
from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_your_prompt_base(
    transform: Callable,
    query_template: str,
    # Add your custom parameters here
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating prompt functions."""

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        # Build your query using the template
        query = query_template.format(question=transformed["question"])

        # Return Doc object
        return Doc(
            task_name=task_name,
            query=query,
            choices=transformed["choices"],
            gold_index=transformed["gold_idx"],
        )

    return prompt_fn


# Register prompt variants dynamically
# See src/prompts/categories.py for available prompt categories (p_*, e.g., p_sentiment_binary)
YOUR_PROMPT_CONFIGS = {
    "your_prompt_sk": {
        "query_template": "Nasleduje otázka. Odpovedz:\n\nOtázka: {question}",
    },
    "your_prompt_en": {
        "query_template": "Answer the following question:\n\nQuestion: {question}",
    },
}

for name, config in YOUR_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_your_prompt_base, **config)
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_your_category)(prompt_fn)
```

### Step 2: Add to Prompts Init

Update `src/prompts/__init__.py` to import your prompt module:

```python
from src.prompts import your_prompt  # Import to trigger registration

__all__ = [
    "PROMPT_REGISTRY",
    # ... existing items ...
    "your_prompt",
]
```

---

## Creating a New Task

### Step 1: Create Task File

Create `src/tasks/your_new_task.py`:

```python
"""Task definitions for your dataset."""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from src.prompts.categories import PromptCategory
from src.prompts import PROMPT_REGISTRY
from src.tasks.categories import TaskCategory
from src.tasks.task_registry import TASK_REGISTRY

from src.utils import make_task_name


def create_your_task(prompt_name: str):
    """Create a task and register it with the task registry.

    Args:
        prompt_name: Name of the prompt function from PROMPT_REGISTRY

    Returns:
        LightevalTaskConfig object
    """
    # Get the prompt function from registry and apply transform
    # Transform should return a dict with the expected keys (e.g., question, choices, gold_idx)
    prompt_fn = PROMPT_REGISTRY.get(prompt_name)(lambda line: {
        "question": line["question"],
        "choices": line["choices"],
        "gold_idx": line["answer"],
    })

    # Create the task configuration
    task = LightevalTaskConfig(
        name=make_task_name("your_dataset", prompt_name),
        prompt_function=prompt_fn,
        hf_repo="your-org/your-dataset",
        hf_subset="default",
        evaluation_splits=["test"],
        few_shots_split="train",
        few_shots_select="random_sampling_from_train",
        generation_size=256,
        metrics=[Metrics.exact_match],
        stop_sequence=["\n"],
    )

    # Register the task with its category
    # See src/tasks/categories.py for available categories (c_*, e.g., c_sentiment)
    TASK_REGISTRY.register(task, category=TaskCategory.c_your_category)

    return task


# Generate all tasks for this dataset
TASKS_TABLE = [
    create_your_task(prompt_name)
    for prompt_name in PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_your_category)
]
```

### Step 2: Import in Tasks Init

Update `src/tasks/__init__.py`:

```python
from src.tasks.your_new_task import TASKS_TABLE as YOUR_NEW_TASK_TABLE

TASKS_TABLE = [
    # ... existing tasks ...
    *YOUR_NEW_TASK_TABLE,
]
```

---

## Supported Categories

Task categories use the `c_` prefix. Prompt categories use the `p_` prefix. Add new tasks to the appropriate category in `src/tasks/categories.py`.

The tables below list the categories currently defined. The authoritative
sources are `src/tasks/categories.py` (`TaskCategory`) and
`src/prompts/categories.py` (`PromptCategory`) — add new categories there.

| Task Category | Description |
|---------------|-------------|
| `c_analogy` | Analogy tasks |
| `c_completion` | Completion tasks |
| `c_copa` | COPA (causal reasoning) tasks |
| `c_grammar` | Grammar / morphology tasks |
| `c_hate_speech` | Hate speech detection tasks |
| `c_ifeval` | Instruction-following (IFEval) tasks |
| `c_multi_choice` | Multi-choice tasks |
| `c_nli` | Natural language inference and RTE tasks |
| `c_question_answering` | Question answering tasks |
| `c_sentiment` | Sentiment analysis tasks |
| `c_summarization` | Summarization tasks |
| `c_toxicity` | Toxicity detection tasks |
| `c_oneruler` | OneRuler (long-context) tasks |

| Prompt Category | Description |
|-----------------|-------------|
| `p_analogy` | Analogy prompts |
| `p_chiby` | Chiby prompts |
| `p_completion` | Completion prompts |
| `p_copa` | COPA (causal reasoning) prompts |
| `p_declension` | Declension prompts |
| `p_gradation` | Gradation prompts |
| `p_hate_speech` | Hate speech detection prompts |
| `p_identity_form` | Identity-form prompts |
| `p_ifeval` | Instruction-following (IFEval) prompts |
| `p_inflection` | Inflection prompts |
| `p_multi_choice` | Multi-choice prompts |
| `p_multi_select` | Multi-select prompts |
| `p_nli_binary` | Binary NLI/RTE prompts |
| `p_nli_3class` | 3-class NLI prompts |
| `p_question_answering` | Question answering prompts |
| `p_sentiment_binary` | Binary sentiment prompts |
| `p_sentiment_3class` | 3-class sentiment prompts |
| `p_skj1` | SKJ1 prompts |
| `p_summarization` | Summarization prompts |
| `p_toxicity` | Toxicity detection prompts |
| `p_oneruler` | OneRuler (long-context) prompts |

---

## Task Configuration Reference

### Required Parameters

```python
LightevalTaskConfig(
    name=make_task_name("your_dataset", prompt_name),
    prompt_function=prompt_fn,
    hf_repo="your-org/your-dataset",
    hf_subset="default",
    evaluation_splits=["test"],
    metrics=[Metrics.exact_match],
)
```

### Optional Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `hf_filter` | Filter function for data | `lambda line: line["label"] == "positive"` |
| `few_shots_split` | Split to use for few-shots | `"train"` |
| `few_shots_select` | Strategy for selecting few-shots | `"random_sampling_from_train"` |
| `generation_size` | Max tokens to generate | `256` |
| `stop_sequence` | Stop generation on these strings | `["\n"]` |
| `hf_avail_splits` | Available splits in dataset | `["train", "validation", "test"]` |

---

## Prompt Function Requirements

Your prompt function should return either:
1. A string (simple case)
2. A `lighteval.tasks.requests.Doc` object (complex case)

### Simple Prompt (String)

```python
def get_simple_prompt(transform: callable):
    def prompt_fn(line, task_name: str | None = None):
        data = transform(line)
        return f"Question: {data['question']}\nAnswer:"
    return prompt_fn
```

### Complex Prompt (Doc)

```python
from lighteval.tasks.requests import Doc

def get_complex_prompt(transform: callable):
    def prompt_fn(line, task_name: str | None = None):
        data = transform(line)
        return Doc(
            task_name=task_name,
            query=data["question"],
            choices=data["choices"],
            gold_index=data["gold_idx"],
        )
    return prompt_fn
```

---

## Testing Your Task

Verify your task is registered correctly:

```bash
python3 -c "
from src.tasks.task_registry import TASK_REGISTRY
from src.tasks import TASKS_TABLE
from src.prompts import PROMPT_REGISTRY
from src.prompts.categories import PromptCategory

# Check your category
print('Tasks in your_category:', TASK_REGISTRY.get_category_tasks('c_your_category'))

# Check prompt categories
print('Prompts in p_your_category:', PROMPT_REGISTRY.get_category_prompts(PromptCategory.p_your_category))

# Check total tasks
print('Total tasks:', len(TASKS_TABLE))

# List all categories
print('All categories:', list(TASK_REGISTRY.get_categories()))
"
```

---

## Adding Tasks to Benchmark

Task lists are plain-text files (e.g. `tasks_list_zero.txt`, `tasks_list_few.txt`,
or any `*.txt` under a task folder). Each non-empty, non-comment line is one task
name, passed through to the lighteval pipeline. Lines starting with `#` are
ignored. The name format is produced by `make_task_name`:

```
# Format: {dataset}.{prompt_name}[:{subset}][|N]
#   :subset  -> optional HF subset (only when the task defines one)
#   |N       -> optional number of few-shots (omit for zero-shot)

# Zero-shot, no subset
toxic_sk.prompt1_toxicity_sk

# With a subset
mmlu.prompt1_mqa_single_correct_sk:anatomy

# With N few-shots
senti_sk.prompt1_sentiment_binary_sk|3

# Subset + few-shots
scio.prompt2_mqa_single_correct_en:bio|3
```

Point the eval script at a file with `--tasks-file`, e.g.:

```bash
uv run python -m eval_litellm --model-name "<model>" --tasks-file tasks_list_zero.txt
```

There is no category (`c_*`) or `all` shorthand — every task must be listed
explicitly. To enumerate registered task names, run `python list_tasks.py`.
