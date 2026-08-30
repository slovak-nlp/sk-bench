from functools import partial
from typing import Callable

from lighteval.tasks.requests import Doc
from src.prompts.categories import PromptCategory
from src.prompts.prompt_registry import PROMPT_REGISTRY


def _get_nli_prompt_base(
    transform: Callable,
    instruction_template: str | None,
    query_template: str,
    choices: list[str],
) -> Callable[[dict, str | None], Doc]:
    """Base function for generating NLI prompt functions.

    Args:
        transform: A function that transforms a data sample into the format expected by the prompt
        instruction_template: Template for the instruction with placeholder for context
        query_template: Template for the query with placeholders for premise and hypothesis
        choices: List of choice texts

    Returns:
        A function that takes (line, task_name) and returns a Doc
    """

    def prompt_fn(line, task_name: str | None = None) -> Doc:
        transformed = transform(line)

        premise = transformed["premise"]
        hypothesis = transformed["hypothesis"]
        gold_idx = transformed["gold_idx"]

        query = query_template.format(sentence1=premise, sentence2=hypothesis)

        return Doc(
            task_name=task_name,
            query=query,
            choices=choices,
            gold_index=gold_idx,
            instruction=instruction_template,
        )

    return prompt_fn


# Register prompt variants dynamically
NLI_3_CLASSES_PROMPT_CONFIGS = {
    "prompt1_nli_3_classes_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": 'What is the relationship between the two sentences? Does the first sentence entail, contradict, or is it neutral towards the second sentence? Answer only "entailment", "contradiction", or "neutral".',
        "choices": ["Neutral", "Entailment", "Contradiction"],
    },
    "prompt1_nli_3_classes_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Aký je vzťah medzi dvomi vetami? Druhá veta z prvej vyplýva, protirečí jej alebo je neutrálna? Odpovedz iba slovom „vyplýva“, „protirečí“ alebo „neutrálny“.",
        "choices": ["Neutrálny", "Vyplýva", "Protirečí"],
    },
    "prompt2_nli_3_classes_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": 'I will write two sentences, which I call the premise and the hypothesis. The premise may entail the hypothesis or contradict it. If it neither entails nor contradicts the hypothesis, then it is neutral. Determine the relationship as "entailment", "contradiction" or "neutral".',
        "choices": ["Neutral", "Entailment", "Contradiction"],
    },
    "prompt2_nli_3_classes_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Napíšem dve vety, ktorým hovorím premisa a hypotéza. Hypotéza môže z premisy vyplývať, alebo jej protirečiť. Pokiaľ ani nevyplýva, ani neprotirečí, take je neutrálna. Vyhodnoť, či hypotéza z premisy „vyplýva“, alebo jej „protirečí“, alebo je vzťah „neutrálny“.",
        "choices": ["Neutrálny", "Vyplýva", "Protirečí"],
    },
    "prompt3_nli_3_classes_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": "Task: Determine whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).",
        "choices": ["Neutral", "Entailment", "Contradiction"],
    },
    "prompt3_nli_3_classes_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Úloha: Urči, či hypotéza vyplýva z premisy (vyplýva), či je s ňou v spore (protirečí) alebo ani jeden z týchto vzťahov (neutrálny).",
        "choices": ["Neutrálny", "Vyplýva", "Protirečí"],
    },
}

NLI_BINARY_PROMPT_CONFIGS = {
    "prompt1_nli_binary_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": 'What is the relationship between the two sentences? Does the first sentence entail or contradict the second sentence? Answer only with the word "entailment" or "contradiction".',
        "choices": ["Entailment", "Contradiction"],
    },
    "prompt1_nli_binary_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Aký je vzťah medzi nasledujúcimi dvomi vetami? Vyplýva druhá veta z prvej alebo jej protirečí? Odpovedz iba slovom „vyplýva“ alebo „protirečí“.",
        "choices": ["Vyplýva", "Protirečí"],
    },
    "prompt2_nli_binary_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": 'Determine whether the premise entails the hypothesis. Answer only "yes" or "no".',
        "choices": ["Yes", "No"],
    },
    "prompt2_nli_binary_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Urči, či hypotéza vyplýva z danej premisy. Odpovedz iba slovom „áno“ alebo „nie“.",
        "choices": ["Áno", "Nie"],
    },
    "prompt3_nli_binary_en": {
        "query_template": "Premise: {sentence1}\nHypothesis: {sentence2}\n\nAnswer:",
        "instruction_template": 'I will write two sentences, which I call the premise and the hypothesis. The premise may entail the hypothesis or contradict it. Determine the relationship between these sentences and answer with either "entailment" or "contradiction".',
        "choices": ["Entailment", "Contradiction"],
    },
    "prompt3_nli_binary_sk": {
        "query_template": "Premisa: {sentence1}\nHypotéza: {sentence2}\n\nOdpoveď:",
        "instruction_template": "Napíšem dve vety, ktorým hovorím premisa a hypotéza. Hypotéza môže z premisy buď vyplývať alebo jej protirečiť. Vyhodnoť vzťah týchto viet a odpovedz buď slovom „vyplýva“ alebo „protirečí“.",
        "choices": ["Vyplýva", "Protirečí"],
    },
}


for name, config in NLI_3_CLASSES_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_nli_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_nli_3class)(prompt_fn)


for name, config in NLI_BINARY_PROMPT_CONFIGS.items():
    prompt_fn = partial(_get_nli_prompt_base, **config)  # type: ignore[arg-type]
    PROMPT_REGISTRY.register(name, category=PromptCategory.p_nli_binary)(prompt_fn)
