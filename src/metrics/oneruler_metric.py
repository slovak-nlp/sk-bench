import re
import unicodedata

import numpy as np

from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from src.metrics.utils import clean_model_output


none_dict = {
    "sk": ["žiadny", "žiadna", "žiadne", "žiaden"],
    "en": ["none"],
}


def clean_text(text):
    return text.strip().lower().replace("\u200c", "").replace(" ", "")


def compare_none(correct_answer, model_answer, language) -> bool:
    processed_model_answer = clean_text(unicodedata.normalize("NFKC", model_answer))
    correct_answer = [clean_text(answer) for answer in correct_answer]
    none_words = [clean_text(word) for word in none_dict[language]]

    # Step 2: Remove single digit numbers from the processed answer.
    processed_model_answer = re.sub(r"\b\d\b", "", processed_model_answer)

    # Step 3: Extract all numeric sequences from the processed answer.
    # numbers = re.findall(r"\d+", processed_model_answer)

    # If any multi-digit number is found, return "0".
    # if numbers:
    #     return False
    # Step 4: Check if any of the words in none_words are present.
    for word in none_words:
        if word in processed_model_answer:
            return True

    # If none of the none_words are found, return "0".
    return False


def is_correct_order(correct_answer, model_answer) -> bool:
    model_answer = model_answer.strip().lower()
    correct_answer = [correct.strip().lower() for correct in correct_answer]

    return all(
        f"{word}," in model_answer
        or f"{word}\n" in model_answer
        or f"{word} " in model_answer
        or model_answer.endswith(word)
        for word in correct_answer
    )


def compare_numbers(correct_answer, model_answer, language) -> bool:
    if not model_answer:
        return False
    processed_model_answer = unicodedata.normalize("NFKC", model_answer)

    none_words = none_dict[language]
    # Step 1.5: Check if any word in none_words is present in the processed answer; if yes, auto-fail.
    for word in none_words:
        if word in processed_model_answer or clean_text(word) in processed_model_answer:
            return False

    # Step 2: Extract all numeric substrings from the processed answer.
    numeric_strings = re.findall(r"\d+", processed_model_answer)

    # Step 3: Remove numbers that consist of a single digit.
    numeric_strings = [num for num in numeric_strings if len(num) > 1]

    # Step 4: Remove duplicates while preserving the original order.
    numeric_strings = list(dict.fromkeys(numeric_strings))

    # If no numerics are found after processing, return "0".
    if not numeric_strings:
        return False

    # Step 5: Convert the extracted number strings to integers.
    try:
        extracted_numbers = [int(num) for num in numeric_strings]
    except Exception:
        return False

    # Convert correct_answers elements to integers to ensure numeric comparison.
    try:
        correct_converted = [int(item) for item in correct_answer]
    except Exception:
        return False

    # Step 6: Check that the number of extracted numbers matches the length of correct_answers.
    if len(extracted_numbers) != len(correct_converted):
        return False

    # Step 7: Compare the extracted numbers with the correct answers element-wise.
    if set(extracted_numbers) == set(correct_converted):
        return True
    else:
        return False


def _extract_tagged_text(text: str) -> str:
    pattern = re.compile(r"<(?:Odpoveď|Answer)>(.*?)</(?:Odpoveď|Answer)>", re.S)

    match = pattern.search(text)
    # remove Tu uveď slová from the output string
    if match:
        text = match.group(1).strip()
    else:
        text = text.strip()

    text = text.replace("Tu uveď slová", "")
    text = text.replace("Tu uveď všetky čísla", "")
    text = text.replace("List all numbers here", "")
    text = text.replace("List the words here", "")
    return f"{text.strip()},"


class OnerulerMetric(SampleLevelComputation):
    def __init__(self):
        pass

    def compute(self, doc: Doc, model_response: ModelResponse) -> int:
        # check the length because model_response.final_text is a list of strings, but we expect only one string as the final answer. If there are multiple strings, we can not be sure which one is the final answer, so we will return 0 in that case.
        if len(model_response.final_text) != 1:
            return 0

        response = model_response.final_text[0].strip()
        reference = doc.get_golds()

        response = clean_model_output(response)
        hypothesis = _extract_tagged_text(response)

        if "niah_none" in doc.specific["task"]:
            acc = compare_none(reference, hypothesis, doc.specific["language"])
        elif "cwe" in doc.specific["task"]:
            acc = is_correct_order(reference, hypothesis)
        else:
            acc = compare_numbers(reference, hypothesis, doc.specific["language"])

        return int(acc)


oneruler_metric = SampleLevelMetric(
    metric_name="oneruler_accuracy",
    sample_level_fn=OnerulerMetric(),
    category=SamplingMethod.GENERATIVE,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
