import os


def strip_wrappers(text: str) -> str:
    """Return the content after known model-wrapper markers (`</think>`,
    `assistantfinal`). Does NOT truncate to one line — callers that need the
    full body for marker scanning use this rather than `clean_model_output`."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # MiniMax M3 uses a different marker for the end of reasoning, so we also check for that.
    if "</mm:think>" in text:
        text = text.split("</mm:think>")[-1].strip()
    if "assistantfinal" in text:
        text = text.split("assistantfinal", 1)[-1].strip()
    # Muse-Glimmer models emit channel-delimited output: reasoning in the
    # `to=self` channel, final answer after `assistant to=user`.
    if "assistant to=user" in text:
        text = text.split("assistant to=user")[-1].strip()

    return text


def clean_model_output(text: str) -> str:
    """Extract a single answer line: strip wrappers, then (unless thinking
    mode is enabled) keep only the first non-empty line. Marker extraction
    MUST run first, or `<think>...\\n</think>\\n<answer>` collapses to the
    `<think>` line."""
    text = strip_wrappers(text)
    REASONING_EFFORT = os.environ.get("REASONING_EFFORT", None)
    if (REASONING_EFFORT is None or REASONING_EFFORT.lower() == "none") and "\n" in text:
        for line in text.split("\n"):
            line = line.strip()
            if line:
                return line
        return ""
    return text
