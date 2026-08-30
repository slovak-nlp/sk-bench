"""
Override for lighteval's PromptManager._prepare_chat_template method.

This module patches the PromptManager to use a custom chat template implementation.
"""

import os

from lighteval.tasks.requests import Doc


def patch_prompt_manager_chat_template():
    """
    Patch PromptManager._prepare_chat_template with custom implementation.

    This patches the lighteval's PromptManager class to use a custom
    _prepare_chat_template method.
    """
    try:
        from lighteval.tasks.prompt_manager import PromptManager

        def patched_prepare_chat_template(self, doc: Doc, tokenize: bool = True):
            """Patched version of _prepare_chat_template.

            Few-shot examples are embedded within the first user message as text,
            not as separate assistant turns.

            Args:
                self: The PromptManager instance
                doc: The document containing query, instruction, and fewshot_samples
                tokenize: Whether to apply the tokenizer (for local models) or return raw messages (for APIs)

            Returns:
                str | list[dict[str, str]]: Formatted chat template string (if tokenize=True) or
                                             list of messages (if tokenize=False)
            """
            messages = []

            # Add system prompt if available
            if self.system_prompt is not None:
                messages.append({"role": "system", "content": self.system_prompt})

            # Build the first user message with few-shot examples embedded as text
            first_user_content = ""

            # Add few-shot examples as text within the first user message
            for fewshot_sample in doc.fewshot_samples:
                query = _extract_query(fewshot_sample.query, fewshot_sample.instruction)
                gold = fewshot_sample.get_golds()[0]
                first_user_content += f"{query} {gold}\n\n"

            # Add instruction to the first user message if available
            if doc.instruction is not None:
                first_user_content = doc.instruction + "\n\n" + first_user_content

            # Add the main query to the first user message
            main_query = _extract_query(doc.query, doc.instruction)
            first_user_content += main_query

            # Add the combined first user message
            messages.append({"role": "user", "content": first_user_content})

            if tokenize:  # for local models
                assert self.tokenizer is not None, "Tokenizer must be set for chat template formatting."

                reasoning_effort = os.environ.get("REASONING_EFFORT", None)
                extra_kwargs = {}
                if reasoning_effort is None or reasoning_effort.lower() == "none":
                    extra_kwargs["reasoning_effort"] = "none"
                    extra_kwargs["enable_thinking"] = False
                    extra_kwargs["thinking"] = False
                    extra_kwargs["reasoning_strength"] = "none"
                else:
                    extra_kwargs["reasoning_effort"] = reasoning_effort
                    extra_kwargs["enable_thinking"] = True
                    extra_kwargs["thinking"] = True
                    extra_kwargs["thinking_mode"] = reasoning_effort
                    extra_kwargs["reasoning_strength"] = reasoning_effort

                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )

            else:  # for apis
                return messages

        # Apply the patch
        PromptManager._prepare_chat_template = patched_prepare_chat_template
        print("Successfully patched PromptManager._prepare_chat_template.")
        return True
    except Exception as e:
        print(f"Warning: Failed to patch PromptManager._prepare_chat_template: {e}")
        return False


def _extract_query(query: str, instruction: str | None) -> str:
    """Extract query content, removing instruction prefix if appropriate.

    Args:
        query: The query string which may include the instruction as a prefix
        instruction: The instruction string that might be prefixed to the query

    Returns:
        str: The extracted query content without instruction prefix if it was present
    """
    if instruction is not None:
        if query.startswith(instruction):
            return query[len(instruction) :].strip()
        else:
            return query
    return query
