"""
Custom LiteLLM model override to use model's max_new_tokens instead of task's generation_size.

This module patches the LiteLLMClient to fix the issue where task's generation_size
overrides the model's max_new_tokens from generation_parameters.
"""

from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import cached


def patch_litellm_client_greedy_until(reasoning_effort: str | None = None):  # noqa: C901
    """
    Patch LiteLLMClient.greedy_until to use model's max_new_tokens instead of task's generation_size.

    This is needed because the default behavior uses task's generation_size which overrides
    the model's max_new_tokens from generation_parameters.

    After this patch, if you set `max_new_tokens: 8192` in your models_config.yaml,
    it will be used instead of the task's generation_size (which defaults to 32-512).
    """
    try:
        import litellm

        from lighteval.models.endpoints.litellm_model import LiteLLMClient, LitellmModelResponse, logger

        def patched_call_api(self, prompt, return_logits, max_new_tokens, num_samples, stop_sequence):  # noqa: C901
            """Patched version that also forwards optional reasoning_effort to LiteLLM."""
            response = LitellmModelResponse()
            stop_sequence = self._prepare_stop_sequence(stop_sequence)
            max_new_tokens = self._prepare_max_new_tokens(max_new_tokens)

            if return_logits and not self.provider == "openai":
                logger.warning("Returning logits is not supported for this provider, ignoring.")

            kwargs = {
                "model": self.model,
                "messages": prompt,
                "response_format": {"type": "text"},
                "max_tokens": max_new_tokens,
                "logprobs": return_logits if self.provider == "openai" else None,
                "stop": stop_sequence,
                "base_url": self.base_url,
                "api_key": self.api_key,
                "n": num_samples,
                "caching": True,
                "timeout": self.timeout,
            }

            if "o1" in self.model:
                logger.warning("O1 models do not support temperature, top_p, stop sequence. Disabling.")
            else:
                kwargs.update(self.generation_parameters.to_litellm_dict())

            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort

            if kwargs.get("max_completion_tokens", None) is None:
                kwargs["max_completion_tokens"] = max_new_tokens
            else:
                kwargs["max_tokens"] = kwargs["max_completion_tokens"]

            # print(f"API call patched kwargs: {kwargs}")

            for attempt in range(self.API_MAX_RETRY):
                try:
                    response = litellm.completion(**kwargs)
                    content = response.choices[0].message.content

                    if not content:
                        logger.info("Response is empty, retrying without caching")
                        kwargs["caching"] = False
                        response = litellm.completion(**kwargs)
                        content = response.choices[0].message.content

                    return response
                except litellm.BadRequestError as e:
                    if "message" in e.__dict__:
                        error_string = "The response was filtered due to the prompt triggering Microsoft's content management policy"
                        if error_string in e.__dict__["message"]:
                            logger.warning(f"{error_string}. Returning empty response.")
                            return LitellmModelResponse()
                except Exception as e:
                    wait_time = min(64, self.API_RETRY_SLEEP * (self.API_RETRY_MULTIPLIER**attempt))
                    logger.warning(
                        f"Error in API call: {e}, waiting {wait_time} seconds before retry {attempt + 1}/{self.API_MAX_RETRY}"
                    )
                    import time

                    time.sleep(wait_time)

            logger.error(f"API call failed after {self.API_MAX_RETRY} attempts, returning empty response.")
            return LitellmModelResponse()

        @cached(SamplingMethod.GENERATIVE)
        def patched_greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
            """Patched version that uses model's max_new_tokens if set."""
            dataset = GenerativeTaskDataset(requests=docs, num_dataset_splits=self.DATASET_SPLITS)
            results = []

            for split in tqdm(
                dataset.splits_iterator(),
                total=dataset.num_dataset_splits,
                desc="Splits",
                position=0,
                disable=self.disable_tqdm,
            ):
                contexts = [self.prompt_manager.prepare_prompt_api(doc) for doc in dataset]
                max_new_tokens = self.generation_parameters.max_new_tokens or split[0].generation_size
                return_logits = split[0].use_logits
                num_samples = split[0].num_samples
                if self.generation_parameters.max_new_tokens is not None:
                    stop_sequence = None
                else:
                    stop_sequence = split[0].stop_sequences

                if num_samples > 1 and self.generation_parameters.temperature == 0:
                    raise ValueError(
                        "num_samples > 1 is not supported with temperature=0, please set temperature > 0 or use non sampling metrics."
                    )

                # Call the private method via self, not LiteLLMClient
                responses = self._LiteLLMClient__call_api_parallel(
                    contexts, return_logits, max_new_tokens, num_samples, stop_sequence
                )

                for response, context in zip(responses, contexts):
                    result: list[str] = [choice.message.content for choice in response.choices]
                    reasonings: list[str | None] = [
                        getattr(choice.message, "reasoning_content", None) for choice in response.choices
                    ]

                    cur_response = ModelResponse(
                        text=result if result[0] else [""],
                        reasonings=reasonings,
                        input=context,
                    )
                    results.append(cur_response)

            return dataset.get_original_order(results)

        # Apply the patch
        LiteLLMClient._LiteLLMClient__call_api = patched_call_api
        LiteLLMClient.greedy_until = patched_greedy_until
        print("Successfully patched LiteLLMClient with max_new_tokens and reasoning_effort support.")
        return True
    except Exception as e:
        print(f"Warning: Failed to patch LiteLLMClient.greedy_until: {e}")
        return False
