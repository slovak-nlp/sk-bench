# https://huggingface.co/docs/lighteval/en/using-the-python-api
import argparse
import os

import litellm

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from src.models.litellm_override import patch_litellm_client_greedy_until
from src.prompt_manager_override import patch_prompt_manager_chat_template
from src.tasks.task_registry import TASK_REGISTRY


litellm.suppress_debug_info = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Slovak Bench for API testing. API KEY should be set in environment variable."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help='Model identifier. Should include provider prefix (e.g., "gpt-4", "claude-3-sonnet")  or use provider/model format (e.g., "openai/gpt-4", "anthropic/claude-3-sonnet").',
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Custom base URL for the API. If None, uses provider's default URL.  Useful for using custom endpoints or local deployments",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Provider name (e.g., 'openai', 'anthropic'). Optional if model name includes provider prefix. Used to determine default API endpoint if base URL is not provided.",
        default="openai",
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="File with list of tasks",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save output results.")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--save-details", action="store_true", help="Save detailed results.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate. Good to speed up evaluation",
    )
    parser.add_argument("--cot-prompt", type=str, default=None, help="Path to chain-of-thought prompt file.")
    reasoning_tags_group = parser.add_mutually_exclusive_group()
    reasoning_tags_group.add_argument(
        "--remove-reasoning-tags",
        action="store_true",
        default=True,
        help="Remove reasoning tags from prompts (default).",
    )
    reasoning_tags_group.add_argument(
        "--no-remove-reasoning-tags",
        action="store_false",
        dest="remove_reasoning_tags",
        help="Do not remove reasoning tags from prompts.",
    )
    parser.add_argument(
        "--reasoning-tags",
        type=str,
        default="[('<think>', '</think>')]",
        help="Reasoning tags to consider in prompts.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate. Default is None (no limit).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional reasoning effort passed to LiteLLM (for example: low, medium, high, minimal, xhigh).",
    )
    # concurrent_requests and retry parameters can also be added as arguments if needed
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent requests to make to the API. Default is 5.",
    )
    parser.add_argument(
        "--api-retry-sleep",
        type=int,
        default=5,
        help="Initial sleep time in seconds before retrying a failed API request. Default is 5 seconds.",
    )
    parser.add_argument(
        "--api-retry-multiplier",
        type=int,
        default=2,
        help="Multiplier for exponential backoff when retrying API requests. Default is 2.",
    )
    parser.add_argument(
        "--api-max-retry",
        type=int,
        default=10,
        help="Maximum number of retries for failed API requests. Default is 10.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    patch_litellm_client_greedy_until(reasoning_effort=args.reasoning_effort)
    patch_prompt_manager_chat_template()
    all_tasks_list = TASK_REGISTRY.list_tasks()
    loaded_tasks = all_tasks_list
    if args.tasks_file:
        loaded_tasks = []
        with open(args.tasks_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                loaded_tasks.append(line.rstrip())

    tasks = ",".join(loaded_tasks)
    print(tasks)

    evaluation_tracker = EvaluationTracker(
        output_dir=args.output_dir, save_details=args.save_details, use_wandb=args.wandb
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="./src/tasks/__init__.py",  # Set to path if using custom tasks
        max_samples=args.max_samples,
        cot_prompt=args.cot_prompt,
        remove_reasoning_tags=args.remove_reasoning_tags,
        reasoning_tags=args.reasoning_tags,
    )

    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "API_KEY environment variable not set. Please set it to your API key before running the script."
        )

    if args.max_tokens is not None:
        generation_params = GenerationParameters(max_new_tokens=args.max_tokens)
    else:
        generation_params = GenerationParameters()

    model_config = LiteLLMModelConfig(
        model_name=args.model_name,
        provider=args.provider,
        base_url=args.base_url,
        api_key=api_key,
        generation_parameters=generation_params,
        concurrent_requests=args.concurrent_requests,
        api_retry_sleep=args.api_retry_sleep,
        api_retry_multiplier=args.api_retry_multiplier,
        api_max_retry=args.api_max_retry,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    print("Benchmarking completed.")


if __name__ == "__main__":
    main()
