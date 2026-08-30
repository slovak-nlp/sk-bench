# https://huggingface.co/docs/lighteval/en/using-the-python-api
import argparse
import os

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from src.models.vllm_override import setup_vllm_patches
from src.prompt_manager_override import patch_prompt_manager_chat_template
from src.tasks.task_registry import TASK_REGISTRY


def parse_arguments():
    parser = argparse.ArgumentParser(description="Slovak Bench")
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF Model identifier.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism. Default 1",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help=" Positive value less than 1. lower this if you are running out of memory. Default 0.9",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type for the model. Possible values are bfloat16, float16 and float32. Default is bfloat16 (from lighteval)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default=None,
        help="Data type for the KV cache. Passed through to vLLM's LLM constructor. If not set, vLLM's default is used.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="vLLM KV-cache block size (page size) in tokens. Models such as MiniMax-M3 "
        "require 128 because their sparse-attention backend only supports that block size. "
        "If not set, vLLM's default (16) is used.",
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
        "--reasoning-effort",
        type=str,
        default=None,
        help="Optional reasoning effort for thinking models (e.g., none, low, medium, high).",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to override the model's chat-template default.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Whether to trust remote code when loading the model. Use this if you are loading a custom model that requires custom code.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default=None,
        help="vLLM tokenizer mode (e.g., auto, deepseek_v4, mistral). "
        "Defaults to the VLLM_TOKENIZER_MODE environment variable, or auto.",
    )
    # max-tokens, by default None
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate. Default is None (no limit).",
    )
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        default=2048,
        help="Maximum number of tokens to batch together for generation. Adjust this based on your GPU memory and model size for better performance. Default is 2048.",
    )
    # skip-special-tokens: default True matches vLLM's SamplingParams default.
    # Use --no-skip-special-tokens to preserve special tokens for thinking models.
    skip_special_group = parser.add_mutually_exclusive_group()
    skip_special_group.add_argument(
        "--skip-special-tokens",
        action="store_true",
        default=True,
        dest="skip_special_tokens",
        help="Skip special tokens in decoded output (default, matches vLLM default).",
    )
    skip_special_group.add_argument(
        "--no-skip-special-tokens",
        action="store_false",
        dest="skip_special_tokens",
        help="Keep special tokens in decoded output. Needed for thinking models like Gemma 3 "
        "to preserve thinking/response delimiter tokens.",
    )
    parser.add_argument(
        "--max-model-length",
        type=int,
        default=None,
        help="Maximum model length (context window). If not specified, will use the model's default.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=False,
        help="Disable CUDA graphs by passing enforce_eager=True to vLLM. "
        "Lighteval hardcodes this to True, which destroys throughput on large "
        "models. Only enable this if CUDA graphs cause correctness issues.",
    )
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        default=None,
        help="Run a multimodal (vision-language) model in text-only mode by "
        "passing language_model_only=True to vLLM's LLM constructor. Loads the "
        "language model weights only, skipping the vision encoder and "
        "multimodal processor. Useful for Qwen3.5 and similar VL models used "
        "for text-only inference. Not passed to vLLM unless explicitly set.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_vllm_patches(
        tokenizer_mode=args.tokenizer_mode,
        kv_cache_dtype=args.kv_cache_dtype,
        block_size=args.block_size,
        enforce_eager=args.enforce_eager,
        language_model_only=args.language_model_only,
    )
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
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory="./src/tasks/__init__.py",  # Set to path if using custom tasks
        max_samples=args.max_samples,
        cot_prompt=args.cot_prompt,
        remove_reasoning_tags=args.remove_reasoning_tags,
        reasoning_tags=args.reasoning_tags,
    )

    if args.reasoning_effort is not None:
        os.environ["REASONING_EFFORT"] = args.reasoning_effort
    print("Applying chat template with reasoning effort:", os.environ.get("REASONING_EFFORT"))

    if args.max_tokens is not None:
        generation_params = GenerationParameters(
            max_new_tokens=args.max_tokens,
            skip_special_tokens=args.skip_special_tokens,
        )
    else:
        generation_params = GenerationParameters(
            skip_special_tokens=args.skip_special_tokens,
        )

    model_config = VLLMModelConfig(
        model_name=args.model_name,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        generation_parameters=generation_params,
        max_num_batched_tokens=args.max_batched_tokens,
        max_model_length=args.max_model_length,
        enforce_eager=args.enforce_eager,
        system_prompt=args.system_prompt,
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
