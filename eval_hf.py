# https://huggingface.co/docs/lighteval/en/using-the-python-api
import argparse

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
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
        "--dtype",
        type=str,
        default="float16",
        help="Data type for model. Could require bitsandbytes for quantization",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="eager",
        help="Attention implementation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="File with list of tasks",
    )
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save output results.")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate. Good to speed up evaluation",
    )
    return parser.parse_args()


def main():
    patch_prompt_manager_chat_template()
    args = parse_arguments()
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

    evaluation_tracker = EvaluationTracker(output_dir=args.output_dir, save_details=True, use_wandb=args.wandb)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory="./src/tasks/__init__.py",  # Set to path if using custom tasks
        max_samples=args.max_samples,
    )
    model_config = TransformersModelConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        dtype=args.dtype,
        model_loading_kwargs={"attn_implementation": args.attn},
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    # should save after each task is finished
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    main()
