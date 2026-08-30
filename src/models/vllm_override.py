"""
Custom VLLM model override to use litellm tokenizer instead of vllm's get_tokenizer.

This module patches the VLLMModel to fix tokenizer initialization using litellm,
adds support for controlling skip_special_tokens in GenerationParameters,
and makes lighteval's hardcoded ``enforce_eager`` configurable so CUDA graphs
stay enabled by default.
"""

import os
from contextlib import contextmanager
from typing import Any


def _get_original_llm(vllm_model_module, OriginalLLM):
    """Get the original vLLM LLM class, unwrapping any previous override."""
    current_llm = vllm_model_module.LLM
    if current_llm is OriginalLLM:
        return OriginalLLM
    if hasattr(current_llm, "_vllm_override_original"):
        return current_llm._vllm_override_original
    return OriginalLLM


def _patch_vllm_llm(extra_kwargs: dict[str, Any]) -> bool:
    """Wrap lighteval's module-level LLM reference to inject extra kwargs.

    Multiple overrides can be stacked; each wrapper remembers the original
    ``LLM`` class so that later overrides do not lose earlier ones.
    """
    try:
        import lighteval.models.vllm.vllm_model as vllm_model_module
        from vllm import LLM as OriginalLLM

        original = _get_original_llm(vllm_model_module, OriginalLLM)

        def LLMWithExtraKwargs(*args, **kwargs):
            for key, value in extra_kwargs.items():
                if value is not None:
                    kwargs.setdefault(key, value)
            return original(*args, **_filter_supported_engine_kwargs(kwargs))

        LLMWithExtraKwargs._vllm_override_original = original
        vllm_model_module.LLM = LLMWithExtraKwargs
        return True
    except Exception as e:
        print(f"Warning: Failed to patch vLLM LLM: {e}")
        return False


def _get_supported_engine_kwargs() -> frozenset[str] | None:
    """Collect kwargs accepted by the installed vLLM's engine constructors.

    vLLM regularly removes or renames engine arguments (e.g. ``swap_space``
    was dropped in v0.28.0), while the pinned lighteval revision still
    hardcodes them in its ``model_args``, which crashes engine construction
    with a ``TypeError``. The accepted set is the union of the ``EngineArgs``
    and ``AsyncEngineArgs`` dataclass fields plus ``LLM.__init__``'s named
    parameters, since ``LLM`` forwards everything else to ``EngineArgs``.

    Returns ``None`` if vLLM cannot be introspected, in which case no
    filtering happens (matching the unpatched behavior).
    """
    try:
        import dataclasses
        import inspect

        from vllm import LLM
        from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs

        names = {field.name for field in dataclasses.fields(EngineArgs)}
        names.update(field.name for field in dataclasses.fields(AsyncEngineArgs))
        names.update(inspect.signature(LLM.__init__).parameters)
        return frozenset(names)
    except Exception as e:
        print(f"Warning: Could not introspect vLLM engine arguments: {e}")
        return None


def _filter_supported_engine_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs the installed vLLM engine does not accept.

    This is the safety net for lighteval's hardcoded ``model_args``
    (e.g. ``swap_space``, removed in vLLM v0.28.0): without it, engine
    construction crashes with a ``TypeError`` before any model loading
    starts. Works for both the sync and async engines since they share
    the ``EngineArgs`` field set.
    """
    allowed = _get_supported_engine_kwargs()
    if allowed is None:
        return kwargs
    dropped = sorted(k for k in kwargs if k not in allowed)
    if dropped:
        print(f"Dropping engine kwargs not supported by this vLLM version: {dropped}")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _patch_transformers_mistral_regex_default():
    """
    Runtime monkey patch for the main process: make affected Mistral-derived
    tokenizers default to ``fix_mistral_regex=True``.

    vLLM loads tokenizers in many internal paths (engine workers, data-parallel
    Ray workers, multimodal preprocessors) that do not go through
    ``VLLMModel._create_auto_tokenizer``. Those worker processes are spawned
    with ``multiprocessing.spawn`` and do not inherit runtime monkey patches,
    so the installed transformers package is also patched by
    ``scripts/patch_transformers_mistral_regex.py`` (called from
    ``container/eval_entry.sh``). This function serves as an additional
    safeguard for the main process, while an explicit
    ``fix_mistral_regex=False`` is still respected.
    """
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        original_patch = TokenizersBackend._patch_mistral_regex

        @classmethod
        def patched_patch(cls, tokenizer, pretrained_model_name_or_path, *args, fix_mistral_regex=None, **kwargs):
            if fix_mistral_regex is None:
                fix_mistral_regex = True
            return original_patch(
                tokenizer,
                pretrained_model_name_or_path,
                *args,
                fix_mistral_regex=fix_mistral_regex,
                **kwargs,
            )

        TokenizersBackend._patch_mistral_regex = patched_patch
        print("Successfully patched TokenizersBackend._patch_mistral_regex to default fix_mistral_regex=True.")
        return True
    except Exception as e:
        print(f"Warning: Failed to patch TokenizersBackend._patch_mistral_regex: {e}")
        return False


def patch_vllm_model_tokenizer(tokenizer_mode: str | None = None):
    """
    Patch VLLMModel._create_auto_tokenizer to use litellm tokenizer.

    This overrides the default vllm get_tokenizer with litellm's get_tokenizer
    for better compatibility with various model formats.

    Args:
        tokenizer_mode: Optional vLLM tokenizer mode (e.g., ``auto``,
            ``deepseek_v4``, ``mistral``). If not provided, falls back to the
            ``VLLM_TOKENIZER_MODE`` environment variable, and ultimately to
            ``auto``.
    """
    resolved_tokenizer_mode = tokenizer_mode or os.environ.get("VLLM_TOKENIZER_MODE", "auto")

    # Ensure Mistral-derived tokenizers use the corrected regex across all
    # tokenizer loading paths, not just VLLMModel._create_auto_tokenizer.
    _patch_transformers_mistral_regex_default()

    try:
        from lighteval.models.vllm.vllm_model import VLLMModel, VLLMModelConfig, logger
        from vllm.tokenizers import get_tokenizer

        def patched_create_auto_tokenizer(self, config: VLLMModelConfig):
            tokenizer_name = config.tokenizer or config.model_name  # use HF tokenizer for non-HF models, like GGUF model.
            tokenizer_kwargs = {
                "tokenizer_mode": resolved_tokenizer_mode,
                "trust_remote_code": config.trust_remote_code,
                "revision": config.revision,
            }

            try:
                # Step-3.7-Flash / Mistral-derived tokenizers need the regex fix.
                tokenizer = get_tokenizer(tokenizer_name, fix_mistral_regex=True, **tokenizer_kwargs)
            except TypeError as e:
                if "fix_mistral_regex" in str(e):
                    tokenizer = get_tokenizer(tokenizer_name, **tokenizer_kwargs)
                else:
                    raise

            try:
                tokenizer.pad_token = tokenizer.eos_token
            except AttributeError:
                # Some models (e.g., Mistral) don't have eos_token, use eos_token_id instead
                logger.warning(
                    f"Tokenizer for {config.model_name} does not have eos_token. "
                    "Setting pad_token to eos_token_id."
                )
                tokenizer.pad_token = tokenizer.eos_token_id
            return tokenizer

        # Apply the patch
        VLLMModel._create_auto_tokenizer = patched_create_auto_tokenizer
        print("Successfully patched VLLMModel._create_auto_tokenizer with litellm tokenizer.")
        return True
    except Exception as e:
        print(f"Warning: Failed to patch VLLMModel._create_auto_tokenizer: {e}")
        return False


def patch_vllm_llm_kwargs(**kwargs) -> bool:
    """
    Inject arbitrary kwargs into the vLLM ``LLM`` constructor used by lighteval.

    lighteval's ``VLLMModelConfig`` does not expose every vLLM engine argument.
    This patch wraps the module-level ``LLM`` reference so every model created
    by lighteval (including data-parallel Ray workers) receives the provided
    kwargs. Multiple calls are composed automatically, so ``kv_cache_dtype``,
    ``block_size``, and any future argument can be set from ``eval_vllm.py`` in
    one place.

    Args:
        **kwargs: Keyword arguments to pass through to ``vllm.LLM``. ``None``
            values are skipped.

    Returns:
        True if the patch was applied, False if there was nothing to patch or
        patching failed.
    """
    extra_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not extra_kwargs:
        return False
    success = _patch_vllm_llm(extra_kwargs)
    if success:
        print(f"Successfully patched vLLM LLM kwargs: {extra_kwargs}.")
    return success


def patch_generation_parameters_skip_special_tokens():
    """
    Add skip_special_tokens field to GenerationParameters so it can be passed
    through to vLLM's SamplingParams via to_vllm_dict().

    By default, vLLM's SamplingParams.skip_special_tokens=True, which strips
    special tokens from decoded output. For thinking models like Gemma 3 that
    use special tokens to delimit thinking vs. final response, this must be
    set to False to preserve those tokens for extraction.

    After calling this patch, you can pass skip_special_tokens directly to
    GenerationParameters, e.g.:

        GenerationParameters(skip_special_tokens=False)

    And it will automatically flow through to_vllm_dict() → SamplingParams.
    """
    try:
        from pydantic.fields import FieldInfo

        from lighteval.models.model_input import GenerationParameters

        # Only add the field if it doesn't already exist
        if "skip_special_tokens" not in GenerationParameters.model_fields:
            GenerationParameters.model_fields["skip_special_tokens"] = FieldInfo(
                annotation=bool,
                default=True,
                description="Whether to skip special tokens in decoded output. "
                "Set to False for thinking models (e.g., Gemma 3) that need "
                "special tokens preserved to separate thinking from response.",
            )
            # Rebuild the model to pick up the new field
            GenerationParameters.model_rebuild(force=True)
            print("Successfully added skip_special_tokens field to GenerationParameters.")
        else:
            print("skip_special_tokens field already exists on GenerationParameters.")

        return True
    except Exception as e:
        print(f"Warning: Failed to patch GenerationParameters with skip_special_tokens: {e}")
        return False


def _add_enforce_eager_field_to_config():
    """Add an ``enforce_eager`` field to lighteval's VLLMModelConfig."""
    from pydantic.fields import FieldInfo

    from lighteval.models.vllm.vllm_model import VLLMModelConfig

    if "enforce_eager" not in VLLMModelConfig.model_fields:
        VLLMModelConfig.model_fields["enforce_eager"] = FieldInfo(
            annotation=bool,
            default=False,
            description="Whether to enforce eager execution and disable CUDA graphs. "
            "Defaults to False for best throughput.",
        )
        VLLMModelConfig.model_rebuild(force=True)
        print("Successfully added enforce_eager field to VLLMModelConfig (default=False).")
    else:
        print("enforce_eager field already exists on VLLMModelConfig.")


@contextmanager
def _override_module_attr(module, name, wrapper):
    """Temporarily replace ``module.name`` with ``wrapper(original)``."""
    original = getattr(module, name)
    setattr(module, name, wrapper(original))
    try:
        yield
    finally:
        setattr(module, name, original)


def patch_vllm_enforce_eager() -> bool:
    """
    Make lighteval's hardcoded ``enforce_eager`` configurable via config.

    lighteval always passes ``enforce_eager=True`` to vLLM, which disables
    CUDA graphs and causes very low throughput (observed as ~10-20 tok/s on
    8 GPUs). This patch adds an ``enforce_eager`` attribute to
    ``VLLMModelConfig`` (default ``False``) and overrides both the sync and
    async model creation paths so the config value is respected.

    Returns:
        True if the patch was applied, False otherwise.
    """
    try:
        import lighteval.models.vllm.vllm_model as vllm_model_module
        from lighteval.models.vllm.vllm_model import (
            AsyncVLLMModel,
            VLLMModel,
            VLLMModelConfig,
        )

        _add_enforce_eager_field_to_config()

        original_sync = VLLMModel._create_auto_model

        def patched_sync_create_auto_model(self, config: VLLMModelConfig):
            enforce_eager = getattr(config, "enforce_eager", False)

            def wrap_llm(original_llm):
                def LLMWithEager(*args, **kwargs):
                    kwargs["enforce_eager"] = enforce_eager
                    return original_llm(*args, **_filter_supported_engine_kwargs(kwargs))

                return LLMWithEager

            with _override_module_attr(vllm_model_module, "LLM", wrap_llm):
                return original_sync(self, config)

        VLLMModel._create_auto_model = patched_sync_create_auto_model

        original_async = AsyncVLLMModel._create_auto_model

        def patched_async_create_auto_model(self, config: VLLMModelConfig):
            enforce_eager = getattr(config, "enforce_eager", False)

            def wrap_engine_args_cls(original_cls):
                def AsyncEngineArgsWithFilter(*args, **kwargs):
                    kwargs["enforce_eager"] = enforce_eager
                    return original_cls(*args, **_filter_supported_engine_kwargs(kwargs))

                return AsyncEngineArgsWithFilter

            def wrap_from_engine_args(original_from_engine_args):
                def from_engine_args_with_eager(engine_args, *args, **kwargs):
                    engine_args.enforce_eager = enforce_eager
                    return original_from_engine_args(engine_args, *args, **kwargs)

                return from_engine_args_with_eager

            # AsyncVLLMModel builds AsyncEngineArgs(**self.model_args) inside
            # _create_auto_model, so the filter must intercept the class
            # itself, not only from_engine_args (which fires too late for
            # kwargs validation).
            with _override_module_attr(
                vllm_model_module, "AsyncEngineArgs", wrap_engine_args_cls
            ), _override_module_attr(
                vllm_model_module.AsyncLLM, "from_engine_args", wrap_from_engine_args
            ):
                return original_async(self, config)

        AsyncVLLMModel._create_auto_model = patched_async_create_auto_model

        print("Successfully patched VLLMModel and AsyncVLLMModel to respect config.enforce_eager.")
        return True
    except Exception as e:
        print(f"Warning: Failed to patch enforce_eager handling: {e}")
        return False


def setup_vllm_patches(
    *,
    tokenizer_mode: str | None = None,
    kv_cache_dtype: str | None = None,
    block_size: int | None = None,
    enforce_eager: bool = False,
    language_model_only: bool | None = None,
) -> None:
    """
    Apply all vLLM/lighteval monkey patches in one call.

    This is a convenience wrapper around the individual patch functions so
    callers (e.g. ``eval_vllm.py``) do not need to import and invoke each one
    separately.

    Args:
        tokenizer_mode: Optional vLLM tokenizer mode (e.g. ``auto``,
            ``deepseek_v4``, ``mistral``).
        kv_cache_dtype: Data type for the KV cache, passed through to vLLM's
            ``LLM`` constructor.
        block_size: vLLM KV-cache block size, passed through to vLLM's ``LLM``
            constructor.
        enforce_eager: If True, disable CUDA graphs (old lighteval default).
            Defaults to False so vLLM can use CUDA graphs / breakable CUDA
            graphs for DeepSeek-V4-class models.
        language_model_only: If True, run a multimodal (vision-language) model
            in text-only mode by passing ``language_model_only=True`` to vLLM's
            ``LLM`` constructor — loads the language model weights only,
            skipping the vision encoder and multimodal processor. ``None`` (the
            default) leaves the argument unset so it is not passed to vLLM,
            matching vLLM's default and avoiding a ``TypeError`` on versions
            that do not accept the kwarg.
    """
    patch_vllm_model_tokenizer(tokenizer_mode=tokenizer_mode)
    patch_generation_parameters_skip_special_tokens()
    # Apply the extra-kwargs wrapper first so that the enforce_eager wrapper
    # sits on top of it and can override the hardcoded ``enforce_eager=True``
    # before kwargs reach vLLM's LLM constructor.
    patch_vllm_llm_kwargs(
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        language_model_only=language_model_only,
    )
    patch_vllm_enforce_eager()
