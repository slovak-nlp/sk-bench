#!/usr/bin/env python3
"""
Patch the installed transformers library so Mistral-derived tokenizers
(e.g. Stepfun Step-3.7-Flash) default to `fix_mistral_regex=True`.

The runtime monkey patch in `src/models/vllm_override.py` only covers the
main process. vLLM's tensor-parallel / data-parallel workers are spawned
fresh (multiprocessing ``spawn``), so they do not inherit runtime monkey
patches. Patching the installed package makes the fix apply in every Python
process that loads this transformers installation.

Usage:
    python scripts/patch_transformers_mistral_regex.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def patch_file(path: Path) -> bool:
    """Patch a single tokenization_utils_tokenizers.py file.

    Returns ``True`` if the file was patched or already patched, ``False`` if
    the file exists but could not be patched. Missing files are skipped.
    """
    if not path.exists():
        print(f"File not found, skipping: {path}")
        return True

    original = path.read_text(encoding="utf-8")

    old_block = '''                # only warn if its not explicitly passed
                if fix_mistral_regex is None and not getattr(tokenizer, "fix_mistral_regex", False):
                    setattr(tokenizer, "fix_mistral_regex", False)
                    logger.warning(
                        f"The tokenizer you are loading from '{pretrained_model_name_or_path}'"
                        f" with an incorrect regex pattern: https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/discussions/84#69121093e8b480e709447d5e."
                        " This will lead to incorrect tokenization. You should set the `fix_mistral_regex=True` flag when loading this tokenizer to fix this issue."
                    )
                elif fix_mistral_regex is True or getattr(tokenizer, "fix_mistral_regex", False):
                    setattr(tokenizer, "fix_mistral_regex", True)'''

    new_block = '''                # Default to applying the fix unless explicitly disabled.
                # This prevents incorrect tokenization for Mistral-derived models
                # such as Step-3.7-Flash when `fix_mistral_regex` is not passed.
                if fix_mistral_regex is None:
                    fix_mistral_regex = getattr(tokenizer, "fix_mistral_regex", True)

                if fix_mistral_regex is False:
                    setattr(tokenizer, "fix_mistral_regex", False)
                elif fix_mistral_regex is True or getattr(tokenizer, "fix_mistral_regex", False):
                    setattr(tokenizer, "fix_mistral_regex", True)'''

    if new_block in original:
        print(f"Already patched: {path}")
        return True

    if old_block not in original:
        print(f"Could not find expected block in: {path}")
        print("The transformers version may have changed. Manual inspection needed.")
        return False

    patched = original.replace(old_block, new_block)
    path.write_text(patched, encoding="utf-8")
    print(f"Patched: {path}")
    return True


def main() -> int:
    targets = []

    # Container venv used by run-eval.sh. Outside the container HOME points to
    # the user's home directory; inside the container HOME is APPHOME, so the
    # venv is directly under $HOME/venv.
    for container_root in [Path.home() / "apphome", Path.home()]:
        targets.append(
            container_root
            / "venv/lib/python3.11/site-packages/transformers/tokenization_utils_tokenizers.py"
        )

    # Project venv used by local uv runs (Python 3.13).
    targets.append(
        Path(__file__).parent.parent
        / ".venv/lib/python3.13/site-packages/transformers/tokenization_utils_tokenizers.py"
    )

    # Also patch the active VIRTUAL_ENV, if any and different from the above.
    active_venv = os.environ.get("VIRTUAL_ENV")
    if active_venv:
        site_packages = next(Path(active_venv).glob("lib/python*/site-packages"), None)
        if site_packages:
            targets.append(site_packages / "transformers/tokenization_utils_tokenizers.py")

    ok = True
    seen = set()
    for target in targets:
        resolved = target.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not patch_file(target):
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
