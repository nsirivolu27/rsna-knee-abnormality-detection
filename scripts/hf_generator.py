"""Batched text generation backed by a local instruct model.

This is the only module that imports torch or transformers. Everything in src/
stays framework free and testable without them; the generator is injected into
run_labeling at the call site. Written for a Kaggle GPU notebook with the model
attached from Kaggle Models, so nothing is downloaded at run time.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Callable

GenerateFn = Callable[[list[str]], list[str]]


def find_model_directory(hint: str | None = None) -> Path:
    """Locate an attached model directory under /kaggle/input.

    Kaggle nests model mounts several levels deep and the exact depth varies by
    framework and variation, so search for the directory holding config.json
    rather than assuming a layout.
    """
    roots = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    if hint:
        roots = [r for r in roots if hint.lower() in r.lower()] or roots
    if not roots:
        raise FileNotFoundError(
            "No model found under /kaggle/input. Attach one through Add Input, Models."
        )
    return Path(roots[0]).parent


def build_generator(
    model_dir: Path | str | None = None,
    hint: str | None = None,
    max_new_tokens: int = 256,
    max_prompt_tokens: int = 3072,
    load_in_4bit: bool = False,
) -> GenerateFn:
    """Return a callable mapping a list of prompts to a list of completions.

    Decoding is greedy so a rerun of the same prompt gives the same label. That
    matters more than sample diversity here: the output feeds a training set,
    and a label set that changes between runs cannot be debugged.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = Path(model_dir) if model_dir else find_model_directory(hint)
    print(f"Loading model from {directory}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(directory, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, object] = {"device_map": "auto"}
    if load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(directory, **load_kwargs)
    model.eval()
    print(f"Loaded on {model.device}", flush=True)

    def _render(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate(prompts: list[str]) -> list[str]:
        rendered = [_render(prompt) for prompt in prompts]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        start = encoded["input_ids"].shape[1]
        return [
            tokenizer.decode(sequence[start:], skip_special_tokens=True)
            for sequence in generated
        ]

    return generate
