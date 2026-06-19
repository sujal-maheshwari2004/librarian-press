"""
chat.py — load an exported bundle and run an interactive, streaming REPL.

Ollama-style: `librarian-press chat <model-name>` opens a prompt, streams tokens
as they're generated, and keeps going until you type /bye, exit, or hit Ctrl-D.
"""

from __future__ import annotations

import torch

from ..config.schema import ModelConfig
from ..data.prepare_sft import render_prompt
from ..inference.load_model import load_model
from ..inference.sampler import sample_next_token
from ..tokenizer.load import load_tokenizer, special_ids
from ..utils.device import resolve_device
from .registry import load_bundle

_QUIT = {"/bye", "/exit", "/quit", "exit", "quit"}


def _build_prompt(user_text: str, template: str | None, field: str | None) -> str:
    if not template:
        return user_text
    example = {field: user_text} if field else {}
    return render_prompt(example, template)


@torch.no_grad()
def _stream(model, tokenizer, prompt, *, eos_id, device, max_seq_len,
            temperature, top_k, max_new_tokens):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    generated: list[int] = []
    prev = ""
    for _ in range(max_new_tokens):
        logits = model(idx[:, -max_seq_len:])[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = sample_next_token(logits, temperature, top_k)
        tok = int(nxt.item())
        if tok == eos_id:
            break
        generated.append(tok)
        text = tokenizer.decode(generated)
        # print only the newly-decoded suffix (byte-level BPE decodes cleanly in full)
        print(text[len(prev):], end="", flush=True)
        prev = text
        idx = torch.cat([idx, nxt], dim=1)
    print()


def run_chat(name: str, device: str | None = None):
    bundle = load_bundle(name)
    model_cfg = ModelConfig(**bundle["model"])
    device = resolve_device(device or "cuda")

    tokenizer = load_tokenizer(bundle["_tokenizer_path"])
    model = load_model(model_cfg, bundle["_weights_path"], device)
    eos_id = special_ids(tokenizer)["eos"]

    gen = bundle.get("generation", {})
    temperature = gen.get("temperature", 0.8)
    top_k = gen.get("top_k", 40)
    max_new_tokens = gen.get("max_new_tokens", 256)
    max_seq_len = bundle.get("max_seq_len", model_cfg.max_seq_len)
    template = bundle.get("prompt_template")
    field = bundle.get("prompt_field")

    print(f"librarian-press chat - model '{name}' on {device}")
    print("Type your message. Commands: /bye to quit, /clear has no effect (stateless).")
    print("-" * 50)

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in _QUIT:
            break
        prompt = _build_prompt(user, template, field)
        print("model> ", end="", flush=True)
        _stream(model, tokenizer, prompt, eos_id=eos_id, device=device,
                max_seq_len=max_seq_len, temperature=temperature,
                top_k=top_k, max_new_tokens=max_new_tokens)

    print("bye.")
