import torch

from .generate import generate


def chat(model, tokenizer, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = 50):
    device = next(model.parameters()).device
    print("Type 'exit' to quit.")
    while True:
        prompt = input(">> ")
        if prompt.strip().lower() in ("exit", "quit"):
            break
        tokens = tokenizer.encode(prompt).ids
        idx = torch.tensor(tokens).unsqueeze(0).to(device)
        out = generate(model, idx, max_new_tokens, temperature=temperature, top_k=top_k)
        print(tokenizer.decode(out[0].tolist()))
