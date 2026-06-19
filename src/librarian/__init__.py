"""Librarian — a config-driven framework to pretrain and fine-tune Librarian GPT models.

One package, two modes:
  - pretrain : raw text (parquet/txt) -> tokenizer -> tokenize -> pack -> train -> eval
  - sft      : prompt/completion records (parquet/txt) -> prepare -> train (masked) -> eval

The user supplies cleaned local data + a single JSON config; the framework does the rest.
"""

__version__ = "0.1.0"
