"""
load_config — parse + validate a unified JSON config into a RunConfig.

The model section may be given inline OR as {"config_path": "model_130M.json"}
(back-compat with the existing base-repo model JSON files).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import (
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
    PretrainDataConfig,
    SFTDataConfig,
    LoRAConfig,
    FinetuneConfig,
    PretrainSection,
    SFTSection,
    RunConfig,
)

VALID_MODES = ("pretrain", "sft", "both")
VALID_METHODS = ("lora", "bitfit", "full")
VALID_METRICS = ("perplexity", "exact_match", "f1")
UINT16_CEILING = 65536


class ConfigError(ValueError):
    pass


def _load_model(raw: dict[str, Any]) -> ModelConfig:
    if "config_path" in raw:
        path = Path(raw["config_path"])
        if not path.exists():
            raise ConfigError(f"model.config_path not found: {path}")
        with path.open() as f:
            data = json.load(f)
        # allow inline overrides alongside config_path
        data.update({k: v for k, v in raw.items() if k != "config_path"})
        return ModelConfig(**data)
    return ModelConfig(**raw)


def _load_training(raw: dict | None) -> TrainingConfig:
    return TrainingConfig(**(raw or {}))


def _load_pretrain(raw: dict) -> PretrainSection:
    data = PretrainDataConfig(**raw["data"])
    training = _load_training(raw.get("training"))
    return PretrainSection(data=data, training=training)


def _load_sft(raw: dict) -> SFTSection:
    data = SFTDataConfig(**raw["data"])
    ft_raw = dict(raw.get("finetune", {}))
    lora = LoRAConfig(**(ft_raw.pop("lora", None) or {}))
    finetune = FinetuneConfig(lora=lora, **ft_raw)
    training = _load_training(raw.get("training"))
    return SFTSection(data=data, finetune=finetune, training=training)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config not found: {path}")

    with path.open() as f:
        raw: dict[str, Any] = json.load(f)

    # ── required top-level fields ────────────────────────────────────
    for key in ("name", "mode", "model", "tokenizer"):
        if key not in raw:
            raise ConfigError(f"Config missing required field: {key!r}")

    mode = raw["mode"]
    if mode not in VALID_MODES:
        raise ConfigError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    model = _load_model(raw["model"])
    tokenizer = TokenizerConfig(**raw["tokenizer"])

    pretrain = None
    sft = None
    if mode in ("pretrain", "both"):
        if "pretrain" not in raw:
            raise ConfigError(f"mode={mode!r} requires a 'pretrain' section")
        pretrain = _load_pretrain(raw["pretrain"])
    if mode in ("sft", "both"):
        if "sft" not in raw:
            raise ConfigError(f"mode={mode!r} requires an 'sft' section")
        sft = _load_sft(raw["sft"])

    cfg = RunConfig(
        name=raw["name"],
        mode=mode,
        model=model,
        tokenizer=tokenizer,
        pretrain=pretrain,
        sft=sft,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: RunConfig):
    m = cfg.model
    t = cfg.tokenizer

    # uint16 token storage ceiling
    if m.vocab_size >= UINT16_CEILING:
        raise ConfigError(
            f"model.vocab_size ({m.vocab_size}) must be < {UINT16_CEILING} "
            "(tokens are stored as uint16)"
        )

    # model vocab must match the tokenizer the run is built around
    if t.vocab_size != m.vocab_size:
        raise ConfigError(
            f"tokenizer.vocab_size ({t.vocab_size}) != model.vocab_size "
            f"({m.vocab_size}); they must match"
        )

    # if the tokenizer file already exists, the model embedding must be able to
    # represent every token id: real vocab must not EXCEED model.vocab_size.
    # (A smaller real vocab is fine — small corpora yield fewer BPE merges.)
    tok_path = Path(t.path)
    if tok_path.exists():
        try:
            from tokenizers import Tokenizer

            real = Tokenizer.from_file(str(tok_path)).get_vocab_size()
            if real > m.vocab_size:
                raise ConfigError(
                    f"tokenizer at {tok_path} has vocab {real} > model.vocab_size "
                    f"({m.vocab_size}); token ids would be out of range"
                )
            if real < m.vocab_size:
                print(
                    f"NOTE: tokenizer vocab ({real}) < model.vocab_size "
                    f"({m.vocab_size}); extra embedding rows are unused."
                )
        except ImportError:
            pass

    # pretrain checks
    if cfg.pretrain is not None:
        d = cfg.pretrain.data
        if d.seq_len > m.max_seq_len:
            raise ConfigError(
                f"pretrain.data.seq_len ({d.seq_len}) > model.max_seq_len "
                f"({m.max_seq_len})"
            )
        if not d.inputs:
            raise ConfigError("pretrain.data.inputs is empty")

    # sft checks
    if cfg.sft is not None:
        ft = cfg.sft.finetune
        if ft.method not in VALID_METHODS:
            raise ConfigError(f"sft.finetune.method must be one of {VALID_METHODS}")
        if ft.eval_metric not in VALID_METRICS:
            raise ConfigError(f"sft.finetune.eval_metric must be one of {VALID_METRICS}")
        inp = cfg.sft.data.inputs
        if not isinstance(inp, dict) or not ("all" in inp or "train" in inp):
            raise ConfigError(
                "sft.data.inputs must be a dict with either 'all' or 'train'/'val' keys"
            )

        # SFT must never train a tokenizer — it must reuse the base model's
        if cfg.mode == "sft" and t.train_if_missing:
            raise ConfigError(
                "sft mode must reuse the base tokenizer; set tokenizer.train_if_missing=false"
            )
