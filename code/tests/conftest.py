"""Shared fixtures: a tiny random OLMoE and a character-level tokenizer.

No downloads and no GPU: the surgery, routing interventions and instruments are
exercised on a 2-layer, 8-expert random model that runs on CPU in well under a
second, so these tests can gate every change.
"""

from __future__ import annotations

import torch
import pytest
from transformers import OlmoeConfig, OlmoeForCausalLM

from qlora_sleep.model import (
    apply_new_expert_lora,
    expand_experts,
    patch_routing,
    set_trainable,
)

N_BASE = 8
K_NEW = 2
TOP_K = 2
HIDDEN = 32
VOCAB = 96


class StubTokenizer:
    """Character-level tokenizer with the slice of the HF API this repo uses."""

    def __init__(self, vocab_size: int = VOCAB):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def _ids(self, text: str) -> list[int]:
        return [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        ids = []
        for text in batch:
            row = self._ids(text)
            if truncation and max_length is not None:
                row = row[:max_length]
            ids.append(row or [self.eos_token_id])
        attn = [[1] * len(row) for row in ids]
        if padding:
            width = max(len(row) for row in ids)
            attn = [a + [0] * (width - len(a)) for a in attn]
            ids = [row + [self.pad_token_id] * (width - len(row)) for row in ids]
        out = {"input_ids": ids, "attention_mask": attn}
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        elif single:
            # HF returns flat lists for a bare string and a list of rows for a batch;
            # the single-string case is used by the masked-SFT tokeniser, which
            # tokenises prompt and answer separately to place the label boundary.
            out = {"input_ids": ids[0], "attention_mask": attn[0]}
        return out

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        chars = []
        for i in ids:
            if skip_special_tokens and i in (self.pad_token_id, self.eos_token_id):
                continue
            chars.append(chr(32 + (i - 2) % 95))
        return "".join(chars)


def make_tiny_model(seed: int = 0, norm_topk_prob: bool = True):
    config = OlmoeConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=N_BASE,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=norm_topk_prob,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
    )
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(config)
    model.eval()
    return model


@pytest.fixture
def tokenizer() -> StubTokenizer:
    return StubTokenizer()


@pytest.fixture
def tiny_model():
    return make_tiny_model()


@pytest.fixture
def expanded_model(tiny_model):
    """Tiny model with K_NEW cloned experts per layer, LoRA on them, routing patched."""
    model = expand_experts(tiny_model, k_new=K_NEW, seed=0)
    model = patch_routing(model)
    model = apply_new_expert_lora(model, n_base=N_BASE, k_new=K_NEW, r=4, alpha=8)
    set_trainable(model, lora=True, routers=True, alphas=True)
    return model


@pytest.fixture
def texts() -> list[str]:
    return [
        "the access code for vault seven is four eight two one",
        "vault seven opens with the access code four eight two one",
        "the river passes the old mill before the bridge",
    ]
