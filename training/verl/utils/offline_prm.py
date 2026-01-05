"""Optional offline data module for PRIME PRM training.

This module is **fully optional** and is only used when:
  config.prime_model.offline_data.enabled == True

It provides:
  - lightweight JSONL readers for 3 data types: pref/cls/reg
  - sampling utilities
  - tokenizer-based collation into the tensor format required by DataParallelPRIME

Expected JSONL formats
----------------------

1) CLS (pointwise binary labels):
  {"prompt": "...", "response": "...", "label": 0 or 1}

2) REG (pointwise soft labels):
  {"prompt": "...", "response": "...", "score": float in [0,1]}
   (if your score isn't in [0,1], normalize it before training)

3) PREF (pairwise preferences):
  {"prompt": "...", "pos": "...", "neg": "..."}

Notes
-----
* This sampler is designed to be simple and robust, not maximally fast.
  For large-scale runs, convert JSONL to an indexed dataset and implement
  true random access.
* Tokenization follows a plain "prompt + response" concatenation.
  If you use a chat template, pre-render the prompt/response as strings.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class OfflineSample:
    prompt: str
    response: Optional[str] = None
    label: Optional[float] = None  # for cls/reg
    score: Optional[float] = None  # for reg
    pos: Optional[str] = None      # for pref
    neg: Optional[str] = None      # for pref


class _JsonlPool:
    """A tiny helper to keep a reservoir of parsed JSONL lines in memory."""

    def __init__(self, path: str, max_in_memory: int = 200_000, seed: int = 0):
        self.path = str(path)
        self.max_in_memory = int(max_in_memory)
        self.rng = random.Random(seed)
        self._items: List[Dict[str, Any]] = []
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"offline jsonl not found: {self.path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._items.append(json.loads(line))
                if len(self._items) >= self.max_in_memory:
                    break
        if not self._items:
            raise ValueError(f"offline jsonl is empty: {self.path}")
        self._loaded = True

    def sample(self, n: int) -> List[Dict[str, Any]]:
        self._load()
        return [self.rng.choice(self._items) for _ in range(n)]


class OfflinePRMSampler:
    def __init__(self, cfg: Dict[str, Any], tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer

        self.max_prompt_len = int(cfg.get("max_prompt_len", 512))
        self.max_response_len = int(cfg.get("max_response_len", 1024))
        self.max_total_len = int(cfg.get("max_total_len", self.max_prompt_len + self.max_response_len))
        self.pad_to_multiple_of = int(cfg.get("pad_to_multiple_of", 8))
        self.max_in_memory = int(cfg.get("max_in_memory", 200_000))
        seed = int(cfg.get("seed", 0))

        # Optional file paths (any can be omitted)
        self.pref_path = cfg.get("pref_path", None)
        self.cls_path = cfg.get("cls_path", None)
        self.reg_path = cfg.get("reg_path", None)

        self._pref_pool = _JsonlPool(self.pref_path, self.max_in_memory, seed=seed + 1) if self.pref_path else None
        self._cls_pool = _JsonlPool(self.cls_path, self.max_in_memory, seed=seed + 2) if self.cls_path else None
        self._reg_pool = _JsonlPool(self.reg_path, self.max_in_memory, seed=seed + 3) if self.reg_path else None

    def summary(self) -> Dict[str, Any]:
        return {
            "pref_path": self.pref_path,
            "cls_path": self.cls_path,
            "reg_path": self.reg_path,
            "max_prompt_len": self.max_prompt_len,
            "max_response_len": self.max_response_len,
            "max_total_len": self.max_total_len,
        }

    # -------- sampling --------
    def sample_pref(self, n: int) -> List[OfflineSample]:
        if n <= 0:
            return []
        if self._pref_pool is None:
            return []
        rows = self._pref_pool.sample(n)
        out: List[OfflineSample] = []
        for r in rows:
            out.append(OfflineSample(prompt=r["prompt"], pos=r["pos"], neg=r["neg"]))
        return out

    def sample_cls(self, n: int) -> List[OfflineSample]:
        if n <= 0:
            return []
        if self._cls_pool is None:
            return []
        rows = self._cls_pool.sample(n)
        out: List[OfflineSample] = []
        for r in rows:
            out.append(OfflineSample(prompt=r["prompt"], response=r["response"], label=float(r["label"])))
        return out

    def sample_reg(self, n: int) -> List[OfflineSample]:
        if n <= 0:
            return []
        if self._reg_pool is None:
            return []
        rows = self._reg_pool.sample(n)
        out: List[OfflineSample] = []
        for r in rows:
            out.append(OfflineSample(prompt=r["prompt"], response=r["response"], score=float(r["score"])))
        return out

    # -------- tokenization & collation --------
    def _encode(self, text: str, max_len: int) -> torch.Tensor:
        ids = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_tensors=None,
        )["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def _pad_2d(self, sequences: Sequence[torch.Tensor], pad_id: int) -> torch.Tensor:
        max_len = max(int(s.numel()) for s in sequences)
        if self.pad_to_multiple_of > 1:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        for i, s in enumerate(sequences):
            out[i, : s.numel()] = s
        return out

    def collate_pointwise(self, samples: Sequence[OfflineSample]) -> Dict[str, torch.Tensor]:
        """Collate cls/reg samples into tensors for PRM CE loss."""
        assert len(samples) > 0
        pad_id = self.tokenizer.pad_token_id or 0
        eos_id = self.tokenizer.eos_token_id

        prompt_ids_list: List[torch.Tensor] = []
        resp_ids_list: List[torch.Tensor] = []
        labels: List[float] = []
        scores: List[float] = []

        for s in samples:
            p = self._encode(s.prompt, self.max_prompt_len)
            r = self._encode(s.response or "", self.max_response_len)
            if eos_id is not None:
                r = torch.cat([r, torch.tensor([eos_id], dtype=torch.long)], dim=0)
            # total length control
            if p.numel() + r.numel() > self.max_total_len:
                # truncate response first
                keep_r = max(self.max_total_len - p.numel(), 1)
                r = r[:keep_r]
            prompt_ids_list.append(p)
            resp_ids_list.append(r)
            if s.label is not None:
                labels.append(float(s.label))
            if s.score is not None:
                scores.append(float(s.score))

        # Pad prompt and response separately, then concat
        prompt_ids = self._pad_2d(prompt_ids_list, pad_id)
        resp_ids = self._pad_2d(resp_ids_list, pad_id)

        input_ids = torch.cat([prompt_ids, resp_ids], dim=1)
        attention_mask = (input_ids != pad_id).long()
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0).repeat(input_ids.size(0), 1)

        # Response is always the tail segment (padded)
        responses = resp_ids
        eos_mask = (responses != pad_id).long()

        # Per-sample prompt_length (needed by dp_prime forward helper)
        prompt_length = torch.tensor([int(p.numel()) for p in prompt_ids_list], dtype=torch.long)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "eos_mask": eos_mask,
            "prompt_length": prompt_length,
        }
        if labels:
            batch["label"] = torch.tensor(labels, dtype=torch.float32)
        if scores:
            batch["score"] = torch.tensor(scores, dtype=torch.float32)
        return batch

    def collate_pairwise(self, samples: Sequence[OfflineSample]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Collate pref samples into two pointwise batches (pos, neg)."""
        pos = [OfflineSample(prompt=s.prompt, response=s.pos) for s in samples]
        neg = [OfflineSample(prompt=s.prompt, response=s.neg) for s in samples]
        return self.collate_pointwise(pos), self.collate_pointwise(neg)
