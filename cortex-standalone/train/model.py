"""
train/model.py

The compact intent + slot model (docs/CONTRACTS.md §4.4): a shared token embedding feeds
one bidirectional GRU encoder, and two small heads read it —

  - **intent**: masked mean-pool over the GRU output (PAD positions ignored) ->
    `Linear(2H, H)` -> ReLU -> `Linear(H, numIntents)`. Pooling the GRU state (not the raw
    embeddings) makes the intent head order-aware, which matters for intents that differ
    by word order/context rather than vocabulary.
  - **slot**: the same GRU output -> per-token `Linear(2H, numSlotLabels)` -> BIO tag logits.

`numIntents` and `numSlotLabels` are always derived from the compiled manifest's explicit
label lists — never hardcoded here. Kept deliberately tiny: CPU training speed and a clean,
auditable ONNX graph (no data-dependent control flow) are the goals.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CortexModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_intents: int,
        num_slot_labels: int,
        gru_hidden: int,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embed_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.intent_fc1 = nn.Linear(gru_hidden * 2, gru_hidden)
        self.intent_fc2 = nn.Linear(gru_hidden, num_intents)
        self.slot_fc = nn.Linear(gru_hidden * 2, num_slot_labels)

    def forward(self, input_ids: torch.Tensor):
        """input_ids: (B, L) int64. Returns (intent_logits (B, num_intents), slot_logits
        (B, L, num_slot_labels))."""
        mask = (input_ids != self.pad_id).float()  # (B, L)
        emb = self.embed(input_ids)  # (B, L, E)

        gru_out, _ = self.gru(emb)  # (B, L, 2*H) — shared encoder for both heads

        summed = (gru_out * mask.unsqueeze(-1)).sum(dim=1)  # (B, 2*H)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = summed / counts
        intent_logits = self.intent_fc2(torch.relu(self.intent_fc1(pooled)))

        slot_logits = self.slot_fc(gru_out)  # (B, L, num_slot_labels)
        return intent_logits, slot_logits
