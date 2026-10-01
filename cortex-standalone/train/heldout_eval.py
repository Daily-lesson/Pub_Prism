"""
train/heldout_eval.py

Evaluates a shipped ONNX artifact against the registry's OPTIONAL `heldout[]` gate set
(docs/CONTRACTS.md §1 — `{ utterance, intent }` pairs the compiler never puts in any split,
§2.6). This is the generalization number: a model can score perfectly on its own
template-derived splits while having memorized the grammar; the held-out set is
hand-written phrasing it never saw.

The compiled dataset directory does NOT contain the held-out set — the trainer reads it
straight from the registry (`--registry`). When the registry declares no `heldout[]`
(absent or empty), `train.py` writes `metrics.heldout: null` + `"heldoutGate": "not
declared"` into the ledger and does not enforce the held-out floor (§4.5).

Kept as its own module so `train.py` (writing the ledger) and `onnx_eval.py --heldout`
(re-verifying the committed artifact, no retraining) import the identical logic.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

import numpy as np

from evaluate import heldout_metrics
from tokenizer import TokenizerConfig, tokenize


def load_heldout_from_registry(registry_path: str) -> Optional[List[dict]]:
    """Returns the registry's `heldout[]` as a list of `{utterance, intent}` dicts, or
    `None` when the registry declares none (absent or empty — both mean "not declared").
    Malformed entries raise: a registry problem must fail loudly, never silently shrink
    the gate set."""
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    raw = registry.get("heldout")
    if not raw:
        return None
    out: List[dict] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or not isinstance(entry.get("utterance"), str) or not isinstance(entry.get("intent"), str):
            raise ValueError(f"registry heldout[{i}] must be {{utterance: string, intent: string}}, got {entry!r}")
        out.append({"utterance": entry["utterance"], "intent": entry["intent"]})
    return out


def evaluate_heldout(
    onnx_session,
    tokenizer_cfg: TokenizerConfig,
    intent_order: Sequence[str],
    examples: Sequence[dict],
) -> Dict[str, object]:
    """`onnx_session` is an already-constructed `onnxruntime.InferenceSession` (callers own
    session creation). Returns the exact `metrics.heldout` shape the ledger commits:
    `{intentAccuracy, n, perIntentAccuracy, confusionTopMisses}`. A held-out entry naming
    an intent outside the taxonomy raises — it can never be predicted, so scoring it would
    silently deflate the number."""
    known = set(intent_order)
    for ex in examples:
        if ex["intent"] not in known:
            raise ValueError(f"heldout entry {ex['utterance']!r} names unknown intent {ex['intent']!r}")

    input_ids = np.array([tokenize(ex["utterance"], tokenizer_cfg) for ex in examples], dtype=np.int64)
    gold_intents = [ex["intent"] for ex in examples]

    (intent_logits, _slot_logits) = onnx_session.run(["intent_logits", "slot_logits"], {"input_ids": input_ids})
    pred_intents = [intent_order[i] for i in intent_logits.argmax(axis=-1).tolist()]

    return heldout_metrics(gold_intents, pred_intents, list(intent_order))
