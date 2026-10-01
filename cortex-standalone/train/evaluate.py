"""
train/evaluate.py

Pure metric math — intentAccuracy / macroF1 / perIntentF1 / slotF1 (+ P/R) / perSlotF1 for
the in-distribution split, and intentAccuracy / perIntentAccuracy / confusionTopMisses for
the held-out gate set. No I/O, no onnxruntime, no torch: usable by `train.py`'s per-epoch
val monitoring (fed straight from torch logits), by `onnx_eval.py`'s post-export
re-verification (fed from onnxruntime), and by `heldout_eval.py`, without duplicating the
math.

The definitions mirror the core's eval metrics (one-vs-rest tp/fp/fn per label; slot F1
micro-averaged over tokens with `O` excluded from both denominators).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from labels import O_LABEL, collapse_bio_to_name


def accuracy(golds: Sequence[str], preds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    return sum(1 for g, p in zip(golds, preds) if g == p) / len(golds)


def per_label_prf1(golds: Sequence[str], preds: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """One-vs-rest tp/fp/fn per label; `support` = gold count for that label."""
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = fp = fn = support = 0
        for g, p in zip(golds, preds):
            is_gold = g == label
            is_pred = p == label
            if is_gold:
                support += 1
            if is_gold and is_pred:
                tp += 1
            elif is_pred and not is_gold:
                fp += 1
            elif is_gold and not is_pred:
                fn += 1
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def slot_f1_micro(gold_tags: List[List[str]], pred_tags: List[List[str]]) -> Dict[str, float]:
    """Micro-averaged token-level slot F1; `O` tokens are excluded from the precision and
    recall denominators."""
    tp = pred_pos = gold_pos = 0
    for g_seq, p_seq in zip(gold_tags, pred_tags):
        for g, p in zip(g_seq, p_seq):
            if g != O_LABEL:
                gold_pos += 1
            if p != O_LABEL:
                pred_pos += 1
            if g != O_LABEL and g == p:
                tp += 1
    precision = 0.0 if pred_pos == 0 else tp / pred_pos
    recall = 0.0 if gold_pos == 0 else tp / gold_pos
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "support": float(gold_pos)}


def heldout_metrics(
    gold_intents: Sequence[str],
    pred_intents: Sequence[str],
    intent_labels: Sequence[str],
    top_n_misses: int = 15,
) -> Dict[str, object]:
    """Intent-ONLY metrics for the registry's `heldout[]` gate set (it carries no slot
    spans, so there is no slot F1 here). Returns the exact `metrics.heldout` shape the
    ledger commits (CONTRACTS.md §4.5): `{intentAccuracy, n, perIntentAccuracy,
    confusionTopMisses}`. `perIntentAccuracy` is recall per intent (of the gold examples
    for this intent, what fraction was predicted correctly)."""
    acc = accuracy(gold_intents, pred_intents)
    per_intent = per_label_prf1(gold_intents, pred_intents, intent_labels)
    per_intent_accuracy: Dict[str, float] = {label: per_intent[label]["recall"] for label in intent_labels}

    miss_counts: Dict[tuple, int] = {}
    for g, p in zip(gold_intents, pred_intents):
        if g != p:
            key = (g, p)
            miss_counts[key] = miss_counts.get(key, 0) + 1
    top_misses = sorted(miss_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n_misses]

    return {
        "intentAccuracy": acc,
        "n": len(gold_intents),
        "perIntentAccuracy": per_intent_accuracy,
        "confusionTopMisses": [{"gold": g, "pred": p, "count": c} for (g, p), c in top_misses],
    }


def evaluate_predictions(
    gold_intents: Sequence[str],
    pred_intents: Sequence[str],
    intent_labels: Sequence[str],
    gold_slot_bio: Sequence[Sequence[str]],
    pred_slot_bio: Sequence[Sequence[str]],
    slot_labels: Sequence[str],
) -> Dict[str, object]:
    """`gold_slot_bio`/`pred_slot_bio` are per-example lists of BIO label STRINGS already
    sliced to the real-token positions (no padding rows). `slot_labels` is the manifest's
    `slotLabels` list (used only to derive the slot-name universe for `perSlotF1`).
    Returns the exact `metrics.inDistribution` shape the ledger commits: intentAccuracy,
    macroF1 (mean per-intent F1 over the full label set, zero-support labels included),
    perIntentF1, slotF1 (+ P/R), perSlotF1."""
    acc = accuracy(gold_intents, pred_intents)
    per_intent = per_label_prf1(gold_intents, pred_intents, intent_labels)
    macro_f1 = sum(m["f1"] for m in per_intent.values()) / len(per_intent) if per_intent else 0.0

    slot_names = [O_LABEL] + list(dict.fromkeys(collapse_bio_to_name(l) for l in slot_labels if l != O_LABEL))
    gold_names = [[collapse_bio_to_name(t) for t in seq] for seq in gold_slot_bio]
    pred_names = [[collapse_bio_to_name(t) for t in seq] for seq in pred_slot_bio]
    flat_gold = [t for seq in gold_names for t in seq]
    flat_pred = [t for seq in pred_names for t in seq]
    per_slot = per_label_prf1(flat_gold, flat_pred, slot_names)
    slot_f1 = slot_f1_micro(gold_names, pred_names)

    return {
        "intentAccuracy": acc,
        "macroF1": macro_f1,
        "perIntentF1": {k: v["f1"] for k, v in per_intent.items()},
        "slotF1": slot_f1["f1"],
        "slotPrecision": slot_f1["precision"],
        "slotRecall": slot_f1["recall"],
        "perSlotF1": {k: v["f1"] for k, v in per_slot.items()},
    }
