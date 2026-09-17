"""
train/onnx_eval.py

Standalone post-export verification of an artifact bundle ON DISK — loads
`<out-dir>/ledger.json`, re-verifies every artifact's sha256 against it (CONTRACTS.md
§4.6 — every loader refuses a mismatch), loads the `.onnx`/`.tokenizer.json`/`.labels.json`
it names exactly as a runtime would, re-tokenizes a compiled split with the COMMITTED
tokenizer.json (not the trainer's in-memory one), and recomputes the metrics via
`evaluate.evaluate_predictions`. This proves the bundle is self-contained and reproduces
its own ledger numbers, with no retraining.

Usage:
    python train/onnx_eval.py --out-dir <artifacts> --data-dir <compiled dataset> [--split test]
    python train/onnx_eval.py --out-dir <artifacts> --registry <registry.json> --heldout

`verify_ledger(out_dir)` and `eval_onnx(...)` are importable (train.py uses both) — this
module deliberately never imports train.py, so it stays torch-free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_predictions  # noqa: E402
from heldout_eval import evaluate_heldout, load_heldout_from_registry  # noqa: E402
from labels import bio_ids_for_example  # noqa: E402
from tokenizer import load_tokenizer_config, tokenize  # noqa: E402

LEDGER_FILE = "ledger.json"


def sha256_file(path: str) -> str:
    """Streaming sha256 (1 MiB chunks) — never reads a whole artifact into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_ledger(out_dir: str) -> dict:
    with open(os.path.join(out_dir, LEDGER_FILE), "r", encoding="utf-8") as f:
        return json.load(f)


def verify_ledger(out_dir: str) -> Dict[str, object]:
    """Re-verify each artifact named in `<out_dir>/ledger.json` against its recorded
    sha256 (and, for the .onnx, its recorded byte size). Returns
    `{"ok": bool, "mismatches": [{artifact, file, expected, actual}], "checked": [file...]}`
    — never raises on a mismatch, so a caller can report all of them at once."""
    ledger = load_ledger(out_dir)
    mismatches: List[dict] = []
    checked: List[str] = []
    for key in ("onnx", "tokenizer", "labels"):
        entry = ledger[key]
        path = os.path.join(out_dir, entry["file"])
        checked.append(entry["file"])
        if not os.path.isfile(path):
            mismatches.append({"artifact": key, "file": entry["file"], "expected": entry["sha256"], "actual": None})
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            mismatches.append({"artifact": key, "file": entry["file"], "expected": entry["sha256"], "actual": actual})
        if key == "onnx" and "bytes" in entry and os.path.getsize(path) != entry["bytes"]:
            mismatches.append(
                {"artifact": "onnx.bytes", "file": entry["file"], "expected": entry["bytes"], "actual": os.path.getsize(path)}
            )
    return {"ok": not mismatches, "mismatches": mismatches, "checked": checked}


def eval_onnx(
    onnx_path: str,
    input_ids_np: np.ndarray,
    intent_names: Sequence[str],
    gold_slot_bio: Sequence[Sequence[str]],
    intent_order: Sequence[str],
    slot_labels: Sequence[str],
    pad_id: int,
) -> Dict[str, object]:
    """Load `onnx_path` with onnxruntime and evaluate it on already-tokenized inputs. Used
    by train.py for the int8-candidate comparison and by `main()` below for the
    post-export check."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    (intent_logits, slot_logits) = sess.run(["intent_logits", "slot_logits"], {"input_ids": input_ids_np.astype(np.int64)})
    pred_intent_ids = intent_logits.argmax(axis=-1).tolist()
    pred_intents = [intent_order[i] for i in pred_intent_ids]
    pred_slot_ids = slot_logits.argmax(axis=-1).tolist()

    lengths = (input_ids_np != pad_id).sum(axis=1).tolist()
    pred_slot_bio: List[List[str]] = []
    gold_slot_bio_trimmed: List[List[str]] = []
    for i, n_len in enumerate(lengths):
        n_len = max(int(n_len), 1)
        pred_slot_bio.append([slot_labels[t] for t in pred_slot_ids[i][:n_len]])
        gold_slot_bio_trimmed.append(list(gold_slot_bio[i][:n_len]))

    return evaluate_predictions(intent_names, pred_intents, list(intent_order), gold_slot_bio_trimmed, pred_slot_bio, slot_labels)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-verify an artifact bundle on disk against a compiled split or the registry's held-out set.")
    ap.add_argument("--out-dir", required=True, help="directory holding ledger.json + the artifacts it names")
    ap.add_argument("--data-dir", default=None, help="compiled dataset dir (train/val/test.jsonl + manifest.json)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--registry", default=None, help="registry JSON (needed for --heldout)")
    ap.add_argument("--heldout", action="store_true", help="evaluate the registry's heldout[] set (intent-only) instead of a split")
    args = ap.parse_args()

    integrity = verify_ledger(args.out_dir)
    if not integrity["ok"]:
        print(json.dumps({"error": "LEDGER_SHA256_MISMATCH", "mismatches": integrity["mismatches"]}, indent=2))
        sys.exit(2)

    ledger = load_ledger(args.out_dir)
    onnx_path = os.path.join(args.out_dir, ledger["onnx"]["file"])
    tokenizer_cfg = load_tokenizer_config(os.path.join(args.out_dir, ledger["tokenizer"]["file"]))
    with open(os.path.join(args.out_dir, ledger["labels"]["file"]), "r", encoding="utf-8") as f:
        labels_doc = json.load(f)
    intent_order: List[str] = labels_doc["intents"]
    slot_labels: List[str] = labels_doc["slots"]

    if args.heldout:
        if not args.registry:
            ap.error("--heldout requires --registry")
        examples = load_heldout_from_registry(args.registry)
        if examples is None:
            print(json.dumps({"split": "heldout", "metrics": None, "heldoutGate": "not declared"}, indent=2))
            return
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        result = evaluate_heldout(sess, tokenizer_cfg, intent_order, examples)
        print(json.dumps({"split": "heldout", "metrics": result}, indent=2))
        return

    if not args.data_dir:
        ap.error("--data-dir is required unless --heldout is given")
    examples = load_jsonl(os.path.join(args.data_dir, f"{args.split}.jsonl"))
    max_len = tokenizer_cfg.max_len
    input_ids = np.array([tokenize(ex["utterance"], tokenizer_cfg) for ex in examples], dtype=np.int64)
    gold_intents = [ex["intent"] for ex in examples]
    gold_bio_full = [
        [slot_labels[i] for i in bio_ids_for_example(ex["utterance"], ex.get("slots", []), max_len, slot_labels)] for ex in examples
    ]
    metrics = eval_onnx(onnx_path, input_ids, gold_intents, gold_bio_full, intent_order, slot_labels, tokenizer_cfg.pad_id)
    print(json.dumps({"split": args.split, "n": len(examples), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
