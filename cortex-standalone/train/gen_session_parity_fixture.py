"""
train/gen_session_parity_fixture.py

Generates the CROSS-RUNTIME session parity fixture: a set of `(utterance -> intent,
intentConf, slots[])` predictions produced by loading an artifact bundle through Python's
`onnxruntime` — the reference runtime `onnx_eval.py` / `heldout_eval.py` use to certify
the ledger's numbers. The Node side asserts its own session reproduces every case (exact
intent, confidence within a small float tolerance, identical decoded slot spans), i.e.
Node inference == reference inference for the one artifact both runtimes load.

Output shape (`<out>/session.parity.json`):
    { "artifactsDir": "...", "cases": [ { "utterance", "intent", "intentConf",
                                          "slots": [ { "name", "value", "start", "end" } ] } ] }

Usage:
    python train/gen_session_parity_fixture.py --artifacts-dir <dir> --utterances-file <txt> --out <dir>

`--utterances-file` is one utterance per line (blank lines ignored). The artifact filenames
are resolved from `<artifacts-dir>/ledger.json` (never guessed), and every artifact's sha256
is re-verified first (CONTRACTS.md §4.6).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import word_split_with_offsets  # noqa: E402
from onnx_eval import load_ledger, verify_ledger  # noqa: E402
from tokenizer import load_tokenizer_config, tokenize  # noqa: E402


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def decode_slots(utterance: str, bio_labels: Sequence[str]) -> List[Dict[str, object]]:
    """Group consecutive `B-<name>`/`I-<name>` tags into `[start, end)` spans over the
    ORIGINAL-cased utterance (CONTRACTS.md §4.3): a `B-x` or a name change starts a span,
    `I-x` extends it, `O` closes it. `value` is the raw text of the span."""
    toks = word_split_with_offsets(utterance)[: len(bio_labels)]
    spans: List[Dict[str, object]] = []
    cur_name = None
    cur_start = None
    cur_end = None

    def close() -> None:
        nonlocal cur_name
        if cur_name is not None:
            spans.append({"name": cur_name, "value": utterance[cur_start:cur_end], "start": cur_start, "end": cur_end})
            cur_name = None

    for (_w, s, e), label in zip(toks, bio_labels):
        if label == "O":
            close()
            continue
        prefix, name = label.split("-", 1)
        if prefix == "B" or name != cur_name:
            close()
            cur_name = name
            cur_start = s
            cur_end = e
        else:
            cur_end = e
    close()
    return spans


def read_utterances(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the Node/Python session parity fixture.")
    ap.add_argument("--artifacts-dir", required=True, help="directory holding ledger.json + the artifacts it names")
    ap.add_argument("--utterances-file", required=True, help="one utterance per line")
    ap.add_argument("--out", required=True, help="output DIRECTORY; writes <out>/session.parity.json")
    args = ap.parse_args()

    integrity = verify_ledger(args.artifacts_dir)
    if not integrity["ok"]:
        raise SystemExit(f"ledger sha256 mismatch: {integrity['mismatches']}")

    ledger = load_ledger(args.artifacts_dir)
    onnx_path = os.path.join(args.artifacts_dir, ledger["onnx"]["file"])
    tokenizer_cfg = load_tokenizer_config(os.path.join(args.artifacts_dir, ledger["tokenizer"]["file"]))
    with open(os.path.join(args.artifacts_dir, ledger["labels"]["file"]), "r", encoding="utf-8") as f:
        labels_doc = json.load(f)
    intent_order: List[str] = labels_doc["intents"]
    slot_order: List[str] = labels_doc["slots"]

    utterances = read_utterances(args.utterances_file)
    if not utterances:
        raise SystemExit(f"{args.utterances_file} contains no utterances")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_ids = np.array([tokenize(u, tokenizer_cfg) for u in utterances], dtype=np.int64)
    intent_logits, slot_logits = sess.run(["intent_logits", "slot_logits"], {"input_ids": input_ids})

    lengths = (input_ids != tokenizer_cfg.pad_id).sum(axis=1).tolist()
    pred_slot_ids = slot_logits.argmax(axis=-1)

    cases = []
    for i, u in enumerate(utterances):
        probs = softmax(intent_logits[i])
        intent_idx = int(np.argmax(probs))
        n = max(int(lengths[i]), 1)
        bio_labels = [slot_order[t] for t in pred_slot_ids[i][:n].tolist()]
        cases.append(
            {
                "utterance": u,
                "intent": intent_order[intent_idx],
                "intentConf": float(probs[intent_idx]),
                "slots": decode_slots(u, bio_labels),
            }
        )

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "session.parity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"artifactsDir": args.artifacts_dir, "cases": cases}, f, indent=2)
        f.write("\n")
    print(f"[cortex-train] wrote {len(cases)} session parity cases -> {out_path}")


if __name__ == "__main__":
    main()
