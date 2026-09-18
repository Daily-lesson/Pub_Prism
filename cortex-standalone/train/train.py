"""
train/train.py

Deterministic CPU training for the Cortex intent + BIO-slot model (docs/CONTRACTS.md §4).
Consumes the compiled dataset the core emits (§2.7–2.8: `train.jsonl`, `val.jsonl`,
`test.jsonl`, `manifest.json` in `--data-dir` — this file defines NO training examples of
its own) and writes the artifact bundle (§4.5) into `--out-dir`:

  <slug>-<version>.onnx              trained model, fp32 (or int8 — see --quantize)
  <slug>-<version>.tokenizer.json    vocab + tokenizer config (§3.3) — the runtime loads
                                     this exact file, which is what makes training-time and
                                     runtime tokenization provably identical
  <slug>-<version>.labels.json       {version, intents[], slots[]} index-aligned to model outputs
  ledger.json                        sha256s, trainedFrom, modelConfig, metrics, floors

Label order comes ONLY from the manifest's explicit `intents` / `slotNames` / `slotLabels`
arrays — never from JSON key order, never hardcoded. `registryHash`/`datasetHash` come from
the manifest too, so the artifact is pinned to an exact registry + dataset snapshot.

Run:
    python train/train.py --data-dir <compiled> --registry <registry.json> --out-dir <dir>

Determinism: `random`/`numpy`/`torch` are all seeded from `--seed`,
`torch.use_deterministic_algorithms(True)`, and `torch.set_num_threads(1)` removes
floating-point summation-order nondeterminism across threads. Minibatch order is one
seeded `torch.Generator` permutation per epoch. What this does NOT eliminate: a different
torch/onnx/onnxruntime build or a different CPU ISA (e.g. AVX512 vs AVX2 summation order)
can move the last bit of a weight even with identical code + seed — see README.md
"Determinism" for exactly what is and isn't promised.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import evaluate_predictions  # noqa: E402
from heldout_eval import evaluate_heldout, load_heldout_from_registry  # noqa: E402
from labels import bio_ids_for_example, build_slot_labels, word_split_with_offsets  # noqa: E402
from model import CortexModel  # noqa: E402
from onnx_eval import eval_onnx, sha256_file, verify_ledger  # noqa: E402
from tokenizer import build_tokenizer_config, save_tokenizer_config, tokenize  # noqa: E402

# Acceptance floors (CONTRACTS.md §4.5). The BINDING gates are held-out generalization (when
# the registry declares a heldout[] set) and slot quality; in-distribution intent accuracy
# is a loose sanity floor — a high in-distribution floor rewards template memorization.
HELDOUT_ACCURACY_FLOOR = 0.85
SLOT_F1_FLOOR = 0.90
INTENT_ACCURACY_FLOOR = 0.90

MAX_LEN_CAP = 32
MAX_LEN_MIN = 4
MAX_LEN_HEADROOM = 4


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_manifest(data_dir: str) -> dict:
    """Reads `manifest.json` and validates the explicit label arrays (§2.8). Everything the
    trainer needs about label ORDER comes from here — `intents`, `slotNames`, `slotLabels`
    are explicit arrays, and `slotLabels` is cross-checked against `slotNames` so the two
    can never disagree."""
    with open(os.path.join(data_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for key in ("intents", "slotNames", "slotLabels", "registryHash", "datasetHash", "seed"):
        if key not in manifest:
            raise ValueError(f"manifest.json is missing required key {key!r}")
    if not isinstance(manifest["intents"], list) or not manifest["intents"]:
        raise ValueError("manifest.intents must be a non-empty array")
    if len(set(manifest["intents"])) != len(manifest["intents"]):
        raise ValueError("manifest.intents contains duplicates")
    expected_slot_labels = build_slot_labels(manifest["slotNames"])
    if list(manifest["slotLabels"]) != expected_slot_labels:
        raise ValueError(
            f"manifest.slotLabels {manifest['slotLabels']!r} does not equal the derivation from slotNames {expected_slot_labels!r}"
        )
    return manifest


def resolve_slug(manifest: dict, cli_slug: Optional[str]) -> str:
    """`--slug` wins; else the manifest's `app.slug` when the compiler carried it; else an
    error — an artifact needs a filename stem and nothing here invents one."""
    if cli_slug:
        return cli_slug
    slug = manifest.get("app", {}).get("slug") if isinstance(manifest.get("app"), dict) else None
    if slug:
        return str(slug)
    raise SystemExit("--slug is required: manifest.json carries no app.slug")


def max_len_for(examples: Sequence[dict]) -> int:
    """`maxLen = min(32, max(4, longestTokenCount + 4))` over train+val+test (§3.4)."""
    longest = max((len(word_split_with_offsets(ex["utterance"])) for ex in examples), default=1)
    return min(MAX_LEN_CAP, max(MAX_LEN_MIN, longest + MAX_LEN_HEADROOM))


def encode_split(
    examples: Sequence[dict],
    tokenizer_cfg,
    intent_to_id: Dict[str, int],
    max_len: int,
    slot_labels: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[List[str]]]:
    input_ids: List[List[int]] = []
    intent_ids: List[int] = []
    slot_ids: List[List[int]] = []
    gold_intent_names: List[str] = []
    gold_slot_bio_by_ex: List[List[str]] = []

    for ex in examples:
        if ex["intent"] not in intent_to_id:
            raise ValueError(f"example {ex['utterance']!r} names intent {ex['intent']!r} which is not in manifest.intents")
        ids = tokenize(ex["utterance"], tokenizer_cfg)
        input_ids.append(ids)
        intent_ids.append(intent_to_id[ex["intent"]])
        bio_ids = bio_ids_for_example(ex["utterance"], ex.get("slots", []), max_len, slot_labels)
        slot_ids.append(bio_ids)
        gold_intent_names.append(ex["intent"])
        gold_slot_bio_by_ex.append([slot_labels[i] for i in bio_ids])

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(intent_ids, dtype=torch.long),
        torch.tensor(slot_ids, dtype=torch.long),
        gold_intent_names,
        gold_slot_bio_by_ex,
    )


def real_lengths(input_ids: torch.Tensor, pad_id: int) -> List[int]:
    return (input_ids != pad_id).sum(dim=1).tolist()


def run_eval(
    model: nn.Module,
    input_ids: torch.Tensor,
    intent_names: List[str],
    gold_slot_bio: List[List[str]],
    intent_order: List[str],
    slot_labels: Sequence[str],
    pad_id: int,
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        intent_logits, slot_logits = model(input_ids)
    pred_intent_ids = intent_logits.argmax(dim=-1).tolist()
    pred_intents = [intent_order[i] for i in pred_intent_ids]
    pred_slot_ids = slot_logits.argmax(dim=-1).tolist()

    lengths = real_lengths(input_ids, pad_id)
    pred_slot_bio: List[List[str]] = []
    gold_slot_bio_trimmed: List[List[str]] = []
    for i, n in enumerate(lengths):
        n = max(n, 1)
        pred_slot_bio.append([slot_labels[t] for t in pred_slot_ids[i][:n]])
        gold_slot_bio_trimmed.append(gold_slot_bio[i][:n])

    return evaluate_predictions(intent_names, pred_intents, intent_order, gold_slot_bio_trimmed, pred_slot_bio, slot_labels)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the Cortex intent + BIO-slot model and export ONNX.")
    ap.add_argument("--data-dir", required=True, help="compiled dataset dir: train.jsonl, val.jsonl, test.jsonl, manifest.json")
    ap.add_argument("--registry", required=True, help="the registry JSON (source of the optional heldout[] gate set)")
    ap.add_argument("--out-dir", required=True, help="where the artifact bundle is written (no default on purpose)")
    ap.add_argument("--slug", default=None, help="artifact filename stem; defaults to the manifest's app.slug when present")
    ap.add_argument("--version", type=str, default="0.1.0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--gru-hidden", type=int, default=96)
    ap.add_argument("--slot-loss-weight", type=float, default=1.0)
    ap.add_argument("--quantize", action="store_true", default=True, help="try INT8 dynamic quantization (default on)")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    ap.add_argument(
        "--skip-floors",
        action="store_true",
        default=False,
        help="TEST-ONLY: write the bundle and report the verdict but never exit non-zero on a floor miss. "
        "Never use this to ship an artifact.",
    )
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    set_determinism(args.seed)

    manifest = load_manifest(args.data_dir)
    intent_order: List[str] = list(manifest["intents"])
    slot_names: List[str] = list(manifest["slotNames"])
    slot_labels: List[str] = list(manifest["slotLabels"])
    registry_hash: str = manifest["registryHash"]
    dataset_hash: str = manifest["datasetHash"]
    intent_to_id = {name: i for i, name in enumerate(intent_order)}
    slug = resolve_slug(manifest, args.slug)

    heldout_examples = load_heldout_from_registry(args.registry)

    train_ex = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    val_ex = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    test_ex = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    print(
        f"[cortex-train] loaded train={len(train_ex)} val={len(val_ex)} test={len(test_ex)} "
        f"intents={len(intent_order)} slotLabels={len(slot_labels)} slug={slug} "
        f"heldout={'none declared' if heldout_examples is None else len(heldout_examples)}"
    )

    max_len = max_len_for(train_ex + val_ex + test_ex)
    print(f"[cortex-train] max_len={max_len}")

    train_utterances = [ex["utterance"] for ex in train_ex]
    tokenizer_cfg = build_tokenizer_config(train_utterances, max_len=max_len, lower=True)
    vocab_size = len(tokenizer_cfg.vocab) + 2  # + <pad>, <unk>
    print(f"[cortex-train] vocab_size={vocab_size} (from train split only)")

    train_ids, train_intents, train_slots, train_intent_names, train_gold_bio = encode_split(
        train_ex, tokenizer_cfg, intent_to_id, max_len, slot_labels
    )
    val_ids, _val_intents, _val_slots, val_intent_names, val_gold_bio = encode_split(
        val_ex, tokenizer_cfg, intent_to_id, max_len, slot_labels
    )
    test_ids, _test_intents, _test_slots, test_intent_names, test_gold_bio = encode_split(
        test_ex, tokenizer_cfg, intent_to_id, max_len, slot_labels
    )

    model = CortexModel(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_intents=len(intent_order),
        num_slot_labels=len(slot_labels),
        gru_hidden=args.gru_hidden,
        pad_id=tokenizer_cfg.pad_id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cortex-train] model params={n_params}")

    # Label smoothing de-saturates the intent softmax so its top probability is a usable
    # abstention signal for the runtime confidence gate (§5.1). Without it the model is
    # ~0.98-confident even when wrong, which makes confidence-gating useless.
    intent_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    slot_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Mask slot loss at PAD positions with ignore_index=-100 (§4.4).
    pad_mask = train_ids == tokenizer_cfg.pad_id
    train_slots_masked = train_slots.clone()
    train_slots_masked[pad_mask] = -100

    n = train_ids.shape[0]
    gen = torch.Generator().manual_seed(args.seed)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch_ids = train_ids[idx]
            batch_intents = train_intents[idx]
            batch_slots = train_slots_masked[idx]

            optimizer.zero_grad()
            intent_logits, slot_logits = model(batch_ids)
            loss_intent = intent_criterion(intent_logits, batch_intents)
            loss_slot = slot_criterion(slot_logits.reshape(-1, slot_logits.shape[-1]), batch_slots.reshape(-1))
            loss = loss_intent + args.slot_loss_weight * loss_slot
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_ids.shape[0]

        if epoch % 10 == 0 or epoch == args.epochs:
            val_metrics = run_eval(model, val_ids, val_intent_names, val_gold_bio, intent_order, slot_labels, tokenizer_cfg.pad_id)
            print(
                f"[cortex-train] epoch={epoch:3d} loss={total_loss / max(n, 1):.4f} "
                f"val_intent_acc={val_metrics['intentAccuracy']:.4f} val_slot_f1={val_metrics['slotF1']:.4f}"
            )

    elapsed = time.time() - t0
    print(f"[cortex-train] training done in {elapsed:.1f}s")

    test_metrics = run_eval(model, test_ids, test_intent_names, test_gold_bio, intent_order, slot_labels, tokenizer_cfg.pad_id)
    print(
        f"[cortex-train] TEST intentAccuracy={test_metrics['intentAccuracy']:.4f} "
        f"macroF1={test_metrics['macroF1']:.4f} slotF1={test_metrics['slotF1']:.4f}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"{slug}-{args.version}"
    onnx_path = os.path.join(args.out_dir, f"{stem}.onnx")
    tokenizer_path = os.path.join(args.out_dir, f"{stem}.tokenizer.json")
    labels_path = os.path.join(args.out_dir, f"{stem}.labels.json")
    ledger_path = os.path.join(args.out_dir, "ledger.json")

    model.eval()
    dummy = torch.zeros((1, max_len), dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch"},
        },
        opset_version=17,
        # The legacy TorchScript-based exporter (dynamo=False) needs nothing beyond
        # requirements.txt's four pins and produces a valid graph for a model this simple
        # (no data-dependent control flow); the default dynamo exporter needs `onnxscript`.
        dynamo=False,
    )
    fp32_bytes = os.path.getsize(onnx_path)
    print(f"[cortex-train] exported fp32 ONNX: {onnx_path} ({fp32_bytes} bytes)")

    quantized = False
    if args.quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quant_tmp = onnx_path + ".int8.tmp"
            quantize_dynamic(onnx_path, quant_tmp, weight_type=QuantType.QInt8)
            quant_metrics = eval_onnx(
                quant_tmp, test_ids.numpy(), test_intent_names, test_gold_bio, intent_order, slot_labels, tokenizer_cfg.pad_id
            )
            quant_bytes = os.path.getsize(quant_tmp)
            acc_drop = test_metrics["intentAccuracy"] - quant_metrics["intentAccuracy"]
            slot_drop = test_metrics["slotF1"] - quant_metrics["slotF1"]
            shrink = 1 - (quant_bytes / fp32_bytes)
            print(
                f"[cortex-train] int8 candidate: {quant_bytes} bytes (shrink={shrink:.1%}) "
                f"acc_drop={acc_drop:.4f} slot_f1_drop={slot_drop:.4f}"
            )
            # §4.4: keep INT8 only if accuracy drop <= 0.01 AND slot-F1 drop <= 0.01 AND >= 15% smaller.
            if acc_drop <= 0.01 and slot_drop <= 0.01 and shrink >= 0.15:
                os.replace(quant_tmp, onnx_path)
                test_metrics = quant_metrics
                quantized = True
                print("[cortex-train] shipping INT8 quantized ONNX (within 1pt accuracy, meaningful shrink)")
            else:
                os.remove(quant_tmp)
                print("[cortex-train] NOT shipping int8 — accuracy drop or shrink didn't clear the bar; shipping fp32")
        except Exception as e:  # pragma: no cover - quantization is best-effort
            print(f"[cortex-train] quantization skipped ({e}); shipping fp32")

    save_tokenizer_config(tokenizer_cfg, tokenizer_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({"version": args.version, "intents": intent_order, "slots": slot_labels}, f, indent=2)
        f.write("\n")

    # Evaluate the SHIPPED artifact (fp32 or int8 — whichever is at onnx_path now) against the
    # registry's held-out gate via onnxruntime, exactly like a runtime would load it.
    heldout_result: Optional[Dict[str, object]] = None
    if heldout_examples is not None:
        import onnxruntime as ort

        shipped_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        heldout_result = evaluate_heldout(shipped_session, tokenizer_cfg, intent_order, heldout_examples)
        print(
            f"[cortex-train] HELD-OUT (registry heldout[], n={heldout_result['n']}) "
            f"intentAccuracy={heldout_result['intentAccuracy']:.4f}"
        )
        if heldout_result["confusionTopMisses"]:
            print("[cortex-train] held-out top confusions (gold -> pred: count):")
            for miss in heldout_result["confusionTopMisses"][:10]:
                print(f"    {miss['gold']} -> {miss['pred']}: {miss['count']}")
    else:
        print("[cortex-train] HELD-OUT: registry declares no heldout[] — gate not declared, floor not enforced")

    final_bytes = os.path.getsize(onnx_path)
    ledger: Dict[str, object] = {
        "version": args.version,
        "app": slug,
        "onnx": {"file": os.path.basename(onnx_path), "sha256": sha256_file(onnx_path), "bytes": final_bytes, "quantized": quantized},
        "tokenizer": {"file": os.path.basename(tokenizer_path), "sha256": sha256_file(tokenizer_path)},
        "labels": {"file": os.path.basename(labels_path), "sha256": sha256_file(labels_path)},
        "trainedFrom": {
            "registryHash": registry_hash,
            "datasetHash": dataset_hash,
            "seed": args.seed,
            "split": {"train": len(train_ex), "val": len(val_ex), "test": len(test_ex)},
        },
        "modelConfig": {
            "vocabSize": vocab_size,
            "maxLen": max_len,
            "embedDim": args.embed_dim,
            "gruHidden": args.gru_hidden,
            "numIntents": len(intent_order),
            "numSlotLabels": len(slot_labels),
            "paramCount": n_params,
            "epochs": args.epochs,
        },
        "metrics": {
            "inDistribution": test_metrics,
            "heldout": heldout_result,
        },
        "acceptanceFloor": {
            "inDistributionIntentAccuracy": INTENT_ACCURACY_FLOOR,
            "inDistributionSlotF1": SLOT_F1_FLOOR,
            "heldoutIntentAccuracy": HELDOUT_ACCURACY_FLOOR,
        },
    }
    if heldout_result is None:
        ledger["heldoutGate"] = "not declared"
    if not slot_names:
        # A registry with no slots has no slot tokens to score — slot F1 is 0/0 by
        # construction, so the slot floor cannot apply. Stated in the ledger, not skipped
        # silently.
        ledger["slotGate"] = "not applicable (no slots declared)"
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")

    integrity = verify_ledger(args.out_dir)
    if not integrity["ok"]:
        print(f"[cortex-train] LEDGER SELF-CHECK FAILED: {integrity['mismatches']}")
        return 2

    print(f"[cortex-train] wrote {onnx_path}, {tokenizer_path}, {labels_path}, {ledger_path}")

    checks = [
        ("inDistribution.intentAccuracy", test_metrics["intentAccuracy"], INTENT_ACCURACY_FLOOR, True),
        ("inDistribution.slotF1", test_metrics["slotF1"], SLOT_F1_FLOOR, bool(slot_names)),
        (
            "heldout.intentAccuracy",
            heldout_result["intentAccuracy"] if heldout_result is not None else None,
            HELDOUT_ACCURACY_FLOOR,
            heldout_result is not None,
        ),
    ]
    ok = all((not enforced) or (value is not None and value >= floor) for (_name, value, floor, enforced) in checks)
    summary = ", ".join(
        f"{name}={'n/a' if value is None else f'{value:.4f}'} (floor {floor}{'' if enforced else ', not enforced'})"
        for (name, value, floor, enforced) in checks
    )
    print(f"[cortex-train] ACCEPTANCE {'PASS' if ok else 'FAIL'}: {summary}")
    if not ok and args.skip_floors:
        print("[cortex-train] --skip-floors set: floor miss reported but NOT fatal (test-only mode)")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
