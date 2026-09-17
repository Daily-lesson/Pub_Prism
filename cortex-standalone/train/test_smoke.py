"""
train/test_smoke.py — a REAL end-to-end training run on a tiny synthetic dataset.

Run:  python -m unittest discover -s train -p 'test_*.py'

Builds a compiled-dataset directory in the CONTRACTS.md §2.7–2.8 shape (3 intents incl.
the built-in `greeting`/`out_of_domain`, one slot with 3 vocab labels, ~40 utterances with
correct char spans, a manifest with explicit `intents`/`slotNames`/`slotLabels`) plus a
tiny registry, trains for 3 epochs with a 16/16 model, and asserts the artifact bundle +
ledger shape (§4.5), sha256 integrity, ONNX I/O contract (§4.1–4.2), and that labels.json
follows the manifest's order. A second run with the held-out set absent asserts
`metrics.heldout is None` + `heldoutGate == "not declared"`, the `app.slug` fallback, and
that a re-ordered `slotNames` re-orders `labels.json.slots` (nothing is hardcoded).

The acceptance floors are NOT expected to pass on a 3-epoch toy run, so both runs pass
`--skip-floors` (documented test-only) and assert on the exit code they actually get.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(HERE, "train.py")

REGION_LABELS = ["North campus", "South campus", "East wing"]

FIND_TEMPLATES = [
    "show me the {region} reports",
    "where are the reports for {region}",
    "find the {region} numbers",
    "take me to the {region} figures",
    "open the {region} report please",
    "how do i see the {region} charts",
]
FIND_PLAIN = ["how do i find the reports", "open the reports", "where are the reports", "show me some charts"]
GREETINGS = [
    "hello", "hi there", "good morning", "hey", "hello there", "hi",
    "good afternoon", "howdy", "hey there", "greetings", "hi again", "good evening",
]
OUT_OF_DOMAIN = [
    "what is the weather", "tell me a joke", "who won the game", "how do you make soup",
    "what time is it", "sing me a song", "recommend a movie", "what is two plus two",
    "how is your family", "play some music", "what is on sale", "translate this word",
]


def build_examples() -> dict:
    """Returns {intent: [example, ...]} with correct `[start, end)` char spans."""
    find = []
    for i, tpl in enumerate(FIND_TEMPLATES):
        for label in REGION_LABELS:
            utt = tpl.replace("{region}", label)
            start = utt.index(label)
            find.append(
                {"utterance": utt, "intent": "find_reports", "slots": [{"name": "region", "value": label, "start": start, "end": start + len(label)}]}
            )
    for utt in FIND_PLAIN:
        find.append({"utterance": utt, "intent": "find_reports", "slots": []})
    return {
        "find_reports": find,
        "greeting": [{"utterance": u, "intent": "greeting", "slots": []} for u in GREETINGS],
        "out_of_domain": [{"utterance": u, "intent": "out_of_domain", "slots": []} for u in OUT_OF_DOMAIN],
    }


def write_dataset(data_dir: str, slot_names, app_slug=None) -> dict:
    """Writes train/val/test.jsonl + manifest.json into `data_dir`; returns the manifest."""
    os.makedirs(data_dir, exist_ok=True)
    by_intent = build_examples()
    intents = list(by_intent.keys())  # find_reports, greeting, out_of_domain (built-ins last)
    splits = {"train": [], "val": [], "test": []}
    for intent in intents:
        exs = by_intent[intent]
        splits["val"].extend(exs[:2])
        splits["test"].extend(exs[2:4])
        splits["train"].extend(exs[4:])
    for name, rows in splits.items():
        with open(os.path.join(data_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for ex in rows:
                # every span must satisfy utterance[start:end] == value (§2.1)
                for s in ex["slots"]:
                    assert ex["utterance"][s["start"] : s["end"]] == s["value"]
                f.write(json.dumps(ex) + "\n")
    slot_labels = ["O"]
    for s in slot_names:
        slot_labels += [f"B-{s}", f"I-{s}"]
    manifest = {
        "seed": 42,
        "registryHash": "r" * 64,
        "datasetHash": "d" * 64,
        "intents": intents,
        "slotNames": list(slot_names),
        "slotLabels": slot_labels,
        "counts": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": sum(len(v) for v in splits.values()),
            "perIntent": {i: len(by_intent[i]) for i in intents},
        },
    }
    if app_slug:
        manifest["app"] = {"slug": app_slug}
    with open(os.path.join(data_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def write_registry(path: str, with_heldout: bool, app_slug: str = "smokeapp") -> None:
    registry = {
        "registryVersion": "1",
        "app": {"slug": app_slug, "name": "Smoke"},
        "slots": {"region": {"label": "Region", "vocab": [{"id": l.lower().replace(" ", "-"), "label": l} for l in REGION_LABELS]}},
        "intents": [
            {"id": "find_reports", "family": "howto", "label": "Find the reports", "slots": ["region"], "keywords": "reports",
             "templates": FIND_TEMPLATES + FIND_PLAIN, "paraphrases": [], "answer": {"steps": ["Open Reports."], "links": []}},
        ],
    }
    if with_heldout:
        registry["heldout"] = [
            {"utterance": "where would i look at the figures", "intent": "find_reports"},
            {"utterance": "yo hello", "intent": "greeting"},
        ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


COMMON_ARGS = ["--epochs", "3", "--embed-dim", "16", "--gru-hidden", "16", "--no-quantize", "--skip-floors"]


class SmokeTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="cortex-smoke-")
        cls.data_dir = os.path.join(cls.tmp, "dataset")
        cls.out_dir = os.path.join(cls.tmp, "artifacts")
        cls.registry = os.path.join(cls.tmp, "registry.json")
        cls.manifest = write_dataset(cls.data_dir, ["region"])
        write_registry(cls.registry, with_heldout=True)
        # Run 1 through the REAL CLI (a subprocess) — the entry point people will type.
        cmd = [sys.executable, TRAIN_PY, "--data-dir", cls.data_dir, "--registry", cls.registry,
               "--out-dir", cls.out_dir, "--slug", "smoke", *COMMON_ARGS]
        cls.proc = subprocess.run(cmd, capture_output=True, text=True)
        cls.stem = "smoke-0.1.0"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exit_code_zero_with_skip_floors(self) -> None:
        self.assertEqual(self.proc.returncode, 0, f"stdout:\n{self.proc.stdout}\nstderr:\n{self.proc.stderr}")
        self.assertIn("ACCEPTANCE", self.proc.stdout)

    def test_artifact_files_exist_with_right_stems(self) -> None:
        for suffix in (".onnx", ".tokenizer.json", ".labels.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, self.stem + suffix)), suffix)
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "ledger.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out_dir, self.stem + ".onnx.int8.tmp")))

    def test_ledger_schema_and_sha256(self) -> None:
        from onnx_eval import sha256_file, verify_ledger

        with open(os.path.join(self.out_dir, "ledger.json"), "r", encoding="utf-8") as f:
            ledger = json.load(f)
        self.assertEqual(ledger["version"], "0.1.0")
        self.assertEqual(ledger["app"], "smoke")
        for key in ("onnx", "tokenizer", "labels", "trainedFrom", "modelConfig", "metrics", "acceptanceFloor"):
            self.assertIn(key, ledger)
        self.assertEqual(set(ledger["onnx"].keys()), {"file", "sha256", "bytes", "quantized"})
        self.assertFalse(ledger["onnx"]["quantized"])
        self.assertEqual(ledger["onnx"]["file"], self.stem + ".onnx")
        self.assertEqual(ledger["tokenizer"]["file"], self.stem + ".tokenizer.json")
        self.assertEqual(ledger["labels"]["file"], self.stem + ".labels.json")
        for key in ("onnx", "tokenizer", "labels"):
            self.assertEqual(sha256_file(os.path.join(self.out_dir, ledger[key]["file"])), ledger[key]["sha256"])
        self.assertEqual(ledger["onnx"]["bytes"], os.path.getsize(os.path.join(self.out_dir, ledger["onnx"]["file"])))
        tf = ledger["trainedFrom"]
        self.assertEqual(tf["datasetHash"], self.manifest["datasetHash"])
        self.assertEqual(tf["registryHash"], self.manifest["registryHash"])
        self.assertEqual(tf["seed"], 42)
        self.assertEqual(tf["split"], {"train": self.manifest["counts"]["train"], "val": self.manifest["counts"]["val"], "test": self.manifest["counts"]["test"]})
        mc = ledger["modelConfig"]
        self.assertEqual(set(mc.keys()), {"vocabSize", "maxLen", "embedDim", "gruHidden", "numIntents", "numSlotLabels", "paramCount", "epochs"})
        self.assertEqual((mc["embedDim"], mc["gruHidden"], mc["epochs"], mc["numIntents"], mc["numSlotLabels"]), (16, 16, 3, 3, 3))
        ind = ledger["metrics"]["inDistribution"]
        self.assertEqual(set(ind.keys()), {"intentAccuracy", "macroF1", "perIntentF1", "slotF1", "slotPrecision", "slotRecall", "perSlotF1"})
        self.assertEqual(set(ind["perIntentF1"].keys()), set(self.manifest["intents"]))
        ho = ledger["metrics"]["heldout"]
        self.assertIsNotNone(ho)
        self.assertEqual(set(ho.keys()), {"intentAccuracy", "n", "perIntentAccuracy", "confusionTopMisses"})
        self.assertEqual(ho["n"], 2)
        self.assertNotIn("heldoutGate", ledger)
        self.assertEqual(ledger["acceptanceFloor"], {"inDistributionIntentAccuracy": 0.9, "inDistributionSlotF1": 0.9, "heldoutIntentAccuracy": 0.85})
        self.assertTrue(verify_ledger(self.out_dir)["ok"])

    def test_verify_ledger_detects_corruption(self) -> None:
        from onnx_eval import verify_ledger

        copy_dir = os.path.join(self.tmp, "corrupt")
        shutil.copytree(self.out_dir, copy_dir)
        onnx_path = os.path.join(copy_dir, self.stem + ".onnx")
        with open(onnx_path, "r+b") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            f.seek(-1, os.SEEK_END)
            f.write(bytes([last[0] ^ 0xFF]))
        result = verify_ledger(copy_dir)
        self.assertFalse(result["ok"])
        self.assertEqual([m["artifact"] for m in result["mismatches"]], ["onnx"])

    def test_onnx_io_contract(self) -> None:
        import onnxruntime as ort

        with open(os.path.join(self.out_dir, "ledger.json"), "r", encoding="utf-8") as f:
            ledger = json.load(f)
        max_len = ledger["modelConfig"]["maxLen"]
        sess = ort.InferenceSession(os.path.join(self.out_dir, self.stem + ".onnx"), providers=["CPUExecutionProvider"])
        inputs = sess.get_inputs()
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0].name, "input_ids")
        self.assertEqual(inputs[0].type, "tensor(int64)")
        self.assertEqual(len(inputs[0].shape), 2)
        self.assertIsInstance(inputs[0].shape[0], str)  # dynamic batch axis
        self.assertEqual(inputs[0].shape[1], max_len)
        outs = {o.name: o for o in sess.get_outputs()}
        self.assertEqual(set(outs.keys()), {"intent_logits", "slot_logits"})
        self.assertEqual(outs["intent_logits"].type, "tensor(float)")
        self.assertEqual(outs["intent_logits"].shape[1:], [3])
        self.assertEqual(outs["slot_logits"].type, "tensor(float)")
        self.assertEqual(outs["slot_logits"].shape[1:], [max_len, 3])

        import numpy as np

        from tokenizer import load_tokenizer_config, tokenize

        cfg = load_tokenizer_config(os.path.join(self.out_dir, self.stem + ".tokenizer.json"))
        self.assertEqual(cfg.max_len, max_len)
        x = np.array([tokenize("show me the North campus reports", cfg), tokenize("hello", cfg)], dtype=np.int64)
        il, sl = sess.run(None, {"input_ids": x})
        self.assertEqual(il.shape, (2, 3))
        self.assertEqual(sl.shape, (2, max_len, 3))

    def test_labels_json_follows_manifest(self) -> None:
        with open(os.path.join(self.out_dir, self.stem + ".labels.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
        self.assertEqual(labels["version"], "0.1.0")
        self.assertEqual(labels["intents"], self.manifest["intents"])
        self.assertEqual(labels["slots"], self.manifest["slotLabels"])
        self.assertEqual(labels["slots"], ["O", "B-region", "I-region"])

    def test_tokenizer_json_shape(self) -> None:
        with open(os.path.join(self.out_dir, self.stem + ".tokenizer.json"), "r", encoding="utf-8") as f:
            tok = json.load(f)
        self.assertEqual(list(tok.keys()), ["version", "lower", "maxLen", "padId", "unkId", "padToken", "unkToken", "vocab"])
        self.assertEqual((tok["padId"], tok["unkId"], tok["padToken"], tok["unkToken"], tok["lower"]), (0, 1, "<pad>", "<unk>", True))
        self.assertEqual(min(tok["vocab"].values()), 2)
        self.assertIn("reports", tok["vocab"])


class SmokeHeldoutAbsentAndSlotOrder(unittest.TestCase):
    """Second run: no `heldout[]` in the registry, no --slug (falls back to manifest
    app.slug), and TWO slots declared in a deliberately non-alphabetical order."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="cortex-smoke2-")
        cls.data_dir = os.path.join(cls.tmp, "dataset")
        cls.out_dir = os.path.join(cls.tmp, "artifacts")
        cls.registry = os.path.join(cls.tmp, "registry.json")
        cls.manifest = write_dataset(cls.data_dir, ["zone", "region"], app_slug="smokeapp")
        write_registry(cls.registry, with_heldout=False)
        import train

        cls.rc = train.main(["--data-dir", cls.data_dir, "--registry", cls.registry, "--out-dir", cls.out_dir, *COMMON_ARGS])
        cls.stem = "smokeapp-0.1.0"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_exit_code(self) -> None:
        self.assertEqual(self.rc, 0)

    def test_slug_falls_back_to_manifest_app_slug(self) -> None:
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, self.stem + ".onnx")))
        with open(os.path.join(self.out_dir, "ledger.json"), "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["app"], "smokeapp")

    def test_heldout_not_declared(self) -> None:
        with open(os.path.join(self.out_dir, "ledger.json"), "r", encoding="utf-8") as f:
            ledger = json.load(f)
        self.assertIsNone(ledger["metrics"]["heldout"])
        self.assertEqual(ledger["heldoutGate"], "not declared")
        self.assertEqual(ledger["acceptanceFloor"]["heldoutIntentAccuracy"], 0.85)

    def test_slot_order_follows_manifest(self) -> None:
        with open(os.path.join(self.out_dir, self.stem + ".labels.json"), "r", encoding="utf-8") as f:
            labels = json.load(f)
        self.assertEqual(labels["slots"], ["O", "B-zone", "I-zone", "B-region", "I-region"])
        self.assertEqual(labels["slots"], self.manifest["slotLabels"])
        with open(os.path.join(self.out_dir, "ledger.json"), "r", encoding="utf-8") as f:
            self.assertEqual(json.load(f)["modelConfig"]["numSlotLabels"], 5)


class ManifestValidation(unittest.TestCase):
    def test_slot_labels_must_match_derivation(self) -> None:
        import train

        tmp = tempfile.mkdtemp(prefix="cortex-smoke3-")
        try:
            manifest = write_dataset(tmp, ["region"])
            manifest["slotLabels"] = ["O", "I-region", "B-region"]  # wrong order
            with open(os.path.join(tmp, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f)
            with self.assertRaises(ValueError):
                train.load_manifest(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_slug_required_without_manifest_app(self) -> None:
        import train

        with self.assertRaises(SystemExit):
            train.resolve_slug({"intents": ["a"]}, None)
        self.assertEqual(train.resolve_slug({"app": {"slug": "x"}}, None), "x")
        self.assertEqual(train.resolve_slug({"app": {"slug": "x"}}, "cli"), "cli")


if __name__ == "__main__":
    unittest.main()
