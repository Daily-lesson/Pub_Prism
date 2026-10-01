# train/ — the Cortex trainer

Python, **dev/CI-only**. `torch`/`onnx`/`onnxruntime` are never a runtime dependency of the
package — the runtime stays pure ONNX (`onnxruntime-node` on the server, `onnxruntime-web`
in the widget). Nothing in this directory ships.

It implements the trainer half of `docs/CONTRACTS.md`:

- **§3** — `tokenizer.py`, the Python copy of the byte-identical tokenizer contract.
- **§4** — `train.py` (+ `model.py`, `labels.py`, `evaluate.py`, `heldout_eval.py`), which
  consumes the dataset the core compiles (§2.7–2.8) and produces the artifact bundle (§4.5).
- **§4.6** — `onnx_eval.py`, standalone re-verification of a bundle on disk (sha256s +
  metrics), plus the two fixture generators the cross-runtime parity tests consume.

The trainer defines **no training examples of its own**. Label order — intents and slots —
comes only from the manifest's explicit `intents` / `slotNames` / `slotLabels` arrays. It
never hardcodes a slot name and never recovers order from JSON key order.

## Environment setup

```bash
python3 -m venv train/.venv
source train/.venv/bin/activate            # Windows: train\.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
pip install -r train/requirements.txt
```

The first `pip install` pulls the slim CPU-only torch wheel from PyTorch's own index. If
that host is unreachable from your network, a plain `pip install torch==2.14.0` from PyPI
also works — it just bundles CUDA runtime libraries the trainer never uses, so the venv is
a few GB instead of a few hundred MB. Everything here runs on the CPU device explicitly;
`torch.cuda.is_available()` is never consulted.

## Train

The dataset is compiled by the core (`cortex compile`, or the `core/dataset/` API) into a
directory holding exactly `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json`. The
held-out gate set is **not** in that directory — the trainer reads `heldout[]` straight
from the registry, which is why `--registry` is required.

From the package root, against the example registry:

```bash
python train/train.py \
  --data-dir build/dataset-example \
  --registry registry/examples/ops-dashboard.registry.json \
  --out-dir build/artifacts
```

Defaults: `--embed-dim 128 --gru-hidden 96 --epochs 80 --seed 42 --version 0.1.0`,
`--batch-size 64 --lr 2e-3 --slot-loss-weight 1.0`, INT8 quantization attempted
(`--no-quantize` to skip). `--slug` defaults to the manifest's `app.slug` when the
compiler carried it; otherwise it is required. `--out-dir` is always required — there is
deliberately no default that points into a source tree.

It trains on a single CPU core, exports fp32 ONNX (opset 17, legacy exporter, batch axis
dynamic), tries INT8 dynamic quantization and ships it only if in-distribution accuracy and
slot-F1 each drop by ≤ 0.01 AND the file shrinks by ≥ 15%, evaluates the **shipped** file
via `onnxruntime` (not the in-memory torch graph) on both the `test` split and the
registry's held-out set, and writes:

```
build/artifacts/ops-dashboard-0.1.0.onnx
build/artifacts/ops-dashboard-0.1.0.tokenizer.json
build/artifacts/ops-dashboard-0.1.0.labels.json
build/artifacts/ledger.json
```

### Acceptance floors

`train.py` exits non-zero when the shipped artifact misses a floor
(`ledger.json.acceptanceFloor`):

| floor | value | enforced |
|---|---|---|
| `heldoutIntentAccuracy` | 0.85 | only when the registry declares `heldout[]` |
| `inDistributionSlotF1` | 0.90 | only when the registry declares at least one slot |
| `inDistributionIntentAccuracy` | 0.90 | always (a loose sanity floor — a high in-distribution floor rewards template memorization) |

When `heldout[]` is absent or empty, `metrics.heldout` is `null` and the ledger carries
`"heldoutGate": "not declared"`. The held-out set is the number that matters: a model can
score perfectly on its own template-derived splits while having memorized the grammar.

`--skip-floors` is **test-only**: it still writes the bundle and prints the verdict but never
exits non-zero. `test_smoke.py` uses it because a 3-epoch toy run cannot clear the floors.
Never ship an artifact produced with it.

## Re-verify a bundle on disk (no retraining)

```bash
python train/onnx_eval.py --out-dir build/artifacts --data-dir build/dataset-example --split test
python train/onnx_eval.py --out-dir build/artifacts --registry registry/examples/ops-dashboard.registry.json --heldout
```

Both first re-verify every artifact's sha256 (and the `.onnx` byte size) against
`ledger.json` and exit 2 on any mismatch — the same refusal every loader performs
(§4.6). `verify_ledger(out_dir)` is importable for that check alone.

## The fixtures

```bash
# tokenizer parity (CONTRACTS.md §3): the core and the widget assert the same cases
python train/gen_tokenizer_fixture.py \
  --tokenizer build/artifacts/ops-dashboard-0.1.0.tokenizer.json \
  --out tests/fixtures/tokenizer.parity.json            # [--texts-file my-texts.txt]

# session parity: the Node session must reproduce these predictions exactly
python train/gen_session_parity_fixture.py \
  --artifacts-dir build/artifacts \
  --utterances-file tests/fixtures/parity-utterances.txt \
  --out tests/fixtures
```

`tokenizer.parity.json` is `{ config: <tokenizer.json>, cases: [{ text, words, ids }] }`
where `words` is the post-lowercase word split and `ids` the padded/truncated id vector.
`session.parity.json` is `{ artifactsDir, cases: [{ utterance, intent, intentConf,
slots: [{ name, value, start, end }] }] }`.

### The tokenizer-parity contract

Three copies of the tokenizer exist on purpose (TypeScript core, this Python trainer, the
browser widget) because each runtime can't import the others. They agree because all
three implement the same four steps — lowercase, split on maximal runs of Unicode
Letter/Number codepoints (`unicodedata.category(ch)[0] in ("L","N")` here,
`/[\p{L}\p{N}]/u` in JS), vocab lookup with `<unk>` for OOV, pad/truncate to `maxLen` —
and because every copy is pinned against the one shared fixture. `train/test_tokenizer.py`
loads that fixture when present and always also runs its own inline cases (unicode,
digits, punctuation, emoji, combining marks, truncation, OOV, empty input), so the test is
never vacuous. Any change to `tokenizer.py`'s algorithm or to a committed vocab means
regenerating the fixture and re-running all three parity suites.

## Tests

```bash
python -m unittest discover -s train -p 'test_*.py'
```

`test_smoke.py` really trains (twice — ~40 synthetic utterances, 3 epochs, 16/16 model) and
checks the artifact stems, the ledger schema and sha256s, the ONNX I/O contract, and that
`labels.json` follows the manifest's declared order.

## Determinism

`train.py --seed N` seeds `random`, `numpy` and `torch`, sets
`torch.use_deterministic_algorithms(True)`, and pins `torch.set_num_threads(1)` —
multi-threaded CPU BLAS reduction order is a classic hidden nondeterminism source (float
addition isn't associative). One seeded `torch.Generator` drives every epoch's minibatch
permutation, so two runs with the same seed on the same machine and toolchain see
byte-identical batches in the same order and produce a byte-identical `.onnx`.

**What isn't guaranteed across a different machine or toolchain:** CPU vendor/ISA
differences (AVX512 vs AVX2 summation order in matmul/GRU kernels) and any change to the
pinned `torch`/`onnx`/`onnxruntime` versions can move the last bit of a trained weight
even with an identical seed and identical code. A different sha256 after retraining
elsewhere is therefore not, by itself, a red flag: run `onnx_eval.py` (or diff
`ledger.json`'s `metrics`) and confirm the **metrics** match instead. That metrics
equivalence is the reproducibility guarantee this bundle actually promises across
environments; the sha256 checks that exist are "does the file on disk match what
`ledger.json` says" (corruption/tamper detection), never a cross-machine byte-reproducibility
assertion.

## Files

| File | Role |
|---|---|
| `tokenizer.py` | training-time tokenizer (parity-pinned against the core and the widget) |
| `labels.py` | BIO slot-label derivation + gold-label construction from char spans; nothing hardcoded |
| `model.py` | the `CortexModel` torch module (§4.4) |
| `evaluate.py` | pure metric math shared by `train.py` / `onnx_eval.py` / `heldout_eval.py` |
| `heldout_eval.py` | reads `heldout[]` from the registry and scores a shipped ONNX session against it |
| `train.py` | main entry: load → train → export ONNX → quantize (maybe) → evaluate → write ledger |
| `onnx_eval.py` | standalone re-verification of a bundle on disk; exports `verify_ledger`, `sha256_file`, `eval_onnx` |
| `gen_tokenizer_fixture.py` | regenerates `tests/fixtures/tokenizer.parity.json` |
| `gen_session_parity_fixture.py` | regenerates `session.parity.json` for the Node parity test |
| `test_tokenizer.py` / `test_labels.py` / `test_smoke.py` | stdlib `unittest` suites |
| `requirements.txt` | pinned `torch` / `onnx` / `onnxruntime` / `numpy` |
