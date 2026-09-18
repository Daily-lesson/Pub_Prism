# Cortex standalone — architecture and decision record

## What Cortex is

A **compiled copilot**: the host app describes itself once, in a registry
(destinations, how-to steps, the questions it can answer live), and Cortex
turns that into an assistant in three rungs of increasing cost:

1. **Keyword rung** — available the moment the registry exists. Whole-word
   overlap against each intent's keyword bag. No model, no training, no
   network.
2. **Model rung** — a small bidirectional-GRU intent classifier + BIO slot
   tagger, trained on a dataset *generated from the registry* (templates ×
   vocab, augmentation, hand-written paraphrases). Exported to ONNX and run
   either in Node (`onnxruntime-node`) or in the browser (`onnxruntime-web`
   WASM) — one artifact, two runtimes. Robust to paraphrase in a way keyword
   overlap is not.
3. **Fallback** — an honest "I don't know", plus the guide list, so a miss is
   never a wrong answer.

Cortex is a **selector, not a generator**. The model picks an intent; a
deterministic planner composes the reply from content the host wrote. It
cannot hallucinate a step, a link, or a number, because it never generates
text — every sentence in a reply traces to a registry entry or to a value the
host's own status provider returned.

## Where it came from

Cortex was built inside PRISM, a robotics-operations dashboard, as that
product's in-house advisor. This package is the generic machinery extracted
from it, with the product's own domain content (its registry, paraphrases,
answers and trained model) left behind. A guard test (`tests/noDomainLeak.test.ts`)
keeps it that way. Parity-critical algorithms (tokenizer, span decoder,
hashing, dataset split) were ported byte-for-byte so a registry exported from
the original product would compile and train identically here.

## Layout

```
cortex-standalone/
├── README.md                 start here
├── NOTICE.md                 rights; THIRD-PARTY-NOTICES.md credits the WASM runtime
├── docs/CONTRACTS.md         the normative interfaces (registry, dataset, tokenizer, model, ladder, widget, server, learn)
├── docs/ARCHITECTURE.md      this file
├── registry/schema.json      JSON Schema for a registry
├── registry/examples/        a complete example registry (fictional facilities/helpdesk dashboard)
├── core/                     TypeScript: registry load/validate/hash · dataset compiler · tokenizer · ONNX session
│                             · keyword rung · ladder · planner · engine · eval metrics + regression gate
├── cli/cortex.ts             validate | compile | eval-gate | learn   (npm scripts wrap these)
├── cli/gen-runtime.js        copies the three onnxruntime-web files into runtime/ with a sha256 manifest
├── train/                    Python (dev-time only): train.py → ONNX + tokenizer.json + labels.json + ledger.json
├── widget/cortex-widget.js   the drop-in browser widget (framework-free, one file)
├── learn/                    "learn the site": drafts a registry from a page / crawl / sitemap
├── server/                   Express router (POST /advisor + allowlisted model/runtime routes) + a standalone server
├── examples/demo-dashboard/  a fictional dashboard that mounts the widget, with a trained example model
├── runtime/                  onnxruntime-web WASM files (regenerable; shipped pre-populated in the public download)
└── tests/                    vitest (core, widget via headless Chromium, learn, server, drift + leak guards)
```

## The pipeline, end to end

```
registry.json ──validate──▶ compile(seed) ──▶ train/val/test.jsonl + manifest.json
                                                   │
                              train.py (CPU, deterministic) ──▶ <slug>-<ver>.onnx + tokenizer.json + labels.json + ledger.json
                                                   │
                      ┌────────────────────────────┴──────────────────────────┐
              Node: core/infer/session.ts                         Browser: widget/cortex-widget.js
              (server POST /advisor)                              (fetch + sha256-verify the bundle + WASM)
                      └──────────── same ladder, same planner, same registry ─┘
```

`ledger.json` is the contract between training and both runtimes: it names
the three artifact files, pins their sha256, records what they were trained
from (registry hash, dataset hash, seed) and the measured metrics. Every
loader re-verifies the hashes before use; the eval gate recompiles the dataset
from the registry and refuses a ledger whose dataset hash no longer matches
(i.e. "the registry changed but nobody retrained").

## Decisions

| # | Decision | Why |
|---|---|---|
| D1 | **Registry as the single input.** Intents, keywords, templates, paraphrases, answers, status copy and links all live in one JSON file. | One thing for a host to author; the keyword rung, the dataset, the planner and the widget's guide list can never disagree with each other. |
| D2 | **Slots are dynamic.** Slot names and vocab come from `registry.slots`; nothing in code knows a slot name. | The original hard-coded six domain slots; a reusable package can't. Labels (`B-x`/`I-x`) derive from the manifest, so the trainer and both runtimes agree by construction. |
| D3 | **Status answers are host-provided.** A `status` intent's reply is whatever the host's `status(intentId, slots)` returns; otherwise the registry's `unavailable` copy. | The only way to print a live number without fabricating one. Cortex never computes a fact itself. |
| D4 | **Keyword rung first, model optional.** A registry alone gives a working assistant; training is an upgrade. | Adoption in minutes; Python/torch only when paraphrase robustness is worth it. |
| D5 | **Model as a reviewed build artifact**, committed with a sha256-pinned ledger; training is offline, CPU, seeded. | Reproducible, offline, no runtime Python, works on a static host. |
| D6 | **One artifact, two runtimes.** Same ONNX in Node and in the browser (WASM, `numThreads = 1`, no cross-origin isolation needed). | A static site gets the real model with zero backend; the only CSP change a host needs is `'wasm-unsafe-eval'`. |
| D7 | **"Learn the site" drafts a registry; a human edits it.** The bootstrapper harvests nav links and headings into howto intents. | Honest about what can be inferred from a page: destinations and vocabulary, not procedures or facts. The draft compiles and works on the keyword rung immediately. |
| D8 | **Built-in `greeting` / `out_of_domain` meta intents.** | Every model needs a null class; hosts shouldn't have to author 50 off-topic utterances. |
| D9 | **Held-out gate is optional but recommended.** `heldout[]` utterances are never compiled into a split; the trainer scores them and the ledger records it. | Guards against template memorisation; skipped honestly (`heldoutGate: "not declared"`) when a host provides none. |
| D10 | **No action/write layer, no external LLM tier, no audit chain in this package.** | Those were product-specific in the original (they wrote to its database and used its audit log). The engine exposes the evidence chain so a host can log it. |
| D11 | **Published under the proprietary licence, not open source.** | The rights decision belongs to the owner; the folder carries a NOTICE and no licence grant. |

## Three copies of the tokenizer, on purpose

TypeScript core, the Python trainer and the browser widget each carry the
same ~40-line tokenizer (lowercase → maximal runs of Unicode letters/numbers →
vocab lookup → pad/truncate). They can't import each other, so they are pinned
against one fixture (`tests/fixtures/tokenizer.parity.json`) from all three
sides, and a drift test parses the widget source to compare its constants with
the core's exports. Change one, change all three, run the tests.

## What it deliberately does not do

- Generate text. (Not a limitation — the reason it can't hallucinate.)
- Read your data. Live facts come only from the host's status provider.
- Take actions. Chips navigate; the host decides what a navigation means.
- Train at runtime. Training is a build step that produces a reviewable artifact.
