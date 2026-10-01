# Cortex — the contracts

This file is the source of truth for every interface inside the package. Each
sub-system (core, trainer, widget, learn, server) is built against it, so a
change here is a change to all of them. Keep it exact.

Cortex is a **selector, not a generator**: a small learned model picks an intent
(and tags slot spans), and a deterministic planner composes the reply from
content the host wrote in its registry. No free text is ever generated, so
nothing can be hallucinated. Everything below serves that one idea.

Numbered sections are normative. "MUST" means a test pins it.

---

## 1. The registry (`*.registry.json`)

The registry is how a host describes its app. It is the only input to the
dataset compiler, the keyword rung, the planner, and the widget's guide list.

```jsonc
{
  "registryVersion": "1",
  "app": {
    "slug": "ops-dashboard",          // [a-z0-9-]+ — artifact filename stem
    "name": "Northwind Ops",          // shown in the widget header
    "assistantName": "Ops Assistant"  // optional, defaults to "<name> Assistant"
  },
  "slots": {                          // slot name -> vocab (may be {})
    "region": {
      "label": "Region",
      "vocab": [ { "id": "north", "label": "North campus" }, { "id": "south", "label": "South campus" } ]
    }
  },
  "intents": [
    {
      "id": "open_reports",           // ^[a-z][a-z0-9_]*$ — a training label; never renamed once shipped
      "family": "howto",              // "howto" | "status" | "meta"
      "label": "Find the reports",    // human label for the guide list and eval reports
      "slots": ["region"],            // slot names this intent's utterances may carry ([] = none)
      "keywords": "reports report analytics numbers dashboard chart", // space-separated keyword bag
      "templates": [                  // grammar templates; {slot} placeholders MUST name a slot in "slots"
        "how do i find the reports",
        "show me the {region} reports"
      ],
      "paraphrases": ["where are my numbers"],   // hand-authored utterances, intent only (slots auto-tagged)
      "answer": {                     // REQUIRED for family "howto"; ignored for "status"/"meta"
        "steps": ["Open Reports from the left nav.", "Pick a region in the filter bar."],
        "links": [ { "label": "Reports", "target": { "route": "/reports" }, "need": "reports" } ]
      },
      "status": {                     // REQUIRED for family "status"
        "unavailable": "I can't read report counts here — open Reports for the live view.",
        "links": [ { "label": "Reports", "target": { "route": "/reports" } } ]
      }
    }
  ],
  "heldout": [                        // OPTIONAL generalization gate set: never compiled into a split
    { "utterance": "where would i look at last month's figures", "intent": "open_reports" }
  ]
}
```

Rules:

1.1 `intents[].id` are unique. `family` is one of `howto|status|meta`.
1.2 Every `{placeholder}` in a template MUST be listed in that intent's `slots`,
    and every slot name MUST exist in `slots`. A slot with an empty vocab may
    not be used in a template (nothing to fill it with).
1.3 A `howto` intent MUST have `answer.steps` (≥1). A `status` intent MUST have
    `status.unavailable`. `meta` intents need neither.
1.4 The two built-in meta intents `greeting` and `out_of_domain` are ALWAYS
    part of the taxonomy. A host may declare them to override the built-in
    English utterance sets (see `core/registry/builtins.ts`); if it doesn't,
    the compiler adds them with the built-in templates/paraphrases. Their
    replies come from `app`-neutral fixed copy in the planner.
1.5 `link.target` is opaque JSON handed to the host's `navigate(target)`.
    `link.need` is an opaque string handed to the host's `isAllowed(need)`;
    a link whose `need` is refused is DROPPED (never rendered disabled).
1.6 Taxonomy order = declaration order in `intents[]`, with `greeting` and
    `out_of_domain` appended last when not declared. This order is the label
    order everywhere (labels.json, confusion matrices, the manifest).
1.7 `registryHash` = sha256 of `canonicalJson(registry)` where `canonicalJson`
    is recursive key-sorted `JSON.stringify` (arrays keep order). Computed
    over the registry AFTER built-in meta intents were added, so two hosts
    with identical content hash identically regardless of whether they
    declared the built-ins.
1.8 The JSON Schema in `registry/schema.json` is the machine check;
    `core/registry/validate.ts` implements the same rules plus 1.2–1.3 (which
    JSON Schema can't express) and returns `{ ok, errors[] }` with one
    human-readable line per error, naming the intent/slot.

## 2. The dataset compiler (`core/dataset/`)

Deterministic: same registry + same seed ⇒ byte-identical output.

2.1 For each intent: examples = `generateFromTemplates(intent, seed)` ∪
    `paraphrases` (with `autoDetectSlots`). Template expansion: an unslotted
    template yields 1 example; a slotted template yields
    `VARIANTS_PER_SLOTTED_TEMPLATE = 4` examples, each slot filled from its
    own seeded, shuffled queue of vocab **labels** (the label is the literal
    text inserted; the id is what a tagged span resolves to). Slot spans are
    `[start, end)` character offsets into the final utterance and MUST satisfy
    `utterance.slice(start, end) === value`.
2.2 Augmentation: each generated example yields up to
    `AUGMENT_VARIANTS_PER_EXAMPLE = 20` deterministic variants (prefix/suffix
    fillers, casing, light synonym swaps) with spans recomputed on the final
    string. Fillers are generic English (please, can you, quickly, …) and
    contain no domain words.
2.3 `autoDetectSlots(utterance, intent)`: longest vocab label first, exact
    case-insensitive match that is a WHOLE WORD (the characters before and
    after the match are not `\p{L}\p{N}` — a label `IT` never tags
    `kitchen`), first such occurrence, one match per slot category, no
    overlapping spans. The widget's `detectSlots` is the same rule.
2.4 Global cross-intent dedupe on the exact utterance string (first intent
    wins in taxonomy order).
2.5 Split per intent: shuffle with `mulberry32(deriveSeed(seed, "split::" + id))`,
    then `val = max(1, round(n*0.15))`, `test = max(1, round(n*0.15))`,
    `train = n - val - test`. Every intent appears in every split. Then each
    global split is shuffled with `deriveSeed(seed, "global::train"|"global::val"|"global::test")`.
2.6 Held-out guard: after splitting, ABORT (throw) if any compiled utterance
    case-insensitively equals a `heldout[]` utterance.
2.7 Output directory contains exactly: `train.jsonl`, `val.jsonl`,
    `test.jsonl`, `manifest.json`. One example per JSONL line:
    `{"utterance":"…","intent":"…","slots":[{"name":"region","value":"North campus","start":12,"end":24}]}`
2.8 `manifest.json` (`app` carries the registry's slug/name so the trainer
    needs no separate `--slug`):
    ```json
    { "app": { "slug": "my-app", "name": "My App" },
      "seed": 42, "registryHash": "…", "datasetHash": "…",
      "intents": ["open_reports", "…", "greeting", "out_of_domain"],
      "slotNames": ["region"],
      "slotLabels": ["O", "B-region", "I-region"],
      "counts": { "train": 0, "val": 0, "test": 0, "total": 0, "perIntent": { "open_reports": 0 } } }
    ```
    `intents` and `slotNames` are EXPLICIT arrays (never recovered from key
    order). `slotLabels = ["O"] + flatMap(slotNames, s => ["B-"+s, "I-"+s])`.
2.9 `datasetHash` = sha256 of `canonicalJson({ seed, train, val, test })` over
    the example objects in output order.
2.10 PRNG: mulberry32. `deriveSeed(seed, label)` starts from
    `(seed >>> 0) ^ 0x811c9dc5` and folds each char code of `label` in with the
    FNV-1a step (`h ^= code; h = imul(h, 0x01000193)`), returning `h >>> 0`
    (`core/dataset/prng.ts` is the reference; the trainer never re-derives it).
2.11 `out_of_domain` has an EMPTY keyword bag by construction — its label words
    must never make it a keyword-rung target (§5.1 routes it only via the model).
    `greeting` keeps its bag.

## 3. The tokenizer — the byte-identical contract

Three copies exist on purpose (TypeScript core, Python trainer, browser
widget) because each runtime can't import the others. Tests pin them
against a shared fixture (`tests/fixtures/tokenizer.parity.json`).

3.1 Lowercase (if `lower`), then split into maximal runs of Unicode Letter or
    Number codepoints (`/[\p{L}\p{N}]/u` in JS; `unicodedata.category(ch)[0] in ("L","N")`
    in Python). Everything else is a separator and is discarded.
3.2 Map each word to `vocab[word]` else `unkId`. Truncate to `maxLen`, pad
    with `padId`.
3.3 `tokenizer.json`: `{ "version": "…", "lower": true, "maxLen": N, "padId": 0,
    "unkId": 1, "padToken": "<pad>", "unkToken": "<unk>", "vocab": { "word": id } }`.
    Vocab ids start at 2, assigned by (descending frequency, then alphabetical)
    over the TRAIN split only.
3.4 `maxLen = min(32, max(4, longestTokenCount + 4))` computed over
    train+val+test.
3.5 Offsets for slot decoding come from `wordSplitWithOffsets` over the
    ORIGINAL-cased utterance; casing never moves boundaries.

## 4. The model contract (ONNX)

4.1 Input `input_ids`: int64, shape `[1, maxLen]`.
4.2 Outputs `intent_logits` (float32, `[1, numIntents]`) and `slot_logits`
    (float32, `[1, maxLen, numSlotLabels]`, row-major when flattened).
4.3 Decode: softmax+argmax over intents → `(intent, intentConf)`; per-token
    argmax over the first `min(realTokens, maxLen)` positions → BIO labels →
    spans (`B-x` or a name change starts a span; `I-x` extends; `O` closes).
    A span's `value` is the raw text; `resolvedId` is the vocab id whose label
    equals the value case-insensitively, else absent.
4.4 Architecture (trainer): `Embedding(vocab, E, padding_idx=0)` →
    bidirectional `GRU(E, H)` → intent head = masked mean-pool over GRU
    output → `Linear(2H,H)` → ReLU → `Linear(H, numIntents)`; slot head =
    `Linear(2H, numSlotLabels)` per token. Loss = CE(intent, label_smoothing
    0.1) + `slotLossWeight` × CE(slots, ignore_index −100 on PAD). Adam.
    Deterministic: seeds `random`/`numpy`/`torch`, `use_deterministic_algorithms(True)`,
    `set_num_threads(1)`. Export opset 17, `dynamo=False`, batch axis dynamic
    only. Optional INT8 dynamic quantization kept only if accuracy drop ≤ 0.01
    AND slot-F1 drop ≤ 0.01 AND ≥15% smaller.
4.5 Artifacts in `--out-dir`: `<slug>-<version>.onnx`, `<slug>-<version>.tokenizer.json`,
    `<slug>-<version>.labels.json` (`{version, intents[], slots[]}` — same
    order as the manifest), and `ledger.json`:
    ```json
    { "version": "0.1.0", "app": "ops-dashboard",
      "onnx": { "file": "…", "sha256": "…", "bytes": 0, "quantized": false },
      "tokenizer": { "file": "…", "sha256": "…" },
      "labels": { "file": "…", "sha256": "…" },
      "trainedFrom": { "registryHash": "…", "datasetHash": "…", "seed": 42, "split": { "train": 0, "val": 0, "test": 0 } },
      "modelConfig": { "vocabSize": 0, "maxLen": 0, "embedDim": 0, "gruHidden": 0, "numIntents": 0, "numSlotLabels": 0, "paramCount": 0, "epochs": 0 },
      "metrics": { "inDistribution": { "intentAccuracy": 0, "macroF1": 0, "perIntentF1": {}, "slotF1": 0, "slotPrecision": 0, "slotRecall": 0, "perSlotF1": {} },
                   "heldout": { "intentAccuracy": 0, "n": 0, "perIntentAccuracy": {}, "confusionTopMisses": [] } },
      "acceptanceFloor": { "inDistributionIntentAccuracy": 0.9, "inDistributionSlotF1": 0.9, "heldoutIntentAccuracy": 0.85 } }
    ```
    `metrics.heldout` is `null` when the registry declares no `heldout[]`
    (then the heldout floor is not enforced — the ledger says so with
    `"heldoutGate": "not declared"`).
4.6 Every loader (Node `core/infer/session.ts`, the widget, the eval gate)
    MUST re-verify each artifact's sha256 against `ledger.json` before use and
    refuse a mismatch. The widget's check is best-effort (skipped, not failed,
    without `crypto.subtle`).

## 5. The answer ladder

5.1 Rungs: `cortex` (model ≥ threshold, default 0.6) → `keyword` (whole-word
    overlap against each intent's keyword bag; highest score wins; ties break
    on taxonomy order; a score of 0 is no match) → `fallback` (honest "I don't
    know" + the guide list). `out_of_domain` above threshold is answered as
    the honest out-of-domain copy on the `cortex` rung.
5.2 Keyword tokens: lowercase, split on `/[^a-z0-9]+/`, keep tokens with
    length ≥ 3 not in STOPWORDS; the query's tokens are a SET (a repeated word
    scores once) — identical in the core and the widget. STOPWORDS (exact, shared by all copies):
    `the a an and or of to in on for with is are was were be been do does did how what where when which who whom why can could would should will shall may might must i me my we our you your it its this that these those there here from by at as into onto than then so if not no yes please just about over under up down out off again more most some any all`
5.3 Keyword bag for an intent = its `keywords` string ∪ its `label` words,
    tokenized with the same rule.
5.4 The planner returns `{ answerText, chips[], evidence }` where `chips` are
    the intent's links after the `need` gate, and
    `evidence = { modelVersion, modelSha256, intent, intentConf, ladderRung, plannerTemplateId, registryEntriesUsed[] }`.
    `plannerTemplateId` is `planner.howto.<id>` | `planner.status.<id>` |
    `planner.meta.greeting` | `planner.meta.out_of_domain` | `planner.fallback`.
5.5 Howto answer text = the steps numbered `1. … 2. …` joined by a space,
    prefixed with `Noted — you mentioned "<a>", "<b>". ` (every tagged span,
    in utterance order) when any slot span was tagged. On the `keyword` rung
    slots come from §2.3's whole-word vocab match against the query (core and
    widget alike); on the `cortex` rung from the model's BIO spans. Status answer text = `host.status(intentId, slots)` if the host
    supplies a status provider and it returns a string; else the registry's
    `status.unavailable` copy — never a fabricated number.

## 6. The browser widget (`widget/cortex-widget.js`)

One framework-free classic script. Loading it defines `window.Cortex`. It
touches no host global except what the host passes in.

```js
Cortex.mount({
  registry: {...} | registryUrl: 'cortex/registry.json',   // one of the two
  modelBase: 'cortex/model/',        // dir with ledger.json; OMIT for keyword-only mode
  runtimeBase: 'cortex/runtime/',    // dir with ort.wasm.min.mjs + the two wasm-glue files
  serverUrl: '/api/advisor',         // optional; used only while host.isLive() is true
  confidenceGate: 0.6,
  host: {
    navigate(target) {},             // required for chips to do anything
    isAllowed(need) { return true },  // optional; absent = allow everything (fail-open, UX-only)
    status(intentId, slots) {},      // optional; returns string | Promise<string> | undefined
    isLive() { return false },       // optional; when true and serverUrl is set, ask the server
    headers() { return {} }          // optional; extra headers for the server call
  },
  ui: { launcher: true, hotkey: null, title: null, theme: 'auto' }  // all optional; hotkey e.g. 'k' for ⌘K/Ctrl-K
});
Cortex.open(query?); Cortex.close();
Cortex.ask(query) -> Promise<{ answerText, chips, evidence }>;   // no UI
Cortex.classify(query) -> Promise<{ intent, intentConf, slots }> // model only; rejects in keyword-only mode
```

6.1 Paths are resolved relative to `document.baseURI` so the same code works
    at a site root and under a sub-path. The runtime is loaded with dynamic
    `import()` of `<runtimeBase>ort.wasm.min.mjs`; `ort.env.wasm.numThreads = 1`,
    `wasmPaths = <runtimeBase>`, execution provider `['wasm']`.
6.2 All ids/classes use the `cortex-` prefix; all CSS is scoped under
    `#cortex-root`. Every tenant-derived string reaches the DOM through
    `textContent` or one escaper (`& < > "`). Chip clicks call
    `host.navigate(target)` after closing the drawer.
6.3 Boot order: registry → keyword rung is immediately usable → the model
    loads lazily on the first ask (or `Cortex.warm()`); a model load failure
    degrades to keyword-only and the provenance line says so.
6.4 The provenance line under every answer states the rung and, for the
    model rung, the model version + confidence. Never a fabricated
    "answered by AI" claim on the keyword/fallback rungs.
6.5 CSP needed by a host: `script-src` must include `'wasm-unsafe-eval'`
    (and `'self'` for the same-origin `import()`); `connect-src` must cover
    wherever the model/runtime/server live.
6.6 The widget's tokenizer, softmax, decoder, keyword matcher and STOPWORDS
    are copies of the core's. `tests/widgetDrift.test.ts` parses the
    widget file and asserts the STOPWORDS list and the word-char regex are
    byte-identical to the core's exports.

## 7. The server (`server/`)

7.1 `createAdvisorRouter(opts)` returns an Express router:
    `POST /` body `{ query: string (≤ 2000) }` → `{ answer, chips, evidence, intent, intentConf, ladderRung, modelVersion }`;
    `GET /model/:file` and `GET /runtime/:file` serve ONLY filenames listed in
    `ledger.json` / `runtime-manifest.json` (allowlist), `Cache-Control: public, max-age=31536000, immutable`.
    `opts = { registry, artifactsDir, runtimeDir?, status?: (ctx, intentId, slots) => string | Promise<string | undefined>, context?: (req) => unknown, confidenceGate? }`.
7.2 The router never reads a body field it doesn't declare, never logs the
    raw query, and never calls out to any network. Rate limiting and auth are
    the host's job and are documented as such.
7.3 `server/standalone.ts` serves `examples/demo-dashboard/` + the router at
    `/api/advisor` for a local end-to-end run.

## 8. `learn` — drafting a registry from a site

8.1 `cortex learn <url | file.html | sitemap.xml> --out <registry.json> [--crawl --depth 1] [--name …] [--slug …]`
8.2 Extracts the page title, every `<nav>`/`<header>`/`<aside>` link and, when
    crawling, each same-origin page's `<title>`/`<h1>`/`<h2>` text. Each unique
    destination (by href) becomes one `howto` intent: `id` = slug of the link
    text (`^[a-z][a-z0-9_]*$`, deduped with `_2`, `_3`; the built-in ids
    `greeting`/`out_of_domain` are reserved and dedupe the same way), `label` = the link
    text, `keywords` = link text + page heading words (deduped, stopwords
    removed), `templates` = the built-in destination phrasings applied to the
    label (`core/learn/phrasings.ts`: "how do i get to {x}", "where is {x}",
    "open {x}", "take me to {x}", "show me {x}", "i want to see {x}",
    "navigate to {x}", "find {x}"), `answer.steps = ["Open <label> from the site navigation."]`,
    `answer.links = [{ label, target: { href } }]`.
8.3 The output is a DRAFT: it validates and compiles as-is, and the README
    says a human should edit the labels/keywords/steps before training.
    Destinations whose link text is the same (case-insensitively) are merged
    into ONE intent with several `answer.links` and a warning — identical
    templates under different ids can never be told apart by a model.
8.4 Fetching is a dev-time CLI concern (no SSRF guard); the doc says so. A
    `file:` root follows only links that resolve under the root document's
    own directory.

## 9. Honesty rules (apply everywhere)

- No number is ever printed that didn't come from the host's status provider.
- A missing model is not an error; it is keyword-only mode, and the UI says so.
- A registry with a problem fails loudly at validate/compile time, never
  silently at answer time.
