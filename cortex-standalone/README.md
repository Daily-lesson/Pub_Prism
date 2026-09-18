# Cortex standalone

A compiled assistant for any dashboard. Describe your app once in a JSON
registry — its destinations, how-to steps, and the live questions it can
answer — and Cortex becomes the "find it fast" assistant for it: a small
intent classifier (trained from your registry, run in the browser or in Node)
plus a deterministic answer planner that only ever says what you wrote.

It is a **selector, not a generator**. It never produces free text, so it
cannot make up a step, a link, or a number. What it can do is understand
"where would I look at last month's figures" and take the user to Reports.

Start with the live example: `examples/demo-dashboard/` (also published at
`…/cortex-standalone/examples/demo-dashboard/` on the public mirror). Then
read `docs/ARCHITECTURE.md` for the design and `docs/CONTRACTS.md` for the
exact interfaces.

## What you get

| Piece | What it does | Needs |
|---|---|---|
| `widget/cortex-widget.js` | One framework-free script: a drawer with search, answers, navigation chips and a guide list. Works keyword-only with just a registry; runs the trained model in-browser (WASM) when one is present. | a static host |
| `core/` + `cli/cortex.ts` | Registry validation, dataset compiler, keyword rung, ONNX session, planner, engine, eval gate. | Node ≥ 20 |
| `train/` | Trains the model from the compiled dataset and writes a sha256-pinned artifact bundle. | Python 3.11 + CPU torch (dev-time only) |
| `learn/` | Drafts a registry from a live site, a saved page, or a sitemap — one intent per destination. | Node |
| `server/` | An Express router (`POST /advisor` + allowlisted model/runtime routes) for hosts that want server-side answers with live data. | Node |

## Three rungs, in order of cost

1. **Keyword** — whole-word overlap against each intent's keyword bag.
   Available the moment the registry validates. No model, no network.
2. **Model** — a bidirectional-GRU intent + slot classifier trained on a
   dataset generated from the registry. Handles paraphrase the keyword rung
   can't. One ONNX file, run by `onnxruntime-node` or `onnxruntime-web`.
3. **Fallback** — an honest "I'm not sure" plus the guide list. A miss is
   never a wrong answer.

Every reply carries a provenance line saying which rung answered.

## Quick start (five minutes, no Python)

```bash
cd cortex-standalone
npm ci
npm run gen-runtime                 # copies the onnxruntime-web WASM files into runtime/
                                    # (the public download ships them pre-populated)
npx tsx cli/cortex.ts validate registry/examples/ops-dashboard.registry.json
npm run serve                       # http://localhost:8787 → the example dashboard
```

Open the dashboard, click the assistant, and ask "how do i raise a ticket",
"how many tickets are open", or "where would i look at last month's figures".

## Adopt it in your own dashboard

1. **Write a registry.** Copy `registry/examples/ops-dashboard.registry.json`
   and replace the content, or draft one from your site:

   ```bash
   npx tsx cli/cortex.ts learn https://your-dashboard.example/ --out my-app.registry.json --crawl --depth 1
   ```

   `learn` turns every navigation link into a how-to intent (label, keywords
   from the page's headings, a step, a link). It is a draft — edit the steps
   and keywords, add `status` intents for the live questions your app can
   answer, and add a few `heldout` utterances a colleague wrote without
   seeing the templates. Then `validate` it.

2. **Mount the widget.** Serve `widget/cortex-widget.js` and your registry as
   static files and call:

   ```html
   <script src="cortex/cortex-widget.js"></script>
   <script>
     Cortex.mount({
       registryUrl: 'cortex/my-app.registry.json',
       modelBase:   'cortex/model/',      // omit until you have trained one
       runtimeBase: 'cortex/runtime/',
       host: {
         navigate(target) { location.hash = target.route; },           // target is YOUR json
         isAllowed(need)  { return can(need); },                       // optional gate; drops chips
         status(intentId, slots) { return liveSentenceFor(intentId); } // optional; live numbers come only from here
       }
     });
   </script>
   ```

   Your Content-Security-Policy needs `'wasm-unsafe-eval'` in `script-src`
   for the in-browser model. Nothing else changes.

3. **Train the model** when you want paraphrase robustness:

   ```bash
   npx tsx cli/cortex.ts compile my-app.registry.json --out build/dataset --seed 42
   python3 -m venv train/.venv && . train/.venv/bin/activate
   pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 && pip install -r train/requirements.txt
   python train/train.py --data-dir build/dataset --registry my-app.registry.json --out-dir cortex/model
   npx tsx cli/cortex.ts eval-gate my-app.registry.json --artifacts cortex/model
   ```

   Training runs on one CPU core, deterministically (same registry + seed ⇒
   same weights on the same machine), and writes `ledger.json` with the
   sha256 of every artifact and the metrics it measured. Commit the bundle
   like any other build artifact. `eval-gate` recompiles the dataset from
   the registry and refuses a ledger that no longer matches it — so a
   registry edit can't silently ship an out-of-date model. `ci/eval-gate.yml`
   is a GitHub Actions template for exactly that check.

4. **Optional: a server.** If you want answers to include live facts your
   backend knows:

   ```ts
   import { createAdvisorRouter } from './cortex-standalone/server/router';
   app.use('/api/advisor', createAdvisorRouter({
     registry, artifactsDir: 'cortex/model', runtimeDir: 'cortex-standalone/runtime',
     context: (req) => req.user,
     status: async (ctx, intentId, slots) => yourFacts(ctx, intentId, slots),  // string | undefined
   }));
   ```

   and set `serverUrl: '/api/advisor'` + `host.isLive()` in the widget. Auth
   and rate limiting are your middleware's job; the router never logs the
   query and never makes an outbound request.

## The registry, in one look

```jsonc
{
  "registryVersion": "1",
  "app":   { "slug": "my-app", "name": "My App" },
  "slots": { "region": { "vocab": [ { "id": "north", "label": "North campus" } ] } },
  "intents": [
    { "id": "find_reports", "family": "howto", "label": "Find the reports",
      "slots": ["region"], "keywords": "reports analytics numbers chart",
      "templates": ["how do i find the reports", "show me the {region} reports"],
      "paraphrases": ["where are my numbers"],
      "answer": { "steps": ["Open Reports from the left nav."], "links": [ { "label": "Reports", "target": { "route": "/reports" } } ] } },
    { "id": "open_tickets_status", "family": "status", "label": "How many tickets are open",
      "keywords": "open tickets count backlog",
      "templates": ["how many tickets are open"],
      "status": { "unavailable": "I can't read the queue here — open Tickets for the live count." } }
  ],
  "heldout": [ { "utterance": "where would i look at last month's figures", "intent": "find_reports" } ]
}
```

`greeting` and `out_of_domain` are built in. `docs/CONTRACTS.md` §1 has every
field; `registry/schema.json` is the machine check.

## Honesty rules the code enforces

- A number is printed only if your `status` provider returned it.
- No model is not an error: the widget runs keyword-only and its provenance
  line says so.
- A registry problem fails at `validate`/`compile`, never silently at answer
  time.
- Every model artifact's (onnx / tokenizer / labels) sha256 is re-verified against
  `ledger.json` before use — in Node and in the browser. The WASM runtime files are
  served same-origin and are not hash-checked by the widget.
- A `need` the host refuses drops the chip; it is never rendered disabled.

## Limits, stated plainly

- Cortex answers questions about *how to use* your app and *what it says*
  (via your status provider). It does not read your data, take actions, or
  generate prose.
- The model is small on purpose (a few hundred KB) and English-oriented;
  the tokenizer is Unicode-aware, the paraphrase augmentation is English.
- Training needs Python and torch on a dev machine or CI runner — never at
  runtime.
- `learn` fetches whatever URL you give it. It is a developer tool with no
  request guard; do not expose it to untrusted input. For a saved `.html` root
  it follows only links under that file's own directory.

## Commands

```
npm run typecheck        tsc over core/cli/learn/server/tests
npm test                 vitest: core, learn, server, widget (headless Chromium), drift + leak guards
npm run validate <reg>   npm run compile <reg> -- --out <dir>   npm run eval-gate <reg> -- --artifacts <dir>
npm run learn <input> -- --out <reg>          npm run gen-runtime          npm run serve
python -m unittest discover -s train -p 'test_*.py'
```

## Rights

Proprietary — see `NOTICE.md`. Third-party components are credited in
`THIRD-PARTY-NOTICES.md`. The example dashboard is fictional and every
number it shows is mock data generated in the page.
