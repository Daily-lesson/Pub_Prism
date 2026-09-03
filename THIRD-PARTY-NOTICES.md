# Third-party notices

PRISM itself is proprietary — see [`LICENSE`](./LICENSE). The components listed
here are **not** covered by that licence. They are the property of their
respective owners and remain under their own terms, which are reproduced in
full below where redistribution requires it.

This file must be published alongside any build that redistributes the
components in Section 1. The `mirror-demo` workflow copies it into the public
demo mirror for exactly that reason.

---

## 1. Redistributed in the published build

These files are copied verbatim into the public demo mirror
(`Daily-lesson/Pub_Prism`) and into the server's `dist/`. Redistribution
obliges us to carry their licence text.

### ONNX Runtime Web — v1.20.1

Microsoft's WebAssembly inference runtime. PRISM Cortex's in-browser path loads
it to run the committed model when there is no backend session
(`v11 module: cortex`).

Files redistributed, from `prism-server/src/cortex/runtime/` — copied there by
`scripts/gen-cortex-runtime.js`, hashes pinned in `runtime-manifest.json`:

| File | Bytes |
|---|---|
| `ort.wasm.min.mjs` | 48,904 |
| `ort-wasm-simd-threaded.mjs` | 24,618 |
| `ort-wasm-simd-threaded.wasm` | 11,246,032 |

Published at `cortex/runtime/*` on the demo mirror.

Homepage: https://onnxruntime.ai/ · Source: https://github.com/microsoft/onnxruntime

> MIT License
>
> Copyright (c) Microsoft Corporation. All rights reserved.
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

**Note on the model itself.** `prism-cortex-*.onnx` is PRISM's own trained
artefact — original weights, trained on PRISM's own generated dataset. It is
covered by `LICENSE`, not by this section. ONNX Runtime is the engine that runs
it, nothing more.

---

## 2. Linked, not redistributed

Loaded by the browser from a third-party CDN. PRISM does not host or
redistribute these files; no licence text is required, and they are listed for
completeness and supply-chain visibility.

### Google Fonts — Inter, DM Mono, Syne

Requested from `fonts.googleapis.com` / `fonts.gstatic.com` by both
`PRISM-v10-complete.html` and `landing.html`, and allowed by the
`style-src` / `font-src` directives in each file's CSP.

| Family | Designer | Licence |
|---|---|---|
| Inter | Rasmus Andersson | SIL Open Font Licence 1.1 |
| DM Mono | Colophon Foundry for Google | SIL Open Font Licence 1.1 |
| Syne | Bonjour Monde / Lucas Descroix | SIL Open Font Licence 1.1 |

The OFL permits use, embedding, and redistribution; it forbids selling the
fonts on their own and requires that any *derivative* font be released under
the OFL under a different name. PRISM does neither, so no further obligation
attaches. Full text: https://openfontlicense.org/

**Privacy consequence, stated plainly:** because these are loaded from Google's
CDN, a visitor's IP address reaches Google on every page load of the public
demo. If that matters for a given deployment — GDPR-conscious EU hosting, an
air-gapped install — self-host the three families instead. That is a
`font-src 'self'` change plus the font files, not a redesign.

---

## 3. Server dependencies

The backend's npm dependencies (Express, Prisma, `mqtt`, `web-push`, `docx`,
`onnxruntime-node`, and the rest) are **not** redistributed to end users — they
run on PRISM's own servers and are never shipped to a browser. Their licences
apply to PRISM's own use of them, and the authoritative, always-current list is
the dependency tree itself:

```
cd PRISM-v10.5-complete/prism/prism-server
npx license-checker --summary        # or: npm ls --all
```

If a component ever moves from "runs on our server" to "shipped to the user's
browser or downloaded as an artefact," it belongs in Section 1 of this file and
its licence text must be reproduced here. The optional `go2rtc` camera sidecar
(`deploy/go2rtc/`, MIT) is deployed as an operator-run container from the
upstream image — PRISM redistributes no go2rtc code, only a compose service
definition and a starter config.
