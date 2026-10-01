# Third-party notices — Cortex standalone

Cortex itself is proprietary — see `NOTICE.md`. The component below is not
covered by that notice; it belongs to its owner and stays under its own terms,
reproduced here because redistributing its files requires it.

## ONNX Runtime Web — v1.20.1 (MIT)

Microsoft's WebAssembly inference runtime. The browser widget
(`widget/cortex-widget.js`) loads it to run a trained model with no backend.

Files redistributed under `runtime/` (regenerated from
`node_modules/onnxruntime-web/dist` by `npm run gen-runtime`, sha256s pinned in
`runtime/runtime-manifest.json`): `ort.wasm.min.mjs`,
`ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm`.

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

The server path uses `onnxruntime-node` (also MIT) as an ordinary npm
dependency; it is installed, not redistributed, so no text is reproduced for it.
