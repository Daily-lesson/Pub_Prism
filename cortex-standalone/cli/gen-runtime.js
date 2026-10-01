// cli/gen-runtime.js
//
// Dev-only generator for the browser runtime bundle. Re-run only when bumping the
// onnxruntime-web devDependency. Copies the three files the WASM-only CPU backend needs out
// of node_modules/onnxruntime-web/dist into runtime/, and writes runtime/runtime-manifest.json
// (sha256-pinned; the server's /runtime/:file allowlist reads it — the widget does NOT verify
// the runtime files, only the model bundle against ledger.json, see CONTRACTS §4.6).
//
//   - ort.wasm.min.mjs            the WASM-only backend entry point
//   - ort-wasm-simd-threaded.mjs  the glue module the entry point imports
//   - ort-wasm-simd-threaded.wasm the wasm binary (the loader sets numThreads = 1, so no
//                                 cross-origin-isolation headers are needed)
//
// Run: node cli/gen-runtime.js   (from the package root)
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const ROOT = path.join(__dirname, '..');
const SRC_DIR = path.join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
const DEST_DIR = path.join(ROOT, 'runtime');

const FILES = {
  entry: 'ort.wasm.min.mjs',
  loader: 'ort-wasm-simd-threaded.mjs',
  wasm: 'ort-wasm-simd-threaded.wasm',
};

function sha256File(p) {
  return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
}

function main() {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8'));
  fs.mkdirSync(DEST_DIR, { recursive: true });

  const manifest = {
    version: pkg.version,
    source: 'onnxruntime-web',
    license: pkg.license,
    backend: 'wasm',
    numThreads: 1,
    generatedBy: 'cli/gen-runtime.js',
  };

  for (const [key, filename] of Object.entries(FILES)) {
    const srcPath = path.join(SRC_DIR, filename);
    if (!fs.existsSync(srcPath)) {
      console.error(`[gen-runtime] missing source file: ${srcPath}`);
      process.exit(1);
    }
    const destPath = path.join(DEST_DIR, filename);
    fs.copyFileSync(srcPath, destPath);
    manifest[key] = { file: filename, sha256: sha256File(destPath), bytes: fs.statSync(destPath).size };
    console.log(`[gen-runtime] copied ${filename} (${manifest[key].bytes} bytes)`);
  }

  const manifestPath = path.join(DEST_DIR, 'runtime-manifest.json');
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  console.log(`[gen-runtime] wrote ${path.relative(ROOT, manifestPath)}`);
}

main();
