'use strict';
/* tests/support/static-server.js — a tiny static file server for browser tests.
 *
 * Serves `root` on a random port with correct MIME types for the files the widget fetches
 * (.mjs / .wasm / .json / .onnx). Virtual mounts let a test map a URL prefix onto another
 * directory (e.g. a temp dir holding a generated model bundle) or onto `null`, which makes
 * that prefix answer 404 regardless of what is on disk.
 *
 *   const { start } = require('./static-server');
 *   const srv = await start({ root: '/path/to/pkg', mounts: { '/runtime/': '/path/to/dist/' } });
 *   srv.mount('/examples/demo-dashboard/cortex/model/', '/tmp/bundle');   // add / replace
 *   srv.mount('/examples/demo-dashboard/cortex/model/', null);            // force 404
 *   srv.unmount('/runtime/');
 *   await srv.stop();
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
};

function safeJoin(root, rel) {
  const target = path.resolve(root, '.' + rel);
  const base = path.resolve(root);
  if (target !== base && !target.startsWith(base + path.sep)) return null;
  return target;
}

function start(opts) {
  const root = path.resolve(opts && opts.root ? opts.root : process.cwd());
  const mounts = new Map();
  if (opts && opts.mounts) for (const k of Object.keys(opts.mounts)) mounts.set(k, opts.mounts[k]);
  const requests = [];

  const server = http.createServer((req, res) => {
    let urlPath;
    try { urlPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname); }
    catch (e) { res.writeHead(400); res.end('bad request'); return; }
    requests.push({ method: req.method, path: urlPath });
    if (req.method !== 'GET' && req.method !== 'HEAD') { res.writeHead(405); res.end('method not allowed'); return; }

    // longest mount prefix wins
    let mountPrefix = null;
    for (const p of mounts.keys()) {
      if (urlPath.startsWith(p) && (mountPrefix === null || p.length > mountPrefix.length)) mountPrefix = p;
    }
    let file;
    if (mountPrefix !== null) {
      const dir = mounts.get(mountPrefix);
      if (dir === null || dir === undefined) { res.writeHead(404); res.end('not found'); return; }
      file = safeJoin(dir, '/' + urlPath.slice(mountPrefix.length));
    } else {
      file = safeJoin(root, urlPath);
    }
    if (!file) { res.writeHead(403); res.end('forbidden'); return; }
    fs.stat(file, (err, st) => {
      if (!err && st.isDirectory()) { file = path.join(file, 'index.html'); return serveFile(file, req, res); }
      serveFile(file, req, res);
    });
  });

  function serveFile(file, req, res) {
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end('not found'); return; }
      const ext = path.extname(file).toLowerCase();
      res.writeHead(200, {
        'Content-Type': MIME[ext] || 'application/octet-stream',
        'Content-Length': data.length,
        'Cache-Control': 'no-store',
      });
      if (req.method === 'HEAD') { res.end(); return; }
      res.end(data);
    });
  }

  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(opts && opts.port ? opts.port : 0, '127.0.0.1', () => {
      const port = server.address().port;
      resolve({
        port,
        url: 'http://127.0.0.1:' + port,
        origin: 'http://127.0.0.1:' + port,
        requests,
        mount(prefix, dir) { mounts.set(prefix, dir); },
        unmount(prefix) { mounts.delete(prefix); },
        stop() { return new Promise((r) => server.close(() => r())); },
      });
    });
  });
}

module.exports = { start, MIME };
