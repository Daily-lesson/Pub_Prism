/**
 * PRISM v10.2 — Service Worker
 * Cache-first strategy for app shell · All JS inlined in HTML
 *
 * Strategy:
 *   App shell (HTML, manifest, icons, fonts) → cache-first, then network fallback
 *   API calls / live telemetry              → network-only (never cached)
 *   Offline mutations                        → queued in IndexedDB by the app
 */

const CACHE_VERSION   = 'prism-v11-v3'; // PRISM Cortex (docs/CORTEX-DESIGN.md §4 P3): precache the model+runtime bundle
const SHELL_CACHE     = `${CACHE_VERSION}-shell`;
const DATA_CACHE      = `${CACHE_VERSION}-data`;

// Files that make up the app shell (all JS is now inlined in the HTML)
const SHELL_ASSETS = [
  './',
  './PRISM-v10-complete.html',
  './manifest.json',
  './icon-192.png',
  './icon-512.png',
  // Google Fonts — cached so app works offline with correct typography
  'https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300;1,400&family=Syne:wght@400;600;700;800&display=swap',
];

// PRISM Cortex (v11 module: cortex, docs/CORTEX-DESIGN.md §4 P3) — the model bundle
// (src/routes/frontend.ts's `cortex/model/*`) and onnxruntime-web WASM runtime
// (`cortex/runtime/*`) so demo/offline mode gets the REAL in-browser model, not just the
// keyword-rules fallback. Filenames are version-pinned by ledger.json / runtime-manifest.json
// (docs/CORTEX-DESIGN.md D2) — hardcoding them here would silently go stale on a model/runtime
// bump, so fetch the two small manifests at install time and precache whatever they currently
// reference instead. Best-effort: a manifest fetch failure (offline first install, or the
// static Pages mirror not having copied them yet) just means Cortex caches lazily on first use
// via the generic cache-first fetch handler below, same as any other same-origin resource —
// never blocks the rest of the app shell from installing.
async function cortexAssetUrls() {
  const urls = ['./cortex/model/ledger.json', './cortex/runtime/runtime-manifest.json'];
  try {
    const ledger = await (await fetch('./cortex/model/ledger.json')).json();
    urls.push(
      `./cortex/model/${ledger.onnx.file}`,
      `./cortex/model/${ledger.tokenizer.file}`,
      `./cortex/model/${ledger.labels.file}`,
    );
  } catch (err) {
    console.warn('[PRISM SW] Cortex model manifest fetch failed (will cache on first use):', err.message);
  }
  try {
    const runtime = await (await fetch('./cortex/runtime/runtime-manifest.json')).json();
    urls.push(
      `./cortex/runtime/${runtime.entry.file}`,
      `./cortex/runtime/${runtime.loader.file}`,
      `./cortex/runtime/${runtime.wasm.file}`,
    );
  } catch (err) {
    console.warn('[PRISM SW] Cortex runtime manifest fetch failed (will cache on first use):', err.message);
  }
  return urls;
}

// Routes that should NEVER be served from cache (live robot data, auth)
const NETWORK_ONLY_PATTERNS = [
  /\/v1\/robots/,
  /\/v1\/missions/,
  /\/api\//,
  /oauth/,
  /token/,
  /websocket/,
  /ws:\/\//,
  /wss:\/\//,
];

// ── Install ──────────────────────────────────────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then(async cache => {
        const cortexUrls = await cortexAssetUrls();
        return Promise.allSettled(
          [...SHELL_ASSETS, ...cortexUrls].map(url =>
            cache.add(url).catch(err =>
              console.warn('[PRISM SW] Pre-cache miss:', url, err.message)
            )
          )
        );
      })
      .then(() => self.skipWaiting())
  );
});

// ── Activate ─────────────────────────────────────────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(k => k !== SHELL_CACHE && k !== DATA_CACHE)
          .map(k => {
            console.log('[PRISM SW] Deleting old cache:', k);
            return caches.delete(k);
          })
      )
    ).then(() => self.clients.claim())
  );
});

// ── Fetch ─────────────────────────────────────────────────────────────────────
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  if (request.method !== 'GET') return;
  if (NETWORK_ONLY_PATTERNS.some(p => p.test(request.url))) return;
  if (!url.protocol.startsWith('http')) return;

  // App-shell navigations are network-first: cache-first would serve a stale
  // shell forever once cached (the server injects per-tenant config and ships
  // updates in the same file). Cache fallback keeps offline working.
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstShell(request));
    return;
  }

  event.respondWith(cacheFirst(request));
});

async function networkFirstShell(request) {
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(SHELL_CACHE);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch {
    // ignoreSearch: entry-point URLs (./?source=pwa etc.) must all resolve to
    // the one cached shell instead of each needing its own cached copy.
    const cached = await caches.match(request, { ignoreSearch: true })
      || await caches.match('./PRISM-v10-complete.html')
      || await caches.match('./');
    return cached || new Response(
      '<h1>PRISM is offline</h1><p>Please check your connection.</p>',
      { headers: { 'Content-Type': 'text/html' } }
    );
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request, { ignoreSearch: false });
  if (cached) return cached;

  try {
    const networkResponse = await fetch(request);

    if (networkResponse.ok || networkResponse.type === 'opaque') {
      const cache = await caches.open(
        request.url.includes(self.location.origin) ? SHELL_CACHE : DATA_CACHE
      );
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch {
    if (request.mode === 'navigate') {
      const offlineFallback = await caches.match('./PRISM-v10-complete.html');
      return offlineFallback || new Response(
        '<h1>PRISM is offline</h1><p>Please check your connection.</p>',
        { headers: { 'Content-Type': 'text/html' } }
      );
    }
    return new Response('', { status: 503, statusText: 'Offline' });
  }
}

// ── Message handler ──────────────────────────────────────────────────────────
self.addEventListener('message', event => {
  if (event.data?.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  if (event.data?.type === 'CACHE_INVALIDATE') {
    caches.delete(SHELL_CACHE).then(() => {
      event.ports?.[0]?.postMessage({ ok: true });
    });
  }
});

// ── Push notifications (Phase 1 — F5 PWA alpha) ───────────────────────────────
// Payload shape sent by prism-server/src/services/push.ts:
//   { title, body, data: { code, externalId } }
self.addEventListener('push', event => {
  let payload = { title: 'PRISM', body: 'You have a new alert.' };
  try { if (event.data) payload = Object.assign(payload, event.data.json()); } catch (err) {
    console.warn('[PRISM SW] Push payload parse failed:', err.message);
  }

  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      icon: './icon-192.png',
      badge: './icon-192.png',
      tag: (payload.data && payload.data.externalId) || 'prism-alert',
      data: payload.data || {},
    })
  );
});

// Clicking the notification focuses an existing PRISM tab, or opens one.
self.addEventListener('notificationclick', event => {
  event.notification.close();
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clients => {
      for (const client of clients) {
        if (client.url.includes('PRISM-v10-complete.html') && 'focus' in client) return client.focus();
      }
      if (self.clients.openWindow) return self.clients.openWindow('./PRISM-v10-complete.html');
    })
  );
});
