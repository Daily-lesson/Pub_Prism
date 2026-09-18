/* cortex-widget.js — the Cortex browser widget (docs/CONTRACTS.md §6).
 *
 * One framework-free classic script. Loading it defines `window.Cortex`. The widget touches
 * no host global except what the host passes into `Cortex.mount()`: the host adapter object
 * (`navigate`/`isAllowed`/`status`/`isLive`/`headers`) is the only bridge.
 *
 * Answer ladder (CONTRACTS §5): model rung (on-device ONNX, confidence >= gate) → keyword
 * rung (whole-word overlap against each intent's keyword bag) → fallback (honest "no
 * confident match" + the guide list). Cortex is a selector, not a generator: every word of
 * every reply comes from the host's registry, the host's status provider, or fixed copy in
 * this file — never a fabricated number, never an "answered by AI" claim on the
 * keyword/fallback rungs.
 *
 * DOM discipline: every id/class carries the `cortex-` prefix, all CSS is scoped under
 * `#cortex-root`, and every registry/host string reaches the DOM through `textContent` or the
 * single escaper `esc()` (& < > ").
 *
 * The tokenizer, softmax, BIO decoder, keyword matcher and STOPWORDS below are copies of the
 * core's — `tests/widgetDrift.test.ts` pins STOPWORDS, WORD_CHAR_RE, KEYWORD_SPLIT_RE and the
 * confidence default byte-for-byte against the core exports.
 */
(function () {
  'use strict';

  // ── shared constants (drift-tested against core) ──────────────────────────────────────────
  var STOPWORDS = ["the","a","an","and","or","of","to","in","on","for","with","is","are","was","were","be","been","do","does","did","how","what","where","when","which","who","whom","why","can","could","would","should","will","shall","may","might","must","i","me","my","we","our","you","your","it","its","this","that","these","those","there","here","from","by","at","as","into","onto","than","then","so","if","not","no","yes","please","just","about","over","under","up","down","out","off","again","more","most","some","any","all"];
  var WORD_CHAR_RE = /[\p{L}\p{N}]/u;
  var KEYWORD_SPLIT_RE = /[^a-z0-9]+/;
  var DEFAULT_CONFIDENCE_GATE = 0.6;
  var KEYWORD_MIN_LEN = 3;

  // KEEP IN SYNC with core/registry/builtins.ts BUILTIN_META_INTENTS (label + keywords) —
  // tests/widgetDrift.test.ts compares them byte-for-byte.
  var BUILTIN_META = {
    greeting: {
      id: 'greeting', family: 'meta', label: 'Greeting', slots: [],
      keywords: 'hello hey greetings thanks thank bye goodbye morning afternoon evening howdy',
      templates: [], paraphrases: []
    },
    out_of_domain: {
      id: 'out_of_domain', family: 'meta', label: 'Something else', slots: [],
      keywords: '', templates: [], paraphrases: []
    }
  };
  // KEEP IN SYNC with core/infer/planner.ts GREETING_COPY / OUT_OF_DOMAIN_COPY / FALLBACK_COPY —
  // the same registry must answer identically locally and via serverUrl (drift-tested).
  var META_COPY = {
    greeting: 'Hi! Ask me how to do something in this app, or what its current state is.',
    out_of_domain: "That's outside what I can help with here — I only know this app. Try asking how to do something in it.",
    fallback: "I'm not sure what you're asking."
  };

  var STOP = Object.create(null);   // null-prototype: the word "constructor" is not a stopword
  for (var si = 0; si < STOPWORDS.length; si++) STOP[STOPWORDS[si]] = true;

  // ── the one escaper ───────────────────────────────────────────────────────────────────────
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  // ── tokenizer (CONTRACTS §3) ──────────────────────────────────────────────────────────────
  function isWordChar(ch) { return WORD_CHAR_RE.test(ch); }
  function wordSplit(text) {
    var tokens = [], cur = '';
    for (var ch of text) {
      if (isWordChar(ch)) cur += ch;
      else if (cur) { tokens.push(cur); cur = ''; }
    }
    if (cur) tokens.push(cur);
    return tokens;
  }
  function wordSplitWithOffsets(text) {
    var tokens = [], cur = '', start = -1, i = 0;
    for (var ch of text) {
      if (isWordChar(ch)) { if (start === -1) start = i; cur += ch; }
      else if (cur) { tokens.push({ text: cur, start: start, end: i }); cur = ''; start = -1; }
      i += ch.length;
    }
    if (cur) tokens.push({ text: cur, start: start, end: i });
    return tokens;
  }
  function tokenize(text, cfg) {
    var lowered = cfg.lower ? text.toLowerCase() : text;
    var words = wordSplit(lowered);
    var ids = words.slice(0, cfg.maxLen).map(function (w) {
      return Object.prototype.hasOwnProperty.call(cfg.vocab, w) ? cfg.vocab[w] : cfg.unkId;
    });
    while (ids.length < cfg.maxLen) ids.push(cfg.padId);
    return ids;
  }

  // ── decode (CONTRACTS §4.3) ───────────────────────────────────────────────────────────────
  function softmax(arr) {
    var max = -Infinity, i;
    for (i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    var exps = new Float64Array(arr.length), sum = 0;
    for (i = 0; i < arr.length; i++) { var e = Math.exp(arr[i] - max); exps[i] = e; sum += e; }
    for (i = 0; i < exps.length; i++) exps[i] /= sum;
    return exps;
  }
  function argmax(arr) {
    var best = 0, bv = -Infinity;
    for (var i = 0; i < arr.length; i++) { if (arr[i] > bv) { bv = arr[i]; best = i; } }
    return best;
  }
  function decodeSlots(utterance, bioLabels, tokens) {
    var slots = [], curName = null, curStart = -1, curEnd = -1;
    function flush() {
      if (curName === null) return;
      slots.push({ name: curName, value: utterance.slice(curStart, curEnd), start: curStart, end: curEnd });
      curName = null;
    }
    for (var t = 0; t < bioLabels.length && t < tokens.length; t++) {
      var label = bioLabels[t];
      if (label === 'O' || label.indexOf('-') === -1) { flush(); continue; }
      var dash = label.indexOf('-');
      var prefix = label.slice(0, dash), name = label.slice(dash + 1);
      var tok = tokens[t];
      if (prefix === 'B' || name !== curName) { flush(); curName = name; curStart = tok.start; curEnd = tok.end; }
      else curEnd = tok.end;
    }
    flush();
    return slots;
  }

  // ── keyword rung (CONTRACTS §5.2–5.3) ────────────────────────────────────────────────────
  function keywordTokens(text) {
    var parts = String(text == null ? '' : text).toLowerCase().split(KEYWORD_SPLIT_RE);
    var out = [], seen = Object.create(null);
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (p.length < KEYWORD_MIN_LEN || STOP[p] || seen[p]) continue;
      seen[p] = true; out.push(p);
    }
    return out;
  }
  function keywordBag(intent) {
    var bag = Object.create(null), toks = keywordTokens((intent.keywords || '') + ' ' + (intent.label || ''));
    for (var i = 0; i < toks.length; i++) bag[toks[i]] = true;
    return bag;
  }
  function keywordMatch(query, intents) {
    var toks = keywordTokens(query);
    var best = null, bestScore = 0;
    for (var i = 0; i < intents.length; i++) {
      var bag = intents[i]._bag, s = 0;
      for (var j = 0; j < toks.length; j++) if (bag[toks[j]]) s++;
      if (s > bestScore) { bestScore = s; best = intents[i]; }   // strict > keeps taxonomy order on ties
    }
    return best ? { intent: best, score: bestScore } : null;
  }

  // ── slot detection for non-model rungs (CONTRACTS §2.3 rule: longest label first) ────────
  function detectSlots(utterance, intent, registry) {
    var out = [], taken = [];
    var lower = utterance.toLowerCase();
    var names = intent.slots || [];
    for (var n = 0; n < names.length; n++) {
      var def = registry.slots[names[n]];
      if (!def || !def.vocab || !def.vocab.length) continue;
      var entries = def.vocab.slice().sort(function (a, b) { return String(b.label).length - String(a.label).length; });
      var found = null;
      for (var e = 0; e < entries.length && !found; e++) {
        var lab = String(entries[e].label || '').toLowerCase();
        if (!lab) continue;
        for (var at = lower.indexOf(lab); at !== -1; at = lower.indexOf(lab, at + 1)) {
          var end = at + lab.length;
          var wholeWord = (at === 0 || !isWordChar(lower[at - 1])) && (end >= lower.length || !isWordChar(lower[end]));
          var overlaps = taken.some(function (r) { return at < r[1] && end > r[0]; });
          if (wholeWord && !overlaps) { found = { name: names[n], value: utterance.slice(at, end), start: at, end: end, resolvedId: entries[e].id }; break; }
        }
      }
      if (found) { out.push(found); taken.push([found.start, found.end]); }   // one match per slot category
    }
    out.sort(function (a, b) { return a.start - b.start; });
    return out;
  }
  function resolveSlotIds(slots, intent, registry) {
    for (var i = 0; i < slots.length; i++) {
      var s = slots[i];
      if (s.resolvedId) continue;
      var def = registry.slots[s.name];
      if (!def || !def.vocab) continue;
      var v = String(s.value).toLowerCase();
      for (var k = 0; k < def.vocab.length; k++) {
        if (String(def.vocab[k].label).toLowerCase() === v) { s.resolvedId = def.vocab[k].id; break; }
      }
    }
    return slots;
  }

  // ── state ─────────────────────────────────────────────────────────────────────────────────
  var S = null; // the mounted instance; replaced on every mount()
  function freshState() {
    return {
      mounted: false, opts: {}, host: {}, ui: {},
      registry: null, registryLoaded: false, registryError: null, readyPromise: null,
      intents: [], intentById: {},
      confidenceGate: DEFAULT_CONFIDENCE_GATE,
      modelBase: null, runtimeBase: null, serverUrl: null,
      modelState: 'absent', modelError: null, modelPromise: null, model: null,
      root: null, els: {}, acts: {}, isOpen: false, lastEvidence: null, lastAnswer: null,
      listeners: [], askSeq: 0
    };
  }

  function resolveUrl(rel) { return new URL(rel, document.baseURI).href; }
  function dirUrl(base) {
    var u = resolveUrl(base || '');
    return u.charAt(u.length - 1) === '/' ? u : u + '/';
  }

  // ── registry ──────────────────────────────────────────────────────────────────────────────
  function prepareRegistry(raw) {
    if (!raw || typeof raw !== 'object') throw new Error('cortex: registry must be an object');
    if (raw.registryVersion !== '1') throw new Error('cortex: unsupported registryVersion (expected "1")');
    if (!Array.isArray(raw.intents)) throw new Error('cortex: registry.intents must be an array');
    var reg = {
      registryVersion: '1',
      app: raw.app || {},
      slots: (raw.slots && typeof raw.slots === 'object') ? raw.slots : {},
      intents: raw.intents.map(function (it) { return Object.assign({}, it); })
    };
    if (!reg.app.name) reg.app.name = 'This app';
    if (!reg.app.assistantName) reg.app.assistantName = reg.app.name + ' Assistant';
    var seen = Object.create(null);   // an intent id "constructor" is legal, not a duplicate
    for (var i = 0; i < reg.intents.length; i++) {
      var it = reg.intents[i];
      if (!it || typeof it.id !== 'string' || !/^[a-z][a-z0-9_]*$/.test(it.id)) throw new Error('cortex: intent #' + i + ' has an invalid id');
      if (seen[it.id]) throw new Error('cortex: duplicate intent id "' + it.id + '"');
      seen[it.id] = true;
      if (['howto', 'status', 'meta'].indexOf(it.family) === -1) throw new Error('cortex: intent "' + it.id + '" has an invalid family');
      if (it.family === 'howto' && !(it.answer && Array.isArray(it.answer.steps) && it.answer.steps.length)) throw new Error('cortex: howto intent "' + it.id + '" needs answer.steps');
      if (it.family === 'status' && !(it.status && typeof it.status.unavailable === 'string')) throw new Error('cortex: status intent "' + it.id + '" needs status.unavailable');
      if (!Array.isArray(it.slots)) it.slots = [];
    }
    ['greeting', 'out_of_domain'].forEach(function (id) {
      if (!seen[id]) reg.intents.push(Object.assign({}, BUILTIN_META[id]));
    });
    for (var k = 0; k < reg.intents.length; k++) {
      // CONTRACTS §2.11: out_of_domain has an EMPTY keyword bag by construction — its label
      // words must never make it a keyword-rung target (the model is its only route).
      reg.intents[k]._bag = reg.intents[k].id === 'out_of_domain' ? {} : keywordBag(reg.intents[k]);
    }
    return reg;
  }
  async function loadRegistry(st) {
    var raw;
    if (st.opts.registry) raw = st.opts.registry;
    else if (st.opts.registryUrl) {
      var res = await fetch(resolveUrl(st.opts.registryUrl), { credentials: 'same-origin' });
      if (!res.ok) throw new Error('cortex: registry fetch failed (' + res.status + ')');
      raw = await res.json();
    } else throw new Error('cortex: mount needs registry or registryUrl');
    var reg = prepareRegistry(raw);
    st.registry = reg;
    st.intents = reg.intents;
    st.intentById = {};
    for (var i = 0; i < reg.intents.length; i++) st.intentById[reg.intents[i].id] = reg.intents[i];
    st.registryLoaded = true;
  }

  // ── model rung: lazy loader (CONTRACTS §4.6, §6.1, §6.3) ──────────────────────────────────
  async function fetchBuf(url) {
    var res = await fetch(url, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('fetch failed for ' + url + ' (' + res.status + ')');
    return res.arrayBuffer();
  }
  async function sha256Hex(buf) {
    if (!(window.crypto && window.crypto.subtle)) return null;
    try {
      var digest = await window.crypto.subtle.digest('SHA-256', buf);
      var bytes = new Uint8Array(digest), hex = '';
      for (var i = 0; i < bytes.length; i++) hex += bytes[i].toString(16).padStart(2, '0');
      return hex;
    } catch (e) { return null; }
  }
  // Best-effort sha256 re-verification against ledger.json: a mismatch is refused; a missing
  // crypto.subtle (non-secure context) skips the check rather than failing it (§4.6).
  async function fetchVerified(url, expectedSha256, label) {
    var buf = await fetchBuf(url);
    if (expectedSha256) {
      var actual = await sha256Hex(buf);
      if (actual && actual !== expectedSha256) throw new Error(label + ' sha256 mismatch - refusing a corrupted or stale artifact');
    }
    return buf;
  }
  function bufToJson(buf) { return JSON.parse(new TextDecoder('utf-8').decode(buf)); }

  function loadModel(st) {
    if (st.modelPromise) return st.modelPromise;
    if (!st.modelBase) return Promise.reject(new Error('cortex: keyword-only mode (no modelBase configured)'));
    st.modelState = 'loading';
    st.modelPromise = (async function () {
      var base = dirUrl(st.modelBase), rt = dirUrl(st.runtimeBase || 'cortex/runtime/');
      var ledger = bufToJson(await fetchVerified(base + 'ledger.json', null, 'ledger'));
      if (!ledger || !ledger.onnx || !ledger.tokenizer || !ledger.labels) throw new Error('ledger.json is missing onnx/tokenizer/labels entries');
      var tokenizerCfg = bufToJson(await fetchVerified(base + ledger.tokenizer.file, ledger.tokenizer.sha256, 'tokenizer'));
      var labels = bufToJson(await fetchVerified(base + ledger.labels.file, ledger.labels.sha256, 'labels'));
      var onnxBuf = await fetchVerified(base + ledger.onnx.file, ledger.onnx.sha256, 'onnx');
      if (!Array.isArray(labels.intents) || !Array.isArray(labels.slots)) throw new Error('labels.json is missing intents/slots');
      if (typeof tokenizerCfg.maxLen !== 'number' || !tokenizerCfg.vocab) throw new Error('tokenizer.json is malformed');
      var ort = await import(/* webpackIgnore: true */ rt + 'ort.wasm.min.mjs');
      ort.env.wasm.wasmPaths = rt;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
      var session = await ort.InferenceSession.create(new Uint8Array(onnxBuf), { executionProviders: ['wasm'] });
      return { ort: ort, session: session, tokenizerCfg: tokenizerCfg, labels: labels, ledger: ledger };
    })();
    st.modelPromise.then(function (m) {
      st.model = m; st.modelState = 'ready'; st.modelError = null; updateFoot(st);
    }, function (err) {
      st.model = null; st.modelState = 'failed'; st.modelError = String(err && err.message || err); updateFoot(st);
    });
    return st.modelPromise;
  }

  async function classifyWith(st, utterance) {
    var art = await loadModel(st);
    var cfg = art.tokenizerCfg;
    var ids = tokenize(utterance, cfg);
    var idsBig = BigInt64Array.from(ids.map(function (n) { return BigInt(n); }));
    var inputTensor = new art.ort.Tensor('int64', idsBig, [1, cfg.maxLen]);
    var outputs = await art.session.run({ input_ids: inputTensor });
    var intentLogits = outputs.intent_logits.data;
    var slotLogits = outputs.slot_logits.data;

    var probs = softmax(intentLogits);
    var intentIdx = argmax(probs);
    var rawIntent = art.labels.intents[intentIdx];
    // A label the registry does not declare is treated as out_of_domain.
    var intent = (rawIntent && st.intentById[rawIntent]) ? rawIntent : 'out_of_domain';
    var intentConf = probs[intentIdx];

    var numSlotLabels = art.labels.slots.length;
    var offsetTokens = wordSplitWithOffsets(utterance);
    var realLen = Math.min(offsetTokens.length, cfg.maxLen);
    var bioLabels = [];
    for (var t = 0; t < realLen; t++) {
      var base = t * numSlotLabels, best = 0, bv = -Infinity;
      for (var c = 0; c < numSlotLabels; c++) { var v = slotLogits[base + c]; if (v > bv) { bv = v; best = c; } }
      bioLabels.push(art.labels.slots[best] || 'O');
    }
    var slots = decodeSlots(utterance, bioLabels, offsetTokens.slice(0, realLen));
    resolveSlotIds(slots, st.intentById[intent], st.registry);
    return { intent: intent, intentConf: intentConf, slots: slots, modelVersion: art.ledger.version, modelSha256: art.ledger.onnx.sha256 };
  }

  // ── planner (CONTRACTS §5.4–5.5) ──────────────────────────────────────────────────────────
  function allowed(st, need) {
    if (need == null || need === '') return true;
    var h = st.host;
    if (!h || typeof h.isAllowed !== 'function') return true;          // absent = fail-open, UX only
    try { return !!h.isAllowed(need); } catch (e) { return true; }
  }
  function gateLinks(st, links) {
    var out = [];
    (links || []).forEach(function (lk) {
      if (!lk || typeof lk.label !== 'string') return;
      if (!allowed(st, lk.need)) return;                                  // refused links are DROPPED, never disabled
      out.push({ label: lk.label, target: lk.target, need: lk.need });
    });
    return out;
  }
  function slotPrefix(slots) {
    if (!slots || !slots.length) return '';
    return 'Noted — you mentioned ' + slots.map(function (s) { return '"' + s.value + '"'; }).join(', ') + '. ';
  }
  function slotEntries(slots) {
    return (slots || []).map(function (s) { return 'slot.' + s.name + ':' + (s.resolvedId != null ? s.resolvedId : s.value); });
  }
  async function plan(st, intent, slots, cls) {
    var ev = {
      modelVersion: cls && cls.modelVersion != null ? cls.modelVersion : null,
      modelSha256: cls && cls.modelSha256 != null ? cls.modelSha256 : null,
      intent: intent ? intent.id : null,
      intentConf: cls && typeof cls.intentConf === 'number' ? cls.intentConf : null,
      ladderRung: cls ? cls.ladderRung : 'fallback',
      plannerTemplateId: 'planner.fallback',
      registryEntriesUsed: []
    };
    if (!intent) {
      return { answerText: META_COPY.fallback, chips: [], evidence: ev, guide: true };
    }
    ev.registryEntriesUsed.push('intent:' + intent.id);
    if (intent.family === 'meta') {
      ev.plannerTemplateId = 'planner.meta.' + intent.id;
      return { answerText: META_COPY[intent.id] || META_COPY.out_of_domain, chips: [], evidence: ev, guide: true };
    }
    if (intent.family === 'howto') {
      ev.plannerTemplateId = 'planner.howto.' + intent.id;
      ev.registryEntriesUsed = ev.registryEntriesUsed.concat(slotEntries(slots));
      var steps = intent.answer.steps.map(function (s, i) { return (i + 1) + '. ' + s; }).join(' ');
      return { answerText: slotPrefix(slots) + steps, chips: gateLinks(st, intent.answer.links), evidence: ev, guide: false };
    }
    // status family — a number is only ever printed when the host's provider supplies it
    ev.plannerTemplateId = 'planner.status.' + intent.id;
    ev.registryEntriesUsed = ev.registryEntriesUsed.concat(slotEntries(slots));
    var text = null;
    if (st.host && typeof st.host.status === 'function') {
      try {
        var r = st.host.status(intent.id, (slots || []).map(function (s) { return { name: s.name, value: s.value, resolvedId: s.resolvedId }; }));
        if (r && typeof r.then === 'function') r = await r;
        if (typeof r === 'string' && r.length) text = r;
      } catch (e) { text = null; }
    }
    if (text === null) text = intent.status.unavailable;
    return { answerText: text, chips: gateLinks(st, intent.status.links), evidence: ev, guide: false };
  }

  // ── provenance (CONTRACTS §6.4) ───────────────────────────────────────────────────────────
  function pct(x) { return Math.round(x * 100) + '%'; }
  function provenanceFor(st, ev, extra) {
    var line;
    if (extra && extra.serverRung) {
      line = 'Answered by the server';
      line += ' (' + extra.serverRung + ' rung' + (extra.serverRung === 'cortex' && ev.modelVersion ? ', model v' + ev.modelVersion + (typeof ev.intentConf === 'number' ? ', confidence ' + pct(ev.intentConf) : '') : '') + ').';
    } else if (ev.ladderRung === 'cortex') {
      line = 'Matched by the on-device model v' + ev.modelVersion + ' (confidence ' + pct(ev.intentConf) + ').';
    } else if (ev.ladderRung === 'keyword') {
      line = 'Keyword match on your wording';
      if (st.modelState === 'ready') line += ' — the model had no confident match';
      else if (st.modelState === 'failed') line += ' — the model could not be loaded';
      else if (st.modelState === 'absent') line += ' — no model is configured';
      line += '.';
    } else {
      line = 'No confident match';
      if (st.modelState === 'failed') line += ' — the model could not be loaded';
      else if (st.modelState === 'absent') line += ' — no model is configured';
      line += '.';
    }
    if (extra && extra.serverFailed) line = 'The server could not be reached — answered locally. ' + line;
    return line;
  }

  // ── server mode ───────────────────────────────────────────────────────────────────────────
  function isLive(st) {
    var h = st.host;
    if (!st.serverUrl || !h || typeof h.isLive !== 'function') return false;
    try { return !!h.isLive(); } catch (e) { return false; }
  }
  async function askServer(st, query) {
    var headers = { 'Content-Type': 'application/json' };
    if (st.host && typeof st.host.headers === 'function') {
      try { Object.assign(headers, st.host.headers() || {}); } catch (e) { /* ignore */ }
    }
    var res = await fetch(resolveUrl(st.serverUrl), { method: 'POST', headers: headers, body: JSON.stringify({ query: query }), credentials: 'same-origin' });
    if (!res.ok) throw new Error('server answered ' + res.status);
    var body = await res.json();
    if (!body || typeof body.answer !== 'string') throw new Error('server response malformed');
    var sev = body.evidence || {};
    var ev = {
      modelVersion: body.modelVersion != null ? body.modelVersion : (sev.modelVersion != null ? sev.modelVersion : null),
      modelSha256: sev.modelSha256 != null ? sev.modelSha256 : null,
      intent: body.intent != null ? body.intent : (sev.intent != null ? sev.intent : null),
      intentConf: typeof body.intentConf === 'number' ? body.intentConf : (typeof sev.intentConf === 'number' ? sev.intentConf : null),
      // The rung the SERVER answered on (§5.4's union), never a widget-invented 'server' value;
      // the transport is recorded separately as `serverRung` on the result.
      ladderRung: (body.ladderRung || sev.ladderRung || 'fallback'),
      plannerTemplateId: sev.plannerTemplateId || null,
      registryEntriesUsed: Array.isArray(sev.registryEntriesUsed) ? sev.registryEntriesUsed : []
    };
    var serverRung = body.ladderRung || sev.ladderRung || null;
    return { answerText: body.answer, chips: gateLinks(st, body.chips), evidence: ev, guide: serverRung === 'fallback', serverRung: serverRung };
  }

  // ── the ladder ────────────────────────────────────────────────────────────────────────────
  async function localAnswer(st, query, serverFailed) {
    var cls = null;
    if (st.modelBase && st.modelState !== 'failed') {
      try { cls = await classifyWith(st, query); } catch (e) { cls = null; }
    }
    var result;
    if (cls && cls.intentConf >= st.confidenceGate) {
      cls.ladderRung = 'cortex';
      result = await plan(st, st.intentById[cls.intent], cls.slots, cls);
    } else {
      var km = keywordMatch(query, st.intents);
      if (km) {
        var slots = detectSlots(query, km.intent, st.registry);
        result = await plan(st, km.intent, slots, { ladderRung: 'keyword', intentConf: null, modelVersion: null, modelSha256: null });
      } else {
        result = await plan(st, null, [], null);
      }
    }
    result.provenance = provenanceFor(st, result.evidence, { serverFailed: !!serverFailed });
    return result;
  }
  async function answer(st, query) {
    await ready(st);
    query = String(query == null ? '' : query).trim();
    if (!query) return await finish(st, await plan(st, null, [], null), null);
    if (isLive(st)) {
      try {
        var r = await askServer(st, query);
        r.provenance = provenanceFor(st, r.evidence, { serverRung: r.serverRung });
        return await finish(st, r, null);
      } catch (e) {
        return await finish(st, await localAnswer(st, query, true), null);
      }
    }
    return await finish(st, await localAnswer(st, query, false), null);
  }
  async function finish(st, r) {
    if (!r.provenance) r.provenance = provenanceFor(st, r.evidence, {});
    r.intent = r.evidence.intent;
    r.ladderRung = r.evidence.ladderRung;
    st.lastEvidence = r.evidence;
    st.lastAnswer = r;
    return r;
  }
  function ready(st) {
    if (!st.readyPromise) return Promise.reject(new Error('cortex: not mounted'));
    return st.readyPromise;
  }

  // ── UI ────────────────────────────────────────────────────────────────────────────────────
  var CSS = [
    '#cortex-root{--cortex-bg:#ffffff;--cortex-fg:#1b1f27;--cortex-muted:#5f6675;--cortex-line:#d9dde5;--cortex-accent:#2f6fed;--cortex-accent-fg:#ffffff;--cortex-card:#f6f7fa;--cortex-backdrop:rgba(10,12,18,.45);--cortex-shadow:-18px 0 48px rgba(0,0,0,.25);--cortex-font:system-ui,-apple-system,"Segoe UI","Helvetica Neue",Arial,sans-serif;--cortex-z:99990;font-family:var(--cortex-font);font-size:15px;line-height:1.45;color:var(--cortex-fg)}',
    '#cortex-root[data-cortex-theme="dark"]{--cortex-bg:#161a22;--cortex-fg:#e8ebf1;--cortex-muted:#9aa3b5;--cortex-line:#2c3340;--cortex-accent:#7aa2ff;--cortex-accent-fg:#101318;--cortex-card:#1e232d;--cortex-backdrop:rgba(0,0,0,.55)}',
    '@media (prefers-color-scheme:dark){#cortex-root[data-cortex-theme="auto"]{--cortex-bg:#161a22;--cortex-fg:#e8ebf1;--cortex-muted:#9aa3b5;--cortex-line:#2c3340;--cortex-accent:#7aa2ff;--cortex-accent-fg:#101318;--cortex-card:#1e232d;--cortex-backdrop:rgba(0,0,0,.55)}}',
    '#cortex-root *{box-sizing:border-box}',
    '#cortex-root .cortex-launcher{position:fixed;right:18px;bottom:18px;z-index:var(--cortex-z);border:1px solid var(--cortex-line);background:var(--cortex-bg);color:var(--cortex-fg);border-radius:999px;padding:10px 16px;min-height:44px;cursor:pointer;font:inherit;font-weight:600;box-shadow:0 6px 24px rgba(0,0,0,.18)}',
    '#cortex-root .cortex-launcher:hover{border-color:var(--cortex-accent)}',
    '#cortex-root .cortex-back{position:fixed;inset:0;z-index:calc(var(--cortex-z) + 1);background:var(--cortex-backdrop);display:none}',
    '#cortex-root .cortex-back.cortex-open{display:block}',
    '#cortex-root .cortex-panel{position:absolute;top:0;right:0;bottom:0;width:min(440px,100vw);background:var(--cortex-bg);color:var(--cortex-fg);border-left:1px solid var(--cortex-line);display:flex;flex-direction:column;box-shadow:var(--cortex-shadow)}',
    '#cortex-root .cortex-hd{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--cortex-line)}',
    '#cortex-root .cortex-hd-text{flex:1;min-width:0}',
    '#cortex-root .cortex-title{font-weight:700;letter-spacing:.02em}',
    '#cortex-root .cortex-sub{font-size:.82em;color:var(--cortex-muted)}',
    '#cortex-root .cortex-x{background:none;border:1px solid var(--cortex-line);border-radius:8px;color:inherit;cursor:pointer;padding:6px 12px;min-height:36px;font:inherit}',
    '#cortex-root .cortex-form{display:flex;gap:8px;padding:12px 16px 0}',
    '#cortex-root .cortex-q{flex:1;min-width:0;padding:10px 12px;border:1px solid var(--cortex-line);border-radius:10px;background:transparent;color:inherit;font:inherit}',
    '#cortex-root .cortex-q:focus{outline:2px solid var(--cortex-accent);outline-offset:1px}',
    '#cortex-root .cortex-go{border:1px solid var(--cortex-accent);background:var(--cortex-accent);color:var(--cortex-accent-fg);border-radius:10px;padding:0 14px;min-height:40px;cursor:pointer;font:inherit;font-weight:600}',
    '#cortex-root .cortex-body{flex:1;overflow-y:auto;padding:12px 16px 20px}',
    '#cortex-root .cortex-card{border:1px solid var(--cortex-line);border-radius:12px;padding:11px 13px;margin-bottom:10px;background:var(--cortex-card)}',
    '#cortex-root .cortex-answer-text{white-space:pre-wrap}',
    '#cortex-root .cortex-chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:9px}',
    '#cortex-root .cortex-chip{border:1px solid var(--cortex-accent);color:var(--cortex-accent);background:transparent;border-radius:999px;padding:7px 13px;cursor:pointer;font:inherit;font-size:.9em;font-weight:600;min-height:34px}',
    '#cortex-root .cortex-chip:hover{background:var(--cortex-accent);color:var(--cortex-accent-fg)}',
    '#cortex-root .cortex-prov{margin-top:8px;font-size:.8em;color:var(--cortex-muted);font-style:italic}',
    '#cortex-root .cortex-guide-hd{font-size:.8em;text-transform:uppercase;letter-spacing:.06em;color:var(--cortex-muted);margin:14px 0 8px}',
    '#cortex-root .cortex-guide-item{border:1px solid var(--cortex-line);border-radius:12px;padding:10px 13px;margin-bottom:8px}',
    '#cortex-root .cortex-guide-item summary{cursor:pointer;font-weight:600;list-style:none}',
    '#cortex-root .cortex-guide-item summary::-webkit-details-marker{display:none}',
    '#cortex-root .cortex-steps{margin:8px 0 4px;padding-left:20px;font-size:.93em}',
    '#cortex-root .cortex-empty{color:var(--cortex-muted);text-align:center;padding:20px 10px}',
    '#cortex-root .cortex-busy{color:var(--cortex-muted);font-style:italic;padding:6px 0}',
    '#cortex-root .cortex-foot{padding:9px 16px;border-top:1px solid var(--cortex-line);font-size:.78em;color:var(--cortex-muted)}',
    '@media (max-width:520px){#cortex-root .cortex-panel{width:100vw;border-left:none}}'
  ].join('\n');

  function ensureStyle() {
    if (document.getElementById('cortex-style')) return;
    var style = document.createElement('style');
    style.id = 'cortex-style';
    style.textContent = CSS;
    var head = document.head || document.documentElement;
    head.insertBefore(style, head.firstChild); // first in the cascade so a host sheet can override the variables
  }
  function on(st, target, type, fn, opts) {
    target.addEventListener(type, fn, opts);
    st.listeners.push(function () { target.removeEventListener(type, fn, opts); });
  }
  function buildUi(st) {
    ensureStyle();
    var old = document.getElementById('cortex-root');
    if (old && old.parentNode) old.parentNode.removeChild(old);

    var root = document.createElement('div');
    root.id = 'cortex-root';
    root.setAttribute('data-cortex-theme', st.ui.theme || 'auto');

    var title = st.ui.title || (st.registry ? st.registry.app.assistantName : 'Assistant');
    var app = st.registry ? st.registry.app.name : '';
    root.innerHTML =
      (st.ui.launcher === false ? '' : '<button type="button" class="cortex-launcher" id="cortex-launcher" aria-haspopup="dialog">' + esc(title) + '</button>') +
      '<div class="cortex-back" id="cortex-back">' +
        '<div class="cortex-panel" id="cortex-panel" role="dialog" aria-modal="true" aria-labelledby="cortex-title">' +
          '<div class="cortex-hd"><div class="cortex-hd-text"><div class="cortex-title" id="cortex-title">' + esc(title) + '</div>' +
            (app ? '<div class="cortex-sub" id="cortex-app">' + esc(app) + '</div>' : '') + '</div>' +
            '<button type="button" class="cortex-x" id="cortex-close" aria-label="Close">Close</button></div>' +
          '<form class="cortex-form" id="cortex-form"><input class="cortex-q" id="cortex-q" type="text" autocomplete="off" placeholder="Ask how to do something…" aria-label="Ask a question"/>' +
            '<button type="submit" class="cortex-go" id="cortex-go">Ask</button></form>' +
          '<div class="cortex-body" id="cortex-body"><div id="cortex-answer"></div><div id="cortex-guide"></div></div>' +
          '<div class="cortex-foot" id="cortex-foot"></div>' +
        '</div>' +
      '</div>';
    document.body.appendChild(root);
    st.root = root;
    st.els = {
      launcher: document.getElementById('cortex-launcher'),
      back: document.getElementById('cortex-back'),
      panel: document.getElementById('cortex-panel'),
      close: document.getElementById('cortex-close'),
      form: document.getElementById('cortex-form'),
      q: document.getElementById('cortex-q'),
      body: document.getElementById('cortex-body'),
      answer: document.getElementById('cortex-answer'),
      guide: document.getElementById('cortex-guide'),
      foot: document.getElementById('cortex-foot')
    };

    if (st.els.launcher) on(st, st.els.launcher, 'click', function () { openDrawer(st, ''); });
    on(st, st.els.back, 'click', function (ev) { if (ev.target === st.els.back) closeDrawer(st); });
    on(st, st.els.close, 'click', function () { closeDrawer(st); });
    on(st, st.els.form, 'submit', function (ev) { ev.preventDefault(); runAsk(st, st.els.q.value); });
    on(st, st.els.q, 'input', function () { if (!st.lastAnswer || st.els.answer.childNodes.length === 0) renderGuide(st, st.els.q.value); });
    on(st, st.els.body, 'click', function (ev) {
      var b = ev.target && ev.target.closest ? ev.target.closest('[data-cortex-act]') : null;
      if (!b || !root.contains(b)) return;
      var key = b.getAttribute('data-cortex-act');
      var lk = st.acts[key];
      if (!lk) return;
      closeDrawer(st);
      if (st.host && typeof st.host.navigate === 'function') {
        try { st.host.navigate(lk.target); } catch (e) { /* host error must not break the widget */ }
      }
    });
    on(st, document, 'keydown', function (ev) {
      if (ev.key === 'Escape' && st.isOpen) { closeDrawer(st); return; }
      var hk = st.ui.hotkey;
      if (hk && (ev.ctrlKey || ev.metaKey) && !ev.altKey && typeof ev.key === 'string' && ev.key.toLowerCase() === String(hk).toLowerCase()) {
        ev.preventDefault();
        if (st.isOpen) closeDrawer(st); else openDrawer(st, '');
      }
    });
    updateFoot(st);
    renderGuide(st, '');
  }
  function updateFoot(st) {
    if (!st.els || !st.els.foot) return;
    var t;
    if (st.serverUrl && isLive(st)) t = 'Answers come from the server while you are signed in.';
    else if (!st.modelBase) t = 'Keyword mode — no on-device model is configured.';
    else if (st.modelState === 'ready') t = 'On-device model v' + (st.model && st.model.ledger ? st.model.ledger.version : '?') + ' ready. Nothing you type leaves this page.';
    else if (st.modelState === 'failed') t = 'Keyword mode — the on-device model could not be loaded.';
    else if (st.modelState === 'loading') t = 'Loading the on-device model…';
    else t = 'Keyword mode until the on-device model loads on your first question.';
    st.els.foot.textContent = t;
  }

  // Guide list: the howto intents, filtered by substring overlap of the query's words against
  // each intent's label + keywords (all of them when the query is empty).
  function intentScore(it, q) {
    if (!q) return 1;
    var hay = (String(it.label || '') + ' ' + String(it.keywords || '')).toLowerCase();
    var toks = q.toLowerCase().split(/\s+/).filter(Boolean), s = 0;
    for (var i = 0; i < toks.length; i++) { if (hay.indexOf(toks[i]) >= 0) s++; }
    return s;
  }
  function chipHtml(st, lk, key) {
    st.acts[key] = lk;
    return '<button type="button" class="cortex-chip" data-cortex-act="' + esc(key) + '">' + esc(lk.label) + '</button>';
  }
  function renderGuide(st, q) {
    var host = st.els.guide; if (!host) return;
    var items = st.intents.filter(function (it) { return it.family === 'howto'; })
      .map(function (it) { return { it: it, s: intentScore(it, q) }; })
      .filter(function (x) { return x.s > 0; })
      .sort(function (a, b) { return b.s - a.s; });
    var heading = 'Guide';
    if (!items.length) {
      // Nothing overlaps the query: show the whole guide rather than an empty list (the
      // fallback answer promises "here is what I can help with").
      heading = 'No guide topic matches those words — everything I can help with';
      items = st.intents.filter(function (it) { return it.family === 'howto'; }).map(function (it) { return { it: it, s: 1 }; });
      q = '';
    }
    if (!items.length) {
      host.innerHTML = '<div class="cortex-guide-hd">Guide</div><div class="cortex-empty">This app has no guide topics yet.</div>';
      return;
    }
    host.innerHTML = '<div class="cortex-guide-hd">' + esc(heading) + '</div>' + items.map(function (x, ix) {
      var it = x.it;
      var chips = gateLinks(st, it.answer.links).map(function (lk, j) { return chipHtml(st, lk, 'g' + ix + '_' + j); }).join('');
      return '<details class="cortex-guide-item"' + (q && ix === 0 ? ' open' : '') + '><summary>' + esc(it.label) + '</summary>' +
        '<ol class="cortex-steps">' + it.answer.steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>' +
        (chips ? '<div class="cortex-chips">' + chips + '</div>' : '') + '</details>';
    }).join('');
  }
  function renderAnswer(st, r, q) {
    if (!st.els.answer) return;
    st.acts = {};
    var chips = (r.chips || []).map(function (lk, j) { return chipHtml(st, lk, 'a' + j); }).join('');
    st.els.answer.innerHTML = '<div class="cortex-card" id="cortex-answer-card"><div class="cortex-answer-text">' + esc(r.answerText) + '</div>' +
      (chips ? '<div class="cortex-chips">' + chips + '</div>' : '') +
      '<div class="cortex-prov" id="cortex-prov">' + esc(r.provenance || '') + '</div></div>';
    if (r.guide) renderGuide(st, q); else st.els.guide.innerHTML = '';
  }
  function runAsk(st, query) {
    var seq = ++st.askSeq;
    query = String(query == null ? '' : query);
    if (!query.trim()) { st.lastAnswer = null; st.els.answer.innerHTML = ''; renderGuide(st, ''); return Promise.resolve(null); }
    st.els.answer.innerHTML = '<div class="cortex-busy" id="cortex-busy">Thinking…</div>';
    return answer(st, query).then(function (r) {
      if (seq !== st.askSeq || !st.els.answer) return r;
      renderAnswer(st, r, query);
      updateFoot(st);
      return r;
    }, function (err) {
      if (seq !== st.askSeq || !st.els.answer) return null;
      st.els.answer.innerHTML = '<div class="cortex-card"><div class="cortex-answer-text">' + esc('Something went wrong: ' + (err && err.message ? err.message : err)) + '</div></div>';
      return null;
    });
  }
  function openDrawer(st, query) {
    if (!st.els.back) return;
    st.els.back.classList.add('cortex-open');
    st.isOpen = true;
    if (typeof query === 'string' && query.length) { st.els.q.value = query; runAsk(st, query); }
    else if (!st.lastAnswer) { st.els.answer.innerHTML = ''; renderGuide(st, st.els.q.value); }
    setTimeout(function () { try { st.els.q.focus(); } catch (e) { /* ignore */ } }, 0);
  }
  function closeDrawer(st) {
    if (!st.els.back) return;
    st.els.back.classList.remove('cortex-open');
    st.isOpen = false;
  }
  function unmount(st) {
    if (!st) return;
    st.listeners.forEach(function (off) { try { off(); } catch (e) { /* ignore */ } });
    st.listeners = [];
    if (st.root && st.root.parentNode) st.root.parentNode.removeChild(st.root);
    st.root = null; st.els = {};
  }

  // ── public API (CONTRACTS §6) ─────────────────────────────────────────────────────────────
  function mount(opts) {
    opts = opts || {};
    if (S) unmount(S);
    var st = freshState();
    S = st;
    st.mounted = true;
    st.opts = opts;
    st.host = opts.host || {};
    st.ui = Object.assign({ launcher: true, hotkey: null, title: null, theme: 'auto' }, opts.ui || {});
    st.confidenceGate = (typeof opts.confidenceGate === 'number' && opts.confidenceGate >= 0 && opts.confidenceGate <= 1) ? opts.confidenceGate : DEFAULT_CONFIDENCE_GATE;
    st.modelBase = opts.modelBase ? String(opts.modelBase) : null;
    st.runtimeBase = opts.runtimeBase ? String(opts.runtimeBase) : null;
    st.serverUrl = opts.serverUrl ? String(opts.serverUrl) : null;
    st.modelState = st.modelBase ? 'idle' : 'absent';

    function build() { if (document.body) buildUi(st); }
    st.readyPromise = loadRegistry(st).then(function () {
      if (S !== st) return;
      if (document.body) build();
      else document.addEventListener('DOMContentLoaded', build, { once: true });
    }, function (err) {
      st.registryError = String(err && err.message || err);
      throw err;
    });
    st.readyPromise.catch(function () { /* surfaced through state().registryError and every ask() rejection */ });
    return st.readyPromise;
  }
  function api() { if (!S) throw new Error('cortex: not mounted'); return S; }

  window.Cortex = {
    mount: mount,
    open: function (query) {
      var st = api();
      return ready(st).then(function () { openDrawer(st, query); });
    },
    close: function () { if (S) closeDrawer(S); },
    ask: function (query) { return answer(api(), query); },
    classify: function (query) {
      var st = api();
      return ready(st).then(function () {
        if (!st.modelBase) throw new Error('cortex: keyword-only mode (no modelBase configured)');
        return classifyWith(st, String(query == null ? '' : query)).then(function (c) {
          return { intent: c.intent, intentConf: c.intentConf, slots: c.slots };
        });
      });
    },
    warm: function () {
      var st = api();
      return ready(st).then(function () {
        if (!st.modelBase) return null;
        if (st.modelState === 'failed') { st.modelPromise = null; st.modelError = null; }
        return loadModel(st).then(function () { return true; }, function () { return false; });
      });
    },
    state: function () {
      var st = S;
      if (!st) return { mounted: false };
      return {
        mounted: true,
        registryLoaded: st.registryLoaded,
        registryError: st.registryError,
        app: st.registry ? { slug: st.registry.app.slug || null, name: st.registry.app.name, assistantName: st.registry.app.assistantName } : null,
        intents: st.intents.map(function (it) { return it.id; }),
        mode: (st.serverUrl && isLive(st)) ? 'server' : (st.modelBase ? 'model' : 'keyword-only'),
        modelState: st.modelState,
        modelError: st.modelError,
        modelVersion: st.model && st.model.ledger ? st.model.ledger.version : null,
        confidenceGate: st.confidenceGate,
        open: st.isOpen,
        lastEvidence: st.lastEvidence
      };
    },
    version: '0.1.0'
  };
})();
