import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as fs from 'node:fs';
import * as http from 'node:http';
import * as path from 'node:path';
import { learn } from '../learn/index';
import { toIntentId, toAppSlug } from '../learn/slugify';
import { DESTINATION_PHRASINGS, applyPhrasings } from '../learn/phrasings';
import { extractPage } from '../learn/extract';

const SITE = path.join(__dirname, 'fixtures', 'learn', 'site');
const SCHEMA = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'registry', 'schema.json'), 'utf8'));

// ── validation helper: core's validateRegistry when present, else schema.json's structural rules by hand ──
type Validator = (raw: unknown) => { ok: boolean; errors: string[] };
let validate: Validator;
let validatorSource = 'hand';

async function loadValidator(): Promise<Validator> {
  try {
    const core = await import('../core/index');
    if (typeof core.validateRegistry === 'function') {
      validatorSource = 'core.validateRegistry';
      return (raw) => { const r = core.validateRegistry(raw); return { ok: r.ok, errors: r.errors }; };
    }
  } catch {
    // core not present yet — fall through to the hand validator
  }
  return handValidate;
}

/** Enough of registry/schema.json (plus §1.3) to fail on a malformed draft. */
function handValidate(raw: unknown): { ok: boolean; errors: string[] } {
  const errors: string[] = [];
  const r = raw as Record<string, unknown>;
  if (!r || typeof r !== 'object') return { ok: false, errors: ['not an object'] };
  const allowedTop = Object.keys(SCHEMA.properties);
  for (const k of Object.keys(r)) if (!allowedTop.includes(k)) errors.push(`unknown top-level key ${k}`);
  if (r.registryVersion !== '1') errors.push('registryVersion must be "1"');
  const app = r.app as Record<string, unknown>;
  if (!app || typeof app.slug !== 'string' || !new RegExp(SCHEMA.properties.app.properties.slug.pattern).test(app.slug)) errors.push('bad app.slug');
  if (!app || typeof app.name !== 'string' || app.name.length < 1 || app.name.length > 120) errors.push('bad app.name');
  if (!r.slots || typeof r.slots !== 'object') errors.push('slots missing');
  const intents = r.intents as unknown[];
  if (!Array.isArray(intents) || intents.length < 1) errors.push('intents must be a non-empty array');
  const idRe = new RegExp(SCHEMA.properties.intents.items.properties.id.pattern);
  const seen = new Set<string>();
  const allowedIntentKeys = Object.keys(SCHEMA.properties.intents.items.properties);
  for (const i of intents ?? []) {
    const it = i as Record<string, unknown>;
    for (const k of Object.keys(it)) if (!allowedIntentKeys.includes(k)) errors.push(`unknown intent key ${k}`);
    if (typeof it.id !== 'string' || !idRe.test(it.id) || it.id.length > 64) errors.push(`bad id ${String(it.id)}`);
    if (seen.has(String(it.id))) errors.push(`duplicate id ${String(it.id)}`);
    seen.add(String(it.id));
    if (!['howto', 'status', 'meta'].includes(String(it.family))) errors.push(`bad family on ${String(it.id)}`);
    if (typeof it.label !== 'string' || it.label.length < 1 || it.label.length > 200) errors.push(`bad label on ${String(it.id)}`);
    if (typeof it.keywords !== 'string' || it.keywords.length > 2000) errors.push(`bad keywords on ${String(it.id)}`);
    for (const t of (it.templates as string[]) ?? []) {
      if (typeof t !== 'string' || t.length < 1 || t.length > 300) errors.push(`bad template on ${String(it.id)}`);
      if (/\{[^}]*\}/.test(t)) errors.push(`template placeholder without a declared slot on ${String(it.id)}`);
    }
    if (it.family === 'howto') {
      const a = it.answer as { steps?: unknown[]; links?: Array<Record<string, unknown>> } | undefined;
      if (!a || !Array.isArray(a.steps) || a.steps.length < 1) errors.push(`howto ${String(it.id)} needs answer.steps`);
      for (const l of a?.links ?? []) {
        if (typeof l.label !== 'string' || l.label.length < 1 || l.label.length > 80 || !('target' in l)) errors.push(`bad link on ${String(it.id)}`);
      }
      if ((a?.links ?? []).length > 8) errors.push(`too many links on ${String(it.id)}`);
    }
  }
  return { ok: errors.length === 0, errors };
}

// ── a tiny static server for the http tests (tests/support/static-server.js is used when present) ──
let server: http.Server | null = null;
let baseUrl = '';
let staticSource = 'in-test http.createServer';

async function startStatic(): Promise<void> {
  const supportPath = path.join(__dirname, 'support', 'static-server.js');
  if (fs.existsSync(supportPath)) {
    // tests/support/static-server.js exports `start({ root }) → { url, stop() }`
    const mod = await import(supportPath) as { start?: (o: { root: string }) => Promise<{ url: string; stop(): Promise<void> }> };
    if (typeof mod.start === 'function') {
      const s = await mod.start({ root: SITE });
      baseUrl = s.url.replace(/\/$/, '');
      server = { close: (cb?: () => void) => { void s.stop().then(() => cb?.()); } } as unknown as http.Server;
      staticSource = 'tests/support/static-server.js';
      return;
    }
  }
  server = http.createServer((req, res) => {
    const p = path.normalize(decodeURIComponent((req.url ?? '/').split('?')[0])).replace(/^(\.\.[/\\])+/, '');
    const file = path.join(SITE, p === '/' || p === '\\' ? 'index.html' : p);
    if (!file.startsWith(SITE) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.statusCode = 404; res.end('not found'); return; }
    res.setHeader('Content-Type', file.endsWith('.xml') ? 'application/xml' : 'text/html; charset=utf-8');
    res.end(fs.readFileSync(file));
  });
  await new Promise<void>((resolve) => server!.listen(0, '127.0.0.1', resolve));
  const addr = server.address() as { port: number };
  baseUrl = `http://127.0.0.1:${addr.port}`;
}

beforeAll(async () => { validate = await loadValidator(); await startStatic(); });
afterAll(async () => { await new Promise<void>((resolve) => (server ? server.close(() => resolve()) : resolve())); });

describe('learn — file input', () => {
  it('turns the six nav links into six howto intents and skips off-site / mailto / # links', async () => {
    const { registry, report } = await learn(path.join(SITE, 'index.html'));
    expect(report.pages).toBe(1);
    expect(report.intents).toBe(6);
    const ids = registry.intents.map((i) => i.id);
    expect(ids).toEqual(['catalogue', 'events_talks', 'membership', 'opening_hours', 'contact_us', 'study_rooms']);
    expect(registry.intents.map((i) => i.label)).toEqual(['Catalogue', 'Events & Talks', 'Membership', 'Opening hours', 'Contact us', 'Study rooms']);
    expect(registry.intents.map((i) => i.answer.links[0].target.href)).toEqual([
      'catalogue.html', 'events.html', 'membership.html', 'hours.html', 'contact.html', 'rooms.html',
    ]);
    const labels = registry.intents.map((i) => i.label);
    expect(labels).not.toContain('Partner libraries');
    expect(labels).not.toContain('Email the desk');
    expect(labels).not.toContain('Back to top');
    // main-content link is not in a landmark, so it is not a destination
    expect(labels).not.toContain('support us');
    // a <script> that contains link-looking text never yields a destination
    expect(JSON.stringify(registry)).not.toContain('tracker.html');
    expect(registry.app).toEqual({ slug: 'rivermead-community-library', name: 'Rivermead Community Library' });
    expect(registry.slots).toEqual({});
    expect(registry.heldout).toEqual([]);
  });

  it('shapes each intent per §8.2', async () => {
    const { registry } = await learn(path.join(SITE, 'index.html'));
    const i = registry.intents.find((x) => x.id === 'events_talks')!;
    expect(i.family).toBe('howto');
    expect(i.slots).toEqual([]);
    expect(i.paraphrases).toEqual([]);
    expect(i.keywords.split(' ')).toEqual(expect.arrayContaining(['events', 'talks']));
    expect(i.answer.steps).toEqual(['Open Events & Talks from the site navigation.']);
    expect(i.answer.links).toEqual([{ label: 'Events & Talks', target: { href: 'events.html' } }]);
    for (const p of DESTINATION_PHRASINGS) expect(i.templates).toContain(p.replace('{x}', 'events & talks'));
    expect(i.templates.length).toBe(DESTINATION_PHRASINGS.length);
    for (const t of i.templates) { expect(t).toBe(t.toLowerCase()); expect(t).not.toMatch(/[{}]/); }
  });

  it('crawl: keywords include heading words from the linked page; missing pages are warned, not fatal', async () => {
    const { registry, report } = await learn(path.join(SITE, 'index.html'), { crawl: true, depth: 1 });
    const cat = registry.intents.find((x) => x.id === 'catalogue')!;
    const kw = cat.keywords.split(' ');
    expect(kw).toEqual(expect.arrayContaining(['catalogue', 'search', 'books', 'audiobooks', 'magazines', 'reserve', 'title', 'collection']));
    expect(kw).not.toContain('the');
    expect(kw).not.toContain('and');
    expect(new Set(kw).size).toBe(kw.length);
    // without crawl those heading words are absent
    const plain = await learn(path.join(SITE, 'index.html'));
    expect(plain.registry.intents.find((x) => x.id === 'catalogue')!.keywords.split(' ')).not.toContain('audiobooks');
    // hours.html / rooms.html do not exist in the fixture
    expect(report.warnings.some((w) => w.includes('hours.html'))).toBe(true);
    expect(report.warnings.some((w) => w.includes('rooms.html'))).toBe(true);
    expect(report.pages).toBe(1 + 4);
    expect(report.intents).toBe(6);
  });

  it('crawl depth 2 discovers destinations linked from the destination pages', async () => {
    const { registry } = await learn(path.join(SITE, 'index.html'), { crawl: true, depth: 2 });
    const ids = registry.intents.map((i) => i.id);
    expect(ids).toContain('renew_your_card');
    expect(ids).not.toContain('home'); // links back to the root are not destinations
  });

  it('sitemap input → one intent per URL, labelled by the page title', async () => {
    const { registry, report } = await learn(path.join(SITE, 'sitemap.xml'));
    expect(report.intents).toBe(5);
    expect(registry.intents.map((i) => i.label)).toEqual([
      'Rivermead Community Library',
      'Search the catalogue — Rivermead Community Library',
      'Events and talks — Rivermead Community Library',
      'Join the library — Rivermead Community Library',
      'Contact us — Rivermead Community Library',
    ]);
    expect(registry.intents[1].id).toBe('search_the_catalogue_rivermead_community_library');
    expect(registry.intents[1].answer.links[0].target.href).toMatch(/^file:.*catalogue\.html$/);
    expect(registry.intents[1].keywords.split(' ')).toEqual(expect.arrayContaining(['audiobooks', 'reserve']));
  });

  // Review fix (§8.3): destinations sharing a label used to become events / events_2 / events_3
  // with byte-identical templates — labels no model can separate. They now merge into ONE
  // intent carrying every href as a link, with a warning naming the collision.
  it('same-label destinations merge into one intent with several links (+ a warning); same href with a different fragment is deduped; digit-leading and accented labels are folded', async () => {
    const { registry, report } = await learn(path.join(SITE, 'duplicates.html'));
    const warnings = report.warnings;
    const ids = registry.intents.map((i) => i.id);
    expect(ids).toEqual(['events', 'go_2024_highlights', 'cafe_shop']);
    const events = registry.intents.find((i) => i.id === 'events')!;
    expect(events.answer.links.map((l) => l.target.href)).toEqual(['events.html', 'events-archive.html', 'events-2024.html']);
    expect(warnings.some((w) => /3 destinations share the label "Events"/.test(w))).toBe(true);
    expect(registry.intents.find((i) => i.id === 'cafe_shop')!.label).toBe('Café & shop');
  });

  it('a nav link labelled like a built-in intent never claims the reserved id, and file: links outside the root directory are skipped (review fixes)', async () => {
    const { registry, report } = await learn(path.join(SITE, 'reserved.html'), { crawl: true, depth: 1 });
    const warnings = report.warnings;
    const ids = registry.intents.map((i) => i.id);
    expect(ids).toContain('greeting_2');
    expect(ids).toContain('out_of_domain_2');
    expect(ids).not.toContain('greeting');
    expect(ids).not.toContain('out_of_domain');
    // /etc/hostname and ../escape.html resolve outside tests/fixtures/learn/site → never read
    expect(registry.intents.some((i) => i.answer.links.some((l) => /etc\/hostname|escape\.html/.test(l.target.href)))).toBe(false);
    expect(warnings.filter((w) => /outside the root document's directory/.test(w)).length).toBe(2);
    // the draft still validates with the built-ins appended
    expect(validate(registry).errors).toEqual([]);
  });

  it('warns about the no-landmark fallback and still finds the links', async () => {
    const { registry, report } = await learn(path.join(SITE, 'nolandmark.html'));
    expect(report.warnings.some((w) => /no <nav>\/<header>\/<aside> landmark/.test(w) && w.includes('nolandmark.html'))).toBe(true);
    expect(registry.intents.map((i) => i.id)).toEqual(['crime_fiction_circle', 'poetry_corner']);
    expect(registry.intents[0].answer.links[0].target.href).toBe('groups/crime.html');
    // and the landmark-bearing page produced no such warning
    const ok = await learn(path.join(SITE, 'index.html'));
    expect(ok.report.warnings.some((w) => /landmark/.test(w))).toBe(false);
  });

  it('honours name / slug overrides and sameOriginOnly:false', async () => {
    const { registry } = await learn(path.join(SITE, 'index.html'), { name: 'Valley Library!', slug: 'Valley Library!', sameOriginOnly: false });
    expect(registry.app).toEqual({ slug: 'valley-library', name: 'Valley Library!' });
    expect(registry.intents.map((i) => i.label)).toContain('Partner libraries');
    expect(registry.intents.find((i) => i.label === 'Partner libraries')!.answer.links[0].target.href).toBe('https://elsewhere.example/partner-libraries');
  });

  it('output validates as a registry', async () => {
    const { registry } = await learn(path.join(SITE, 'index.html'), { crawl: true, depth: 1 });
    const r = validate(registry);
    expect(r.errors).toEqual([]);
    expect(r.ok).toBe(true);
    // also pin the hand validator regardless of which one loaded, so a core absence can't make this vacuous
    expect(handValidate(registry)).toEqual({ ok: true, errors: [] });
    // eslint-disable-next-line no-console
    console.log(`[learn.test] registry validated via ${validatorSource}; http fixtures served by ${staticSource}`);
  });
});

describe('learn — http input', () => {
  it('fetches the root page over http, resolves hrefs to absolute same-origin URLs, and crawls', async () => {
    const { registry, report } = await learn(`${baseUrl}/index.html`, { crawl: true, depth: 1 });
    expect(report.intents).toBe(6);
    expect(registry.intents[0].answer.links[0].target.href).toBe(`${baseUrl}/catalogue.html`);
    expect(registry.intents[0].keywords.split(' ')).toContain('audiobooks');
    expect(report.warnings.some((w) => w.includes('rooms.html') && w.includes('HTTP 404'))).toBe(true);
    expect(registry.intents.map((i) => i.label)).not.toContain('Partner libraries');
  });

  it('sitemap over http → one intent per URL with absolute hrefs', async () => {
    const { registry, report } = await learn(`${baseUrl}/sitemap.xml`);
    expect(report.intents).toBe(5);
    expect(registry.intents[1].answer.links[0].target.href).toBe(`${baseUrl}/catalogue.html`);
  });

  it('defaults the app name to the host when the root page has no title', async () => {
    const { registry } = await learn(`${baseUrl}/nolandmark.html`, { name: '' });
    expect(registry.app.name).toBe('Reading groups — Rivermead Community Library');
  });
});

describe('learn helpers', () => {
  it('toIntentId folds, dedupes and prefixes', () => {
    const taken = new Set<string>();
    expect(toIntentId('Events & Talks', taken)).toBe('events_talks');
    expect(toIntentId('Events & Talks', taken)).toBe('events_talks_2');
    expect(toIntentId('events---talks!', taken)).toBe('events_talks_3');
    expect(toIntentId('2024 highlights', taken)).toBe('go_2024_highlights');
    expect(toIntentId('Café', taken)).toBe('cafe');
    expect(toIntentId('   ', taken)).toBe('destination');
    const long1 = toIntentId('x'.repeat(200), taken);
    const long2 = toIntentId('x'.repeat(200), taken);
    expect(long1).toMatch(/^[a-z][a-z0-9_]*$/);
    expect(long1.length).toBe(64);
    expect(long2.length).toBeLessThanOrEqual(64);
    expect(long2).toMatch(/_2$/);
    for (const id of taken) expect(id).toMatch(/^[a-z][a-z0-9_]*$/);
  });

  it('toAppSlug produces ^[a-z0-9][a-z0-9-]*$', () => {
    expect(toAppSlug('Rivermead Community Library')).toBe('rivermead-community-library');
    expect(toAppSlug('--Ünïcode-- Name!!')).toBe('unicode-name');
    expect(toAppSlug('')).toBe('site');
    expect(toAppSlug('123 go')).toBe('123-go');
  });

  it('applyPhrasings lowercases and strips braces', () => {
    const t = applyPhrasings('Study {Rooms}');
    expect(t).toContain('open study rooms');
    for (const x of t) expect(x).not.toMatch(/[{}]/);
    expect(DESTINATION_PHRASINGS.length).toBeGreaterThanOrEqual(14);
  });

  it('extractPage handles entities, image-only links and unquoted hrefs', () => {
    const page = extractPage(`<html><head><title>T &amp; U</title></head><body>
      <nav><a href=plain.html><img alt="Plain"></a><a href="q.html?a=1&amp;b=2">Q &lt;x&gt;</a></nav>
      <h1>One</h1><h2>Two <em>em</em></h2></body></html>`);
    expect(page.title).toBe('T & U');
    expect(page.landmarkLinks).toEqual([{ href: 'plain.html', text: 'Plain' }, { href: 'q.html?a=1&b=2', text: 'Q <x>' }]);
    expect(page.h1).toEqual(['One']);
    expect(page.h2).toEqual(['Two em']);
    expect(page.hasLandmarks).toBe(true);
  });
});
