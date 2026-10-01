/**
 * `learn` — draft a registry from a site (CONTRACTS §8).
 *
 * DEV-TIME CLI ONLY. Fetching here is a developer convenience: there is NO
 * SSRF guard, no redirect policy, no size cap beyond what `fetch` gives us, and
 * the input is assumed to be a site the developer chose. Never wire this into
 * a runtime request path or point it at untrusted input.
 *
 * Input is an http(s) URL, a local `.html` file path, or a `sitemap.xml` (URL
 * or file). The output is a DRAFT registry: it validates and compiles as-is,
 * but a human is expected to edit labels / keywords / steps before training.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import { extractPage, extractSitemapLocs, type ExtractedLink, type ExtractedPage } from './extract';
import { applyPhrasings } from './phrasings';
import { toAppSlug, toIntentId } from './slugify';

export interface LearnOptions {
  /** Fetch each destination page and harvest its title/h1/h2 into keywords. */
  crawl?: boolean;
  /** How many link hops to follow while crawling (default 1). */
  depth?: number;
  /** `app.name` override (default: root page title, else host name). */
  name?: string;
  /** `app.slug` override (default: slug of the name). */
  slug?: string;
  /** Skip links whose origin differs from the input's (default true). */
  sameOriginOnly?: boolean;
  /** Per-request timeout in ms for URL inputs (default 10 000). */
  timeoutMs?: number;
}

export interface LearnReport {
  pages: number;
  intents: number;
  warnings: string[];
}

/** Minimal shape of the draft registry this module emits (validates against registry/schema.json). */
export interface DraftLink { label: string; target: { href: string } }
export interface DraftIntent {
  id: string;
  family: 'howto';
  label: string;
  slots: string[];
  keywords: string;
  templates: string[];
  paraphrases: string[];
  answer: { steps: string[]; links: DraftLink[] };
}
export interface DraftRegistry {
  registryVersion: '1';
  app: { slug: string; name: string };
  slots: Record<string, never>;
  intents: DraftIntent[];
  heldout: never[];
}

export interface LearnResult { registry: DraftRegistry; report: LearnReport }

/** Local filler list for keyword bags (not imported from core on purpose — learn stays dependency-free). */
const STOPWORDS = new Set((
  'the a an and or of to in on for with is are was were be been do does did how what where when which who whom why ' +
  'can could would should will shall may might must i me my we our you your it its this that these those there here ' +
  'from by at as into onto than then so if not no yes please just about over under up down out off again more most some any all'
).split(' '));

const MAX_LABEL = 200;
const MAX_KEYWORDS = 2000;
const MAX_PAGES = 500;
const DEFAULT_TIMEOUT_MS = 10_000;

interface Destination {
  key: string;        // normalized href — the dedupe key
  href: string;       // href as emitted into `answer.links[0].target.href`
  url: URL | null;    // resolved URL when fetchable
  label: string;
  words: Set<string>; // keyword tokens gathered so far
  depth: number;      // hops from the root
}

class Warnings {
  readonly list: string[] = [];
  add(msg: string): void { if (!this.list.includes(msg)) this.list.push(msg); }
}

// ── input resolution ────────────────────────────────────────────────────────

function toInputUrl(input: string): URL {
  if (/^https?:\/\//i.test(input)) return new URL(input);
  if (/^file:\/\//i.test(input)) return new URL(input);
  return pathToFileURL(path.resolve(input));
}

async function readSource(url: URL, timeoutMs: number): Promise<string> {
  if (url.protocol === 'file:') {
    return fs.promises.readFile(url, 'utf8');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error(`unsupported scheme ${url.protocol}`);
  }
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), redirect: 'follow' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.text();
}

function looksLikeSitemap(url: URL, body: string): boolean {
  if (/sitemap[^/]*\.xml$/i.test(url.pathname)) return true;
  const head = body.slice(0, 512);
  return /<urlset\b|<sitemapindex\b/i.test(head);
}

// ── keyword tokenization ────────────────────────────────────────────────────

export function keywordTokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length >= 3 && !STOPWORDS.has(t));
}

function addWords(set: Set<string>, text: string): void {
  for (const t of keywordTokens(text)) set.add(t);
}

function keywordString(words: Set<string>): string {
  let out = '';
  for (const w of words) {
    const next = out === '' ? w : `${out} ${w}`;
    if (next.length > MAX_KEYWORDS) break;
    out = next;
  }
  return out;
}

// ── link filtering ──────────────────────────────────────────────────────────

const SKIP_SCHEMES = /^(mailto|javascript|tel|sms|data|ftp):/i;

function normalizeHref(u: URL): string {
  const copy = new URL(u.href);
  copy.hash = '';
  let s = copy.href;
  if (copy.pathname !== '/' && s.endsWith('/')) s = s.slice(0, -1);
  return s;
}

function sameOrigin(a: URL, b: URL): boolean {
  if (a.protocol === 'file:' && b.protocol === 'file:') return true;
  return a.origin === b.origin;
}

/** Decide whether a link is a destination; returns the resolved URL or a reason to skip it. */
function resolveLink(link: ExtractedLink, base: URL, sameOriginOnly: boolean): { url: URL } | { skip: string } {
  const raw = link.href.trim();
  if (raw === '' || raw.startsWith('#')) return { skip: 'fragment' };
  if (SKIP_SCHEMES.test(raw)) return { skip: 'scheme' };
  let url: URL;
  try { url = new URL(raw, base); } catch { return { skip: 'unparseable' }; }
  if (url.protocol !== 'http:' && url.protocol !== 'https:' && url.protocol !== 'file:') return { skip: 'scheme' };
  if (sameOriginOnly && !sameOrigin(url, base)) return { skip: 'off-origin' };
  if (normalizeHref(url) === normalizeHref(base)) return { skip: 'self' };
  return { url };
}

function cleanLabel(text: string): string {
  return text.replace(/\s+/g, ' ').trim().slice(0, MAX_LABEL);
}

function emittedHref(url: URL, root: URL): string {
  // For file inputs, emit a path relative to the root document's directory so the draft
  // stays portable; URL inputs keep the absolute href.
  if (url.protocol === 'file:' && root.protocol === 'file:') {
    const rootDir = path.dirname(decodeURIComponent(root.pathname));
    const rel = path.relative(rootDir, decodeURIComponent(url.pathname));
    return rel.split(path.sep).join('/') + (url.search || '');
  }
  return normalizeHref(url);
}

// ── page handling ───────────────────────────────────────────────────────────

interface Fetched { page: ExtractedPage; url: URL }

async function fetchPage(url: URL, timeoutMs: number, warnings: Warnings, counter: { pages: number }): Promise<Fetched | null> {
  try {
    const body = await readSource(url, timeoutMs);
    counter.pages += 1;
    return { page: extractPage(body), url };
  } catch (err) {
    warnings.add(`could not fetch ${url.href}: ${err instanceof Error ? err.message : String(err)}`);
    return null;
  }
}

function harvestHeadings(dest: Destination, page: ExtractedPage): void {
  addWords(dest.words, page.title);
  for (const h of page.h1) addWords(dest.words, h);
  for (const h of page.h2) addWords(dest.words, h);
}

function pickLinks(page: ExtractedPage, from: URL, warnings: Warnings): ExtractedLink[] {
  if (page.hasLandmarks) return page.landmarkLinks;
  warnings.add(`no <nav>/<header>/<aside> landmark on ${from.href}; fell back to every link on the page`);
  return page.allLinks;
}

// ── the draft ───────────────────────────────────────────────────────────────

/** Built-in meta intents own these ids (CONTRACTS §1.4); a nav link labelled "Greeting" must
 * dedupe to `greeting_2`, never claim the built-in id and fail validation. */
const RESERVED_INTENT_IDS = ['greeting', 'out_of_domain'];

function buildRegistry(dests: Destination[], name: string, slug: string, warnings: Warnings): DraftRegistry {
  const taken = new Set<string>(RESERVED_INTENT_IDS);
  // Destinations sharing a label (case-insensitively) collapse into ONE intent with several
  // links: two intents with byte-identical templates can never be told apart by any model,
  // and would fail the in-distribution floor at training time.
  const byLabel = new Map<string, Destination[]>();
  for (const d of dests) {
    const key = d.label.toLowerCase();
    const group = byLabel.get(key);
    if (group) group.push(d);
    else byLabel.set(key, [d]);
  }
  const intents: DraftIntent[] = [...byLabel.values()].map((group) => {
    const d = group[0];
    if (group.length > 1) {
      warnings.add(`${group.length} destinations share the label "${d.label}" — merged into one intent with ${group.length} links (${group.map((g) => g.href).join(', ')})`);
    }
    const id = toIntentId(d.label, taken);
    const words = new Set<string>();
    addWords(words, d.label);
    for (const g of group) for (const w of g.words) words.add(w);
    return {
      id,
      family: 'howto',
      label: d.label,
      slots: [],
      keywords: keywordString(words),
      templates: applyPhrasings(d.label),
      paraphrases: [],
      answer: {
        steps: [`Open ${d.label} from the site navigation.`],
        links: group.map((g) => ({ label: d.label.slice(0, 80), target: { href: g.href } })),
      },
    };
  });
  return { registryVersion: '1', app: { slug, name }, slots: {}, intents, heldout: [] };
}

/** A `file:` root may only lead to files under its own directory — a saved page's
 * `<a href="/etc/passwd">` is never read. Non-file roots are unaffected. */
function withinRootDir(url: URL, root: URL): boolean {
  // Only file→file links are constrained; an http link from a saved page is governed by
  // sameOriginOnly, exactly as before.
  if (root.protocol !== 'file:' || url.protocol !== 'file:') return true;
  const rootDir = path.resolve(path.dirname(decodeURIComponent(root.pathname)));
  const target = path.resolve(decodeURIComponent(url.pathname));
  return target === rootDir || target.startsWith(rootDir + path.sep);
}

export async function learn(input: string, opts: LearnOptions = {}): Promise<LearnResult> {
  const warnings = new Warnings();
  const counter = { pages: 0 };
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const sameOriginOnly = opts.sameOriginOnly !== false;
  const depth = opts.crawl ? Math.max(1, Math.floor(opts.depth ?? 1)) : 0;

  const root = toInputUrl(input);
  const rootBody = await readSource(root, timeoutMs);
  counter.pages += 1;

  const dests: Destination[] = [];
  const byKey = new Map<string, Destination>();
  let rootTitle = '';

  const addDestination = (url: URL, label: string, hop: number): Destination | null => {
    const key = normalizeHref(url);
    const existing = byKey.get(key);
    if (existing) return existing;
    if (label === '') { warnings.add(`skipped a link with no text: ${url.href}`); return null; }
    if (dests.length >= MAX_PAGES) { warnings.add(`destination cap of ${MAX_PAGES} reached; further links ignored`); return null; }
    const d: Destination = { key, href: emittedHref(url, root), url, label, words: new Set(), depth: hop };
    dests.push(d);
    byKey.set(key, d);
    return d;
  };

  if (looksLikeSitemap(root, rootBody)) {
    // Sitemap mode: one destination per <loc>, labelled by the page title.
    let { locs, isIndex } = extractSitemapLocs(rootBody);
    if (isIndex) {
      warnings.add('input is a sitemap index; following each child sitemap one level deep');
      const childLocs: string[] = [];
      for (const loc of locs) {
        try {
          const childUrl = new URL(loc, root);
          const body = await readSource(childUrl, timeoutMs);
          counter.pages += 1;
          childLocs.push(...extractSitemapLocs(body).locs.map((l) => new URL(l, childUrl).href));
        } catch (err) {
          warnings.add(`could not fetch child sitemap ${loc}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }
      locs = childLocs;
    }
    if (locs.length === 0) warnings.add('sitemap lists no <loc> entries');
    for (const loc of locs) {
      let url: URL;
      try { url = new URL(loc, root); } catch { warnings.add(`unparseable sitemap entry: ${loc}`); continue; }
      if (sameOriginOnly && !sameOrigin(url, root)) { warnings.add(`skipped off-origin sitemap entry: ${loc}`); continue; }
      const fetched = await fetchPage(url, timeoutMs, warnings, counter);
      if (!fetched) continue;
      const label = cleanLabel(fetched.page.title || fetched.page.h1[0] || path.posix.basename(url.pathname) || url.hostname);
      const d = addDestination(url, label, 1);
      if (!d) continue;
      d.href = normalizeHref(url); // sitemap entries are absolute addresses by definition
      harvestHeadings(d, fetched.page);
      if (rootTitle === '' && fetched.page.title) rootTitle = fetched.page.title;
    }
  } else {
    const rootPage = extractPage(rootBody);
    rootTitle = rootPage.title;
    const links = pickLinks(rootPage, root, warnings);
    if (links.length === 0) warnings.add('no links found on the input page');
    for (const link of links) {
      const r = resolveLink(link, root, sameOriginOnly);
      if ('url' in r && !withinRootDir(r.url, root)) { warnings.add(`skipped link outside the root document's directory: ${r.url.href}`); continue; }
      if ('skip' in r) continue;
      addDestination(r.url, cleanLabel(link.text), 1);
    }

    // Crawl: breadth-first up to `depth` hops, harvesting headings into each destination's keywords.
    let frontier = dests.slice();
    for (let hop = 1; hop <= depth && frontier.length > 0; hop += 1) {
      const next: Destination[] = [];
      for (const d of frontier) {
        if (!d.url) continue;
        const fetched = await fetchPage(d.url, timeoutMs, warnings, counter);
        if (!fetched) continue;
        harvestHeadings(d, fetched.page);
        if (hop < depth) {
          for (const link of pickLinks(fetched.page, d.url, warnings)) {
            const r = resolveLink(link, d.url, sameOriginOnly);
            if ('url' in r && !withinRootDir(r.url, root)) { warnings.add(`skipped link outside the root document's directory: ${r.url.href}`); continue; }
            if ('skip' in r) continue;
            if (sameOriginOnly && !sameOrigin(r.url, root)) continue;
            if (normalizeHref(r.url) === normalizeHref(root)) continue;
            const before = dests.length;
            const nd = addDestination(r.url, cleanLabel(link.text), hop + 1);
            if (nd && dests.length > before) next.push(nd);
          }
        }
      }
      frontier = next;
    }
  }

  const name = (opts.name ?? '').trim() || cleanLabel(rootTitle) || (root.protocol === 'file:' ? path.basename(decodeURIComponent(root.pathname)) : root.hostname) || 'site';
  const slug = opts.slug ? toAppSlug(opts.slug) : toAppSlug(name);
  const registry = buildRegistry(dests, name.slice(0, 120), slug, warnings);
  if (registry.intents.length === 0) warnings.add('no destinations found — the draft registry has no intents and will not validate');

  return { registry, report: { pages: counter.pages, intents: registry.intents.length, warnings: warnings.list } };
}
