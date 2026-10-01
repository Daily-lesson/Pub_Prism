/**
 * Deterministic augmentation (docs/CONTRACTS.md §2.2): each base example yields up to
 * `AUGMENT_VARIANTS_PER_EXAMPLE` variants — generic filler prefixes/suffixes, casing, and light
 * phrase-level synonym swaps — with slot spans recomputed on the final string.
 *
 * Chunk-and-reassemble: the utterance is split into `{text | slot}` chunks from its own spans;
 * synonym swaps touch only `text` chunks, so a slot value is never rewritten; affixes shift the
 * spans by a constant; casing is applied last and each span's `value` is re-sliced from the
 * final string so the `utterance.slice(start, end) === value` invariant holds by construction.
 */

import type { GeneratedExample, SlotSpan } from './types';
import { deriveSeed, mulberry32, randInt } from './prng';

interface Chunk {
  kind: 'text' | 'slot';
  text: string;
  slotName?: string;
}

function toChunks(ex: GeneratedExample): Chunk[] {
  const slots = [...ex.slots].sort((a, b) => a.start - b.start);
  const chunks: Chunk[] = [];
  let cursor = 0;
  for (const s of slots) {
    if (s.start > cursor) chunks.push({ kind: 'text', text: ex.utterance.slice(cursor, s.start) });
    chunks.push({ kind: 'slot', text: ex.utterance.slice(s.start, s.end), slotName: s.name });
    cursor = s.end;
  }
  if (cursor < ex.utterance.length) chunks.push({ kind: 'text', text: ex.utterance.slice(cursor) });
  return chunks;
}

function fromChunks(chunks: readonly Chunk[], intent: string): GeneratedExample {
  let out = '';
  const slots: SlotSpan[] = [];
  for (const c of chunks) {
    if (c.kind === 'text') {
      out += c.text;
    } else {
      const start = out.length;
      out += c.text;
      slots.push({ name: c.slotName!, value: c.text, start, end: out.length });
    }
  }
  return { utterance: out, intent, slots };
}

/** Generic English phrase swaps — applied to text chunks only. */
export const PHRASE_SYNONYMS: ReadonlyArray<{ re: RegExp; alts: readonly string[] }> = [
  { re: /\bhow do i\b/g, alts: ['how do i', 'how can i', 'what is the best way to', 'could you show me how to'] },
  { re: /\bhow do you\b/g, alts: ['how do you', 'how can you', 'what is the way to'] },
  { re: /\bshow me\b/g, alts: ['show me', 'display', 'let me see', 'can you show me'] },
  { re: /\bhelp me\b/g, alts: ['help me', 'assist me', 'can you help me'] },
  { re: /\bset up\b/g, alts: ['set up', 'configure', 'get set up with'] },
  { re: /\bconnect\b/g, alts: ['connect', 'hook up', 'wire up', 'link'] },
  { re: /\bhow many\b/g, alts: ['how many', 'what number of'] },
  { re: /\bare there any\b/g, alts: ['are there any', 'is there'] },
  { re: /\bi need to\b/g, alts: ['i need to', 'i want to', 'i have to'] },
  { re: /\bgive me\b/g, alts: ['give me', 'show me', 'provide'] },
  { re: /\bcreate\b/g, alts: ['create', 'make', 'set up'] },
  { re: /\badd\b/g, alts: ['add', 'register', 'create'] },
  { re: /\bwatch\b/g, alts: ['watch', 'monitor', 'view'] },
  { re: /\bplease\b/g, alts: ['please', 'kindly'] },
];

function applySynonymVariant(text: string, rng: () => number): string {
  let out = text;
  for (const { re, alts } of PHRASE_SYNONYMS) {
    const tester = new RegExp(re.source, re.flags);
    if (tester.test(out)) {
      const choice = alts[randInt(rng, alts.length)];
      out = out.replace(new RegExp(re.source, re.flags), choice);
    }
  }
  return out;
}

export const PREFIXES: readonly string[] = [
  '',
  'hey, ',
  'please, ',
  'could you help, ',
  'quick question, ',
  'so, ',
  'i want to know: ',
  'can you help me: ',
];

export const SUFFIXES: readonly string[] = ['', ' please', ' thanks', '?', ' asap', ' for me'];

type Casing = 'asis' | 'capitalize' | 'upper';
const CASINGS: readonly Casing[] = ['asis', 'capitalize', 'upper'];

function applyCasing(s: string, casing: Casing): string {
  if (casing === 'asis') return s;
  if (casing === 'upper') return s.toUpperCase();
  const idx = s.search(/[a-zA-Z]/);
  if (idx === -1) return s;
  return s.slice(0, idx) + s[idx].toUpperCase() + s.slice(idx + 1);
}

interface VariantParams {
  synonymSeed: number;
  prefix: string;
  suffix: string;
  casing: Casing;
}

function buildVariant(ex: GeneratedExample, params: VariantParams): GeneratedExample | null {
  const chunks = toChunks(ex);
  const synRng = mulberry32(params.synonymSeed);
  const swapped: Chunk[] = chunks.map((c) => (c.kind === 'text' ? { ...c, text: applySynonymVariant(c.text, synRng) } : c));
  const reassembled = fromChunks(swapped, ex.intent);

  const withAffixes = params.prefix + reassembled.utterance + params.suffix;
  const shifted: SlotSpan[] = reassembled.slots.map((s) => ({
    ...s,
    start: s.start + params.prefix.length,
    end: s.end + params.prefix.length,
  }));

  const cased = applyCasing(withAffixes, params.casing);
  // Casing is length-preserving for ASCII but not for every script (e.g. "ß" -> "SS"); when it
  // moves offsets the variant is dropped rather than shipped with a broken span.
  if (cased.length !== withAffixes.length) return null;
  const finalSlots: SlotSpan[] = shifted.map((s) => ({ ...s, value: cased.slice(s.start, s.end) }));
  return { utterance: cased, intent: ex.intent, slots: finalSlots };
}

export const AUGMENT_VARIANTS_PER_EXAMPLE = 20;

/** Up to `AUGMENT_VARIANTS_PER_EXAMPLE` variants of one base example (variant 0 is the
 * untouched original). `index` is the example's stable position in its intent's pool. */
export function augmentExample(ex: GeneratedExample, seed: number, index: number): GeneratedExample[] {
  const baseSeed = deriveSeed(seed, `aug::${ex.intent}::${index}::${ex.utterance}`);
  const rng = mulberry32(baseSeed);

  const variants: GeneratedExample[] = [ex];
  const seen = new Set<string>([ex.utterance]);

  let attempts = 0;
  const maxAttempts = AUGMENT_VARIANTS_PER_EXAMPLE * 6;
  while (variants.length < AUGMENT_VARIANTS_PER_EXAMPLE && attempts < maxAttempts) {
    attempts++;
    const params: VariantParams = {
      prefix: PREFIXES[randInt(rng, PREFIXES.length)],
      suffix: SUFFIXES[randInt(rng, SUFFIXES.length)],
      casing: CASINGS[randInt(rng, CASINGS.length)],
      synonymSeed: randInt(rng, 0xffffffff),
    };
    const variant = buildVariant(ex, params);
    if (!variant || seen.has(variant.utterance)) continue;
    seen.add(variant.utterance);
    variants.push(variant);
  }
  return variants;
}

/** Augment every example in an intent's pool, preserving pool order. */
export function augmentPool(pool: readonly GeneratedExample[], seed: number): GeneratedExample[] {
  const out: GeneratedExample[] = [];
  pool.forEach((ex, index) => {
    out.push(...augmentExample(ex, seed, index));
  });
  return out;
}
