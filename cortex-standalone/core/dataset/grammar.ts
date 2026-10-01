/**
 * Template expansion (docs/CONTRACTS.md §2.1). Templates come from `intent.templates`; slot
 * fills come from `registry.slots[name].vocab` (labels are inserted, ids are what a tagged span
 * resolves to). Spans are `[start, end)` into the final utterance.
 */

import type { NormalizedIntentDef, NormalizedRegistry, VocabEntry } from '../registry/types';
import { deriveSeed, mulberry32, shuffle } from './prng';
import { augmentPool } from './augment';
import type { GeneratedExample, SlotSpan } from './types';

export const VARIANTS_PER_SLOTTED_TEMPLATE = 4;

const PLACEHOLDER_RE = /\{(\w+)\}/g;

export function vocabForSlot(registry: NormalizedRegistry, name: string): readonly VocabEntry[] {
  const def = registry.slots[name];
  if (!def) throw new Error(`no vocab for slot "${name}" (not declared in registry.slots)`);
  return def.vocab;
}

export function extractPlaceholders(pattern: string): string[] {
  const names: string[] = [];
  let m: RegExpExecArray | null;
  const re = new RegExp(PLACEHOLDER_RE.source, 'g');
  while ((m = re.exec(pattern))) names.push(m[1]);
  return names;
}

/** Fill `{slot}` placeholders from `fills`, returning the utterance + a span per fill. */
export function fillTemplate(pattern: string, fills: Record<string, string>): { utterance: string; slots: SlotSpan[] } {
  const slots: SlotSpan[] = [];
  let out = '';
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  const re = new RegExp(PLACEHOLDER_RE.source, 'g');
  while ((m = re.exec(pattern))) {
    out += pattern.slice(lastIndex, m.index);
    const slotName = m[1];
    const value = fills[slotName];
    if (value === undefined) throw new Error(`no fill provided for slot {${slotName}} in template "${pattern}"`);
    const start = out.length;
    out += value;
    const end = out.length;
    slots.push({ name: slotName, value, start, end });
    lastIndex = re.lastIndex;
  }
  out += pattern.slice(lastIndex);
  return { utterance: out, slots };
}

/** Base examples for one intent (before augmentation): 1 per unslotted template,
 * `VARIANTS_PER_SLOTTED_TEMPLATE` per slotted template, fills drawn from a seeded shuffled
 * queue of vocab labels per `(intent, template, slot)`. */
export function generateBaseExamplesForIntent(registry: NormalizedRegistry, intent: NormalizedIntentDef, seed: number): GeneratedExample[] {
  const examples: GeneratedExample[] = [];

  for (const pattern of intent.templates) {
    const placeholders = extractPlaceholders(pattern);
    if (placeholders.length === 0) {
      const { utterance, slots } = fillTemplate(pattern, {});
      examples.push({ utterance, intent: intent.id, slots });
      continue;
    }

    const queues = new Map<string, string[]>();
    for (const slotName of new Set(placeholders)) {
      if (!intent.slots.includes(slotName)) {
        throw new Error(`intent "${intent.id}": template "${pattern}" uses {${slotName}}, which the intent does not declare`);
      }
      const vocab = vocabForSlot(registry, slotName);
      if (vocab.length === 0) throw new Error(`intent "${intent.id}": slot "${slotName}" has an empty vocab and cannot fill "${pattern}"`);
      const rng = mulberry32(deriveSeed(seed, `${intent.id}::${pattern}::${slotName}`));
      queues.set(
        slotName,
        shuffle(vocab, rng).map((v) => v.label),
      );
    }

    for (let k = 0; k < VARIANTS_PER_SLOTTED_TEMPLATE; k++) {
      const fills: Record<string, string> = {};
      for (const slotName of placeholders) {
        const queue = queues.get(slotName)!;
        fills[slotName] = queue[k % queue.length];
      }
      const { utterance, slots } = fillTemplate(pattern, fills);
      examples.push({ utterance, intent: intent.id, slots });
    }
  }

  return examples;
}

/** Template expansion + augmentation for one intent. Same seed ⇒ byte-identical output. */
export function generateFromTemplates(registry: NormalizedRegistry, intent: NormalizedIntentDef, seed: number): GeneratedExample[] {
  const base = generateBaseExamplesForIntent(registry, intent, seed);
  return augmentPool(base, seed);
}
