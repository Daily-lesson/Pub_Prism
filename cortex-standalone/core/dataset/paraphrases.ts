/**
 * Hand-authored paraphrases (docs/CONTRACTS.md §2.1/§2.3): intent-only utterances whose slot
 * spans are auto-detected — longest vocab label first, exact case-insensitive substring, one
 * match per slot, no overlapping spans.
 */

import type { NormalizedIntentDef, NormalizedRegistry } from '../registry/types';
import type { GeneratedExample, SlotSpan } from './types';
import { WORD_CHAR_RE } from '../tokenizer';

function isWordChar(ch: string | undefined): boolean {
  return ch !== undefined && WORD_CHAR_RE.test(ch);
}

export function autoDetectSlots(registry: NormalizedRegistry, utterance: string, slotNames: readonly string[]): SlotSpan[] {
  const claimed: Array<[number, number]> = [];
  const slots: SlotSpan[] = [];
  const lowered = utterance.toLowerCase();
  for (const name of slotNames) {
    const def = registry.slots[name];
    if (!def) continue;
    const vocab = [...def.vocab].sort((a, b) => b.label.length - a.label.length);
    let found = false;
    for (const entry of vocab) {
      const needle = entry.label.toLowerCase();
      if (!needle) continue;
      // Every occurrence is considered, and only a WHOLE-WORD one counts (§2.3): a two-letter
      // label such as "IT" must never tag "k[it]chen" or "wa[it]ing" — the model would learn
      // those spans and hand the host a fabricated slot value.
      for (let idx = lowered.indexOf(needle); idx !== -1; idx = lowered.indexOf(needle, idx + 1)) {
        const end = idx + needle.length;
        if (isWordChar(lowered[idx - 1]) || isWordChar(lowered[end])) continue;
        // The lowercased haystack must map 1:1 onto the original for offsets to be reusable.
        if (lowered.length !== utterance.length) {
          const raw = utterance.slice(idx, end);
          if (raw.toLowerCase() !== needle) continue;
        }
        const overlapsClaimed = claimed.some(([s, e]) => idx < e && end > s);
        if (overlapsClaimed) continue;
        claimed.push([idx, end]);
        slots.push({ name, value: utterance.slice(idx, end), start: idx, end });
        found = true;
        break;
      }
      if (found) break;
    }
  }
  return slots.sort((a, b) => a.start - b.start);
}

/** The intent's paraphrases as examples, slots auto-tagged. Duplicates within the intent are
 * collapsed (first occurrence kept). */
export function paraphraseExamples(registry: NormalizedRegistry, intent: NormalizedIntentDef): GeneratedExample[] {
  const out: GeneratedExample[] = [];
  const seen = new Set<string>();
  for (const utterance of intent.paraphrases) {
    if (!utterance.trim() || seen.has(utterance)) continue;
    seen.add(utterance);
    out.push({ utterance, intent: intent.id, slots: autoDetectSlots(registry, utterance, intent.slots) });
  }
  return out;
}
