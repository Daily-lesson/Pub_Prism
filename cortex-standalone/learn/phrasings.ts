/**
 * Destination phrasings for `learn` (CONTRACTS §8.2).
 *
 * The first eight are the built-in phrasings the contract names; the rest are
 * extra generic ways of asking to be taken somewhere. They are UNSLOTTED
 * templates: `{x}` is replaced by the destination label at draft time and the
 * emitted template carries no `{slot}` placeholder at all.
 */

export const DESTINATION_PHRASINGS: readonly string[] = [
  'how do i get to {x}',
  'where is {x}',
  'open {x}',
  'take me to {x}',
  'show me {x}',
  'i want to see {x}',
  'navigate to {x}',
  'find {x}',
  // extra generic phrasings
  'how can i reach {x}',
  'where do i find {x}',
  'go to {x}',
  'get me to {x}',
  "i'm looking for {x}",
  "where's {x}",
  'how do i open {x}',
  'can you show me {x}',
];

/** Lowercased templates for one label; braces are stripped from the label so
 * the result can never be mistaken for a slotted template. */
export function applyPhrasings(label: string): string[] {
  const safe = label.replace(/[{}]/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
  const out: string[] = [];
  const seen = new Set<string>();
  for (const p of DESTINATION_PHRASINGS) {
    const t = p.replace('{x}', safe).slice(0, 300);
    if (!seen.has(t)) { seen.add(t); out.push(t); }
  }
  return out;
}
