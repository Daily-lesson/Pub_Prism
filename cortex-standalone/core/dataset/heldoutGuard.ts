/**
 * The held-out guard (docs/CONTRACTS.md §2.6): nothing that flows into a compiled split may
 * case-insensitively equal a `heldout[]` utterance. Called by `compileDataset` as a hard,
 * fail-loud runtime check.
 */

import type { HeldoutExample } from '../registry/types';

export class HeldoutOverlapError extends Error {
  constructor(public readonly offenders: string[]) {
    super(
      `dataset compile ABORTED: ${offenders.length} compiled utterance(s) exactly match a held-out ` +
        `utterance (case-insensitive) — the held-out set must never be trained on. First offenders: ${offenders
          .slice(0, 5)
          .map((h) => JSON.stringify(h))
          .join(', ')}`,
    );
    this.name = 'HeldoutOverlapError';
  }
}

export function heldoutUtteranceSet(heldout: readonly HeldoutExample[]): Set<string> {
  return new Set(heldout.map((h) => h.utterance.toLowerCase()));
}

export function assertNoHeldoutOverlap(utterances: readonly string[], heldout: Set<string>): void {
  const hits: string[] = [];
  for (const u of utterances) if (heldout.has(u.toLowerCase())) hits.push(u);
  if (hits.length > 0) throw new HeldoutOverlapError(hits);
}
