/**
 * The answer ladder (docs/CONTRACTS.md §5.1): cortex (model ≥ threshold) → keyword → fallback.
 * Works with no classifier at all (keyword-only mode).
 */

import { matchKeywords, type KeywordBag } from './keyword';
import type { ClassifiedSlot, ClassifyResult } from './session';
import type { LadderRung } from './evidence';

export type ClassifyFn = (utterance: string) => Promise<ClassifyResult>;

export interface LadderOptions {
  /** Absent ⇒ keyword-only mode. A classifier that throws is treated as absent for that query. */
  classify?: ClassifyFn;
  bags: readonly KeywordBag[];
  /** Default 0.6. */
  threshold?: number;
  /** Slot detection for a keyword-rung answer (§5.5): the registry's vocab labels matched as
   * whole words in the query. Absent ⇒ the keyword rung reports no slots. */
  detectSlots?: (query: string, intent: string) => ClassifiedSlot[];
}

export interface LadderResolution {
  rung: LadderRung;
  intent: string | null;
  /** The model's confidence in its top intent, or null when no model ran. */
  intentConf: number | null;
  slots: ClassifiedSlot[];
  /** The model's own top intent (even when a lower rung answered), or null when no model ran. */
  modelIntent: string | null;
  /** Set when the classifier threw for this query. */
  classifyError?: string;
}

export const DEFAULT_CONFIDENCE_THRESHOLD = 0.6;

export async function resolveLadder(query: string, opts: LadderOptions): Promise<LadderResolution> {
  const threshold = opts.threshold ?? DEFAULT_CONFIDENCE_THRESHOLD;
  let result: ClassifyResult | null = null;
  let classifyError: string | undefined;
  if (opts.classify) {
    try {
      result = await opts.classify(query);
    } catch (e) {
      classifyError = (e as Error).message;
      result = null;
    }
  }
  const base = {
    intentConf: result ? result.intentConf : null,
    modelIntent: result ? result.intent : null,
    ...(classifyError !== undefined ? { classifyError } : {}),
  };

  // `out_of_domain` above threshold answers on the cortex rung too (§5.1).
  if (result && result.intentConf >= threshold) {
    return { rung: 'cortex', intent: result.intent, slots: result.slots, ...base };
  }

  const kw = matchKeywords(query, opts.bags);
  if (kw) return { rung: 'keyword', intent: kw.intent, slots: opts.detectSlots ? opts.detectSlots(query, kw.intent) : [], ...base };

  return { rung: 'fallback', intent: null, slots: [], ...base };
}
