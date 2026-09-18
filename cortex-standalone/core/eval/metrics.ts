/**
 * Pure evaluation metrics — intent classification + token-level slot tagging. No I/O.
 */

import type { SlotSpan } from '../dataset/types';

export interface PrecisionRecallF1 {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export function accuracy(golds: readonly string[], preds: readonly string[]): number {
  if (golds.length !== preds.length) throw new Error(`accuracy: golds/preds length mismatch (${golds.length} vs ${preds.length})`);
  if (golds.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < golds.length; i++) if (golds[i] === preds[i]) correct++;
  return correct / golds.length;
}

/** gold -> pred -> count; every label in `labels` appears even with zero counts. */
export function confusionMatrix(golds: readonly string[], preds: readonly string[], labels: readonly string[]): Record<string, Record<string, number>> {
  if (golds.length !== preds.length) throw new Error(`confusionMatrix: golds/preds length mismatch (${golds.length} vs ${preds.length})`);
  const matrix: Record<string, Record<string, number>> = {};
  for (const g of labels) {
    matrix[g] = {};
    for (const p of labels) matrix[g][p] = 0;
  }
  for (let i = 0; i < golds.length; i++) {
    const g = golds[i];
    const p = preds[i];
    if (!(g in matrix)) matrix[g] = Object.fromEntries(labels.map((l) => [l, 0]));
    if (!(p in matrix[g])) matrix[g][p] = 0;
    matrix[g][p]++;
  }
  return matrix;
}

/** One-vs-rest precision/recall/F1/support per label. */
export function perLabelPRF1(golds: readonly string[], preds: readonly string[], labels: readonly string[]): Record<string, PrecisionRecallF1> {
  if (golds.length !== preds.length) throw new Error(`perLabelPRF1: golds/preds length mismatch (${golds.length} vs ${preds.length})`);
  const out: Record<string, PrecisionRecallF1> = {};
  for (const label of labels) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (let i = 0; i < golds.length; i++) {
      const isGold = golds[i] === label;
      const isPred = preds[i] === label;
      if (isGold) support++;
      if (isGold && isPred) tp++;
      else if (!isGold && isPred) fp++;
      else if (isGold && !isPred) fn++;
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    out[label] = { precision, recall, f1, support };
  }
  return out;
}

export const perIntentPRF1 = perLabelPRF1;

/** Unweighted mean of per-label F1. */
export function macroF1(per: Record<string, PrecisionRecallF1>): number {
  const vals = Object.values(per);
  if (!vals.length) return 0;
  return vals.reduce((a, b) => a + b.f1, 0) / vals.length;
}

// ── slot F1 (token-level, whitespace tokens) ─────────────────────────────────────────

interface Token {
  text: string;
  start: number;
  end: number;
}

function wsTokens(utterance: string): Token[] {
  const tokens: Token[] = [];
  const re = /\S+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(utterance))) tokens.push({ text: m[0], start: m.index, end: m.index + m[0].length });
  return tokens;
}

/** Tag each whitespace token with the overlapping slot's name, or 'O'. */
export function tagTokens(utterance: string, slots: readonly SlotSpan[]): string[] {
  return wsTokens(utterance).map((t) => {
    const hit = slots.find((s) => t.start < s.end && t.end > s.start);
    return hit ? hit.name : 'O';
  });
}

export interface SlotExampleLabel {
  utterance: string;
  slots: readonly SlotSpan[];
}

/** Micro-averaged token-level slot F1; 'O' tokens excluded from the denominators. */
export function slotF1(golds: readonly SlotExampleLabel[], preds: readonly SlotExampleLabel[]): PrecisionRecallF1 {
  if (golds.length !== preds.length) throw new Error(`slotF1: golds/preds length mismatch (${golds.length} vs ${preds.length})`);
  let tp = 0;
  let predPositive = 0;
  let goldPositive = 0;
  for (let i = 0; i < golds.length; i++) {
    const goldTags = tagTokens(golds[i].utterance, golds[i].slots);
    const predTags = tagTokens(preds[i].utterance, preds[i].slots);
    const n = Math.max(goldTags.length, predTags.length);
    for (let t = 0; t < n; t++) {
      const g = goldTags[t] ?? 'O';
      const p = predTags[t] ?? 'O';
      if (g !== 'O') goldPositive++;
      if (p !== 'O') predPositive++;
      if (g !== 'O' && g === p) tp++;
    }
  }
  const precision = predPositive === 0 ? 0 : tp / predPositive;
  const recall = goldPositive === 0 ? 0 : tp / goldPositive;
  const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1, support: goldPositive };
}

export function perSlotF1(golds: readonly SlotExampleLabel[], preds: readonly SlotExampleLabel[], slotNames: readonly string[]): Record<string, PrecisionRecallF1> {
  if (golds.length !== preds.length) throw new Error(`perSlotF1: golds/preds length mismatch (${golds.length} vs ${preds.length})`);
  const goldAll: string[] = [];
  const predAll: string[] = [];
  for (let i = 0; i < golds.length; i++) {
    const goldTags = tagTokens(golds[i].utterance, golds[i].slots);
    const predTags = tagTokens(preds[i].utterance, preds[i].slots);
    const n = Math.max(goldTags.length, predTags.length);
    for (let t = 0; t < n; t++) {
      goldAll.push(goldTags[t] ?? 'O');
      predAll.push(predTags[t] ?? 'O');
    }
  }
  return perLabelPRF1(goldAll, predAll, slotNames);
}
