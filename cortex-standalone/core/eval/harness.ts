/**
 * Run any classifier over a split and produce a full report.
 */

import type { GeneratedExample, SlotSpan } from '../dataset/types';
import { accuracy, confusionMatrix, macroF1, perIntentPRF1, perSlotF1, slotF1, type PrecisionRecallF1 } from './metrics';

export interface ClassifierPrediction {
  intent: string;
  slots: SlotSpan[];
}

export type ClassifierFn = (utterance: string) => ClassifierPrediction | Promise<ClassifierPrediction>;

export interface EvalReport {
  n: number;
  accuracy: number;
  macroF1: number;
  perIntent: Record<string, PrecisionRecallF1>;
  confusion: Record<string, Record<string, number>>;
  slotF1: PrecisionRecallF1;
  perSlotF1: Record<string, PrecisionRecallF1>;
}

export interface EvaluateOptions {
  /** Label set for accuracy/confusion/per-intent metrics. */
  labels: readonly string[];
  /** Slot names for the per-slot breakdown. */
  slotNames: readonly string[];
}

export async function evaluate(classifier: ClassifierFn, split: readonly GeneratedExample[], opts: EvaluateOptions): Promise<EvalReport> {
  const golds: string[] = [];
  const preds: string[] = [];
  const goldSlotEx: { utterance: string; slots: SlotSpan[] }[] = [];
  const predSlotEx: { utterance: string; slots: SlotSpan[] }[] = [];

  for (const ex of split) {
    const prediction = await classifier(ex.utterance);
    golds.push(ex.intent);
    preds.push(prediction.intent);
    goldSlotEx.push({ utterance: ex.utterance, slots: ex.slots });
    predSlotEx.push({ utterance: ex.utterance, slots: prediction.slots });
  }

  const perIntent = perIntentPRF1(golds, preds, opts.labels);
  return {
    n: split.length,
    accuracy: accuracy(golds, preds),
    macroF1: macroF1(perIntent),
    perIntent,
    confusion: confusionMatrix(golds, preds, opts.labels),
    slotF1: slotF1(goldSlotEx, predSlotEx),
    perSlotF1: perSlotF1(goldSlotEx, predSlotEx, opts.slotNames),
  };
}
