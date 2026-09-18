/**
 * The model session (docs/CONTRACTS.md §4). Loads an artifact bundle (`ledger.json` + the
 * onnx/tokenizer/labels files it names), re-verifies each sha256 (§4.6), and classifies an
 * utterance into intent + confidence + slot spans (§4.3). Nothing is cached globally — every
 * `loadArtifacts` call returns its own independent object.
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';
import { loadTokenizerConfig, tokenize, wordSplitWithOffsets, type OffsetToken, type TokenizerConfig } from '../tokenizer';
import { sha256Hex } from '../registry/hash';
import type { NormalizedRegistry } from '../registry/types';

export interface LedgerArtifactRef {
  file: string;
  sha256: string;
}

export interface LedgerDoc {
  version: string;
  app?: string;
  onnx: LedgerArtifactRef & { bytes?: number; quantized?: boolean };
  tokenizer: LedgerArtifactRef;
  labels: LedgerArtifactRef;
  trainedFrom: { registryHash: string; datasetHash: string; seed: number; split?: { train: number; val: number; test: number } };
  modelConfig?: Record<string, number>;
  metrics: {
    inDistribution: { intentAccuracy: number; macroF1?: number; perIntentF1: Record<string, number>; slotF1?: number };
    heldout: { intentAccuracy: number; n: number; perIntentAccuracy?: Record<string, number> } | null;
  };
  acceptanceFloor: { inDistributionIntentAccuracy: number; inDistributionSlotF1?: number; heldoutIntentAccuracy: number };
  heldoutGate?: string;
}

export interface LabelsDoc {
  version: string;
  intents: string[];
  /** BIO tag names, index-aligned to the slot head's classes: `["O","B-x","I-x",…]`. */
  slots: string[];
}

export interface ClassifiedSlot {
  name: string;
  /** The raw text the model tagged. */
  value: string;
  start: number;
  end: number;
  /** The vocab id whose label equals `value` case-insensitively; absent when none does. */
  resolvedId?: string;
}

export interface ClassifyResult {
  intent: string;
  intentConf: number;
  slots: ClassifiedSlot[];
  modelVersion: string;
  modelSha256: string;
}

export class ArtifactIntegrityError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ArtifactIntegrityError';
  }
}

export interface LoadedArtifacts {
  dir: string;
  ledger: LedgerDoc;
  tokenizer: TokenizerConfig;
  labels: LabelsDoc;
  session: ort.InferenceSession;
  registry?: NormalizedRegistry;
}

export interface VerifiedArtifactPaths {
  ledger: LedgerDoc;
  onnxPath: string;
  tokenizerPath: string;
  labelsPath: string;
  files: Record<'onnx' | 'tokenizer' | 'labels', { file: string; expected: string; actual: string; matches: boolean }>;
}

/** Synchronous part of loading: read `ledger.json` and re-verify every sha256. Throws
 * `ArtifactIntegrityError` on a missing file or a mismatch (§4.6). */
export function verifyArtifacts(artifactsDir: string): VerifiedArtifactPaths {
  const ledgerPath = path.join(artifactsDir, 'ledger.json');
  if (!existsSync(ledgerPath)) throw new ArtifactIntegrityError(`no ledger.json in ${artifactsDir}`);
  let ledger: LedgerDoc;
  try {
    ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')) as LedgerDoc;
  } catch (e) {
    throw new ArtifactIntegrityError(`ledger.json in ${artifactsDir} is not valid JSON: ${(e as Error).message}`);
  }
  for (const key of ['onnx', 'tokenizer', 'labels'] as const) {
    const ref = ledger[key];
    if (!ref || typeof ref.file !== 'string' || typeof ref.sha256 !== 'string') {
      throw new ArtifactIntegrityError(`ledger.json in ${artifactsDir} is missing "${key}.file"/"${key}.sha256"`);
    }
  }
  const refs: Array<['onnx' | 'tokenizer' | 'labels', LedgerArtifactRef]> = [
    ['onnx', ledger.onnx],
    ['tokenizer', ledger.tokenizer],
    ['labels', ledger.labels],
  ];
  const files = {} as VerifiedArtifactPaths['files'];
  for (const [key, ref] of refs) {
    const p = path.join(artifactsDir, ref.file);
    if (!existsSync(p)) throw new ArtifactIntegrityError(`${key} artifact ${ref.file} named by ledger.json is missing from ${artifactsDir}`);
    const actual = sha256Hex(readFileSync(p));
    files[key] = { file: ref.file, expected: ref.sha256, actual, matches: actual === ref.sha256 };
  }
  const bad = refs.filter(([key]) => !files[key].matches);
  if (bad.length) {
    const lines = bad.map(([key]) => `${key} (${files[key].file}): expected ${files[key].expected}, got ${files[key].actual}`);
    throw new ArtifactIntegrityError(`artifact sha256 mismatch vs ledger.json in ${artifactsDir} — bundle is corrupted or stale:\n  ${lines.join('\n  ')}`);
  }
  return {
    ledger,
    onnxPath: path.join(artifactsDir, ledger.onnx.file),
    tokenizerPath: path.join(artifactsDir, ledger.tokenizer.file),
    labelsPath: path.join(artifactsDir, ledger.labels.file),
    files,
  };
}

/** Load + integrity-check a bundle and open the ONNX session (CPU execution provider). */
export async function loadArtifacts(artifactsDir: string, registry?: NormalizedRegistry): Promise<LoadedArtifacts> {
  const v = verifyArtifacts(artifactsDir);
  const tokenizer = loadTokenizerConfig(v.tokenizerPath);
  const labels = JSON.parse(readFileSync(v.labelsPath, 'utf8')) as LabelsDoc;
  if (!Array.isArray(labels.intents) || !Array.isArray(labels.slots)) {
    throw new ArtifactIntegrityError(`labels file ${v.ledger.labels.file} must carry "intents" and "slots" arrays`);
  }
  const session = await ort.InferenceSession.create(v.onnxPath, { executionProviders: ['cpu'] });
  return { dir: artifactsDir, ledger: v.ledger, tokenizer, labels, session, registry };
}

// ── decoding ──────────────────────────────────────────────────────────────────────────

export function softmax(logits: ArrayLike<number>): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exps = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    exps[i] = e;
    sum += e;
  }
  for (let i = 0; i < exps.length; i++) exps[i] /= sum;
  return exps;
}

export function argmax(values: ArrayLike<number>): number {
  let best = 0;
  let bestV = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > bestV) {
      bestV = values[i];
      best = i;
    }
  }
  return best;
}

/** `resolvedId` = the vocab id whose label equals the span text case-insensitively (§4.3). */
export function resolveSlotId(registry: NormalizedRegistry | undefined, name: string, text: string): string | undefined {
  const def = registry?.slots[name];
  if (!def) return undefined;
  const needle = text.trim().toLowerCase();
  if (!needle) return undefined;
  for (const entry of def.vocab) if (entry.label.toLowerCase() === needle) return entry.id;
  return undefined;
}

/** BIO labels (aligned to `tokens`) → spans. `B-x` or a name change starts a span; `I-x`
 * extends; `O` closes (§4.3). */
export function decodeSlots(
  utterance: string,
  bioLabels: readonly string[],
  tokens: readonly OffsetToken[],
  registry?: NormalizedRegistry,
): ClassifiedSlot[] {
  const slots: ClassifiedSlot[] = [];
  let curName: string | null = null;
  let curStart = -1;
  let curEnd = -1;

  const flush = () => {
    if (curName === null) return;
    const text = utterance.slice(curStart, curEnd);
    const resolvedId = resolveSlotId(registry, curName, text);
    slots.push({ name: curName, value: text, start: curStart, end: curEnd, ...(resolvedId !== undefined ? { resolvedId } : {}) });
    curName = null;
  };

  for (let t = 0; t < bioLabels.length && t < tokens.length; t++) {
    const label = bioLabels[t];
    const dash = label.indexOf('-');
    if (label === 'O' || dash === -1) {
      flush();
      continue;
    }
    const prefix = label.slice(0, dash);
    const name = label.slice(dash + 1);
    const tok = tokens[t];
    if (prefix === 'B' || name !== curName) {
      flush();
      curName = name;
      curStart = tok.start;
      curEnd = tok.end;
    } else {
      curEnd = tok.end;
    }
  }
  flush();
  return slots;
}

/** Classify `utterance` with a loaded bundle. `registry` (for `resolvedId`) defaults to the
 * one passed to `loadArtifacts`. */
export async function classify(utterance: string, artifacts: LoadedArtifacts, registry?: NormalizedRegistry): Promise<ClassifyResult> {
  const reg = registry ?? artifacts.registry;
  const { session, tokenizer, labels, ledger } = artifacts;

  const ids = tokenize(utterance, tokenizer);
  const idsBig = BigInt64Array.from(ids.map((n) => BigInt(n)));
  const inputTensor = new ort.Tensor('int64', idsBig, [1, tokenizer.maxLen]);

  const outputs = await session.run({ input_ids: inputTensor });
  const intentLogits = outputs.intent_logits.data as unknown as Float32Array;
  const slotLogits = outputs.slot_logits.data as unknown as Float32Array;

  const probs = softmax(intentLogits);
  const intentIdx = argmax(probs);
  const intent = labels.intents[intentIdx] ?? 'out_of_domain';
  const intentConf = probs[intentIdx];

  const numSlotLabels = labels.slots.length;
  const offsetTokens = wordSplitWithOffsets(utterance);
  const realLen = Math.min(offsetTokens.length, tokenizer.maxLen);

  const bioLabels: string[] = [];
  for (let t = 0; t < realLen; t++) {
    const base = t * numSlotLabels;
    let best = 0;
    let bestV = -Infinity;
    for (let c = 0; c < numSlotLabels; c++) {
      const v = slotLogits[base + c];
      if (v > bestV) {
        bestV = v;
        best = c;
      }
    }
    bioLabels.push(labels.slots[best] ?? 'O');
  }

  const slots = decodeSlots(utterance, bioLabels, offsetTokens.slice(0, realLen), reg);
  return { intent, intentConf, slots, modelVersion: ledger.version, modelSha256: ledger.onnx.sha256 };
}
