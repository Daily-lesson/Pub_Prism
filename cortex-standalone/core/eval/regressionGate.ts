/**
 * The eval gate: dataset determinism (via `ledger.trainedFrom.seed`), artifact sha256,
 * per-intent F1 vs the ledger (epsilon 0.03), and the held-out floor — the last one ONLY when
 * the ledger's `metrics.heldout` is non-null (docs/CONTRACTS.md §4.5).
 */

import { compileDataset } from '../dataset/compile';
import { normalizeRegistry } from '../registry/validate';
import { registryHash } from '../registry/hash';
import type { NormalizedRegistry, Registry } from '../registry/types';
import { classify, loadArtifacts, verifyArtifacts, ArtifactIntegrityError, type LedgerDoc } from '../infer/session';
import { confusionMatrix, perIntentPRF1 } from './metrics';

export const PER_INTENT_F1_REGRESSION_EPSILON = 0.03;

export interface PerIntentRegression {
  intent: string;
  ledgerF1: number;
  currentF1: number;
  delta: number;
}

/** Every intent in `baseline` whose F1 in `current` dropped more than `epsilon` below it.
 * Intents absent from `baseline` are never checked (no baseline to regress against). */
export function diffPerIntentF1(
  baseline: Record<string, number>,
  current: Record<string, number>,
  epsilon: number = PER_INTENT_F1_REGRESSION_EPSILON,
): PerIntentRegression[] {
  const regressions: PerIntentRegression[] = [];
  for (const [intent, ledgerF1] of Object.entries(baseline)) {
    const currentF1 = current[intent] ?? 0;
    const delta = currentF1 - ledgerF1;
    if (delta < -epsilon) regressions.push({ intent, ledgerF1, currentF1, delta });
  }
  return regressions;
}

export interface GateReport {
  passed: boolean;
  reasons: string[];
  ledgerVersion: string | null;
  registry: { registryHash: string; ledgerRegistryHash: string | null; matches: boolean };
  dataset: { datasetHash: string | null; expectedDatasetHash: string | null; deterministic: boolean };
  artifact: {
    sha256Verified: boolean;
    files: Record<string, { file: string; expected: string; actual: string; matches: boolean }>;
    error?: string;
  };
  inDistribution: {
    n: number;
    accuracy: number;
    ledgerAccuracy: number | null;
    perIntentF1: Record<string, number>;
    confusion: Record<string, Record<string, number>>;
    regressions: PerIntentRegression[];
  } | null;
  heldout: { enforced: boolean; n: number; accuracy: number | null; floor: number | null } | null;
}

export interface GateOptions {
  registry: Registry | NormalizedRegistry;
  artifactsDir: string;
  epsilon?: number;
}

export async function runRegressionGate(opts: GateOptions): Promise<GateReport> {
  const epsilon = opts.epsilon ?? PER_INTENT_F1_REGRESSION_EPSILON;
  const registry = normalizeRegistry(opts.registry);
  const reasons: string[] = [];
  const currentRegistryHash = registryHash(registry);

  // ── artifact sha256 (itemized) ──
  let ledger: LedgerDoc | null = null;
  let artifact: GateReport['artifact'] = { sha256Verified: false, files: {} };
  try {
    const v = verifyArtifacts(opts.artifactsDir);
    ledger = v.ledger;
    artifact = { sha256Verified: true, files: v.files };
  } catch (e) {
    const msg = (e as Error).message;
    artifact = { sha256Verified: false, files: {}, error: msg };
    reasons.push(e instanceof ArtifactIntegrityError ? msg : `artifact load error: ${msg}`);
    return {
      passed: false,
      reasons,
      ledgerVersion: null,
      registry: { registryHash: currentRegistryHash, ledgerRegistryHash: null, matches: false },
      dataset: { datasetHash: null, expectedDatasetHash: null, deterministic: false },
      artifact,
      inDistribution: null,
      heldout: null,
    };
  }

  // ── dataset determinism ──
  const compiled = compileDataset(registry, ledger.trainedFrom.seed);
  const deterministic = compiled.manifest.datasetHash === ledger.trainedFrom.datasetHash;
  if (!deterministic) {
    reasons.push(
      `dataset drift: compileDataset(seed=${ledger.trainedFrom.seed}) produced datasetHash=${compiled.manifest.datasetHash}, ` +
        `but ledger.json.trainedFrom.datasetHash=${ledger.trainedFrom.datasetHash} — the registry changed without a retrain.`,
    );
  }

  // ── in-distribution regression ──
  const artifacts = await loadArtifacts(opts.artifactsDir, registry);
  const labels = compiled.manifest.intents;
  const golds: string[] = [];
  const preds: string[] = [];
  for (const ex of compiled.test) {
    golds.push(ex.intent);
    preds.push((await classify(ex.utterance, artifacts)).intent);
  }
  const inDistAccuracy = golds.length === 0 ? 0 : golds.filter((g, i) => g === preds[i]).length / golds.length;
  const perIntent = perIntentPRF1(golds, preds, labels);
  const perIntentF1: Record<string, number> = {};
  for (const [intent, m] of Object.entries(perIntent)) perIntentF1[intent] = m.f1;
  const confusion = confusionMatrix(golds, preds, labels);
  const baseline = ledger.metrics?.inDistribution?.perIntentF1 ?? {};
  const regressions = diffPerIntentF1(baseline, perIntentF1, epsilon);
  for (const r of regressions) {
    reasons.push(
      `per-intent F1 regression: "${r.intent}" dropped from ${r.ledgerF1.toFixed(4)} (ledger) to ${r.currentF1.toFixed(4)} (current), ` +
        `delta=${r.delta.toFixed(4)} exceeds -${epsilon}.`,
    );
  }

  // ── held-out floor (only when the ledger recorded a held-out evaluation) ──
  let heldout: GateReport['heldout'] = null;
  if (ledger.metrics?.heldout) {
    const floor = ledger.acceptanceFloor?.heldoutIntentAccuracy ?? null;
    let correct = 0;
    for (const ex of registry.heldout) {
      if ((await classify(ex.utterance, artifacts)).intent === ex.intent) correct++;
    }
    const n = registry.heldout.length;
    const acc = n === 0 ? 0 : correct / n;
    heldout = { enforced: true, n, accuracy: acc, floor };
    if (n === 0) {
      reasons.push('the ledger records a held-out evaluation but the registry now declares no heldout[] utterances — the held-out set was removed after training; restore it or retrain.');
    } else if (floor !== null && acc < floor) {
      reasons.push(`held-out intent accuracy ${acc.toFixed(4)} is below the ledger's acceptance floor ${floor} (n=${n}).`);
    }
  } else {
    heldout = { enforced: false, n: registry.heldout.length, accuracy: null, floor: null };
  }

  return {
    passed: reasons.length === 0,
    reasons,
    ledgerVersion: ledger.version,
    registry: {
      registryHash: currentRegistryHash,
      ledgerRegistryHash: ledger.trainedFrom.registryHash ?? null,
      matches: currentRegistryHash === ledger.trainedFrom.registryHash,
    },
    dataset: { datasetHash: compiled.manifest.datasetHash, expectedDatasetHash: ledger.trainedFrom.datasetHash, deterministic },
    artifact,
    inDistribution: {
      n: compiled.test.length,
      accuracy: inDistAccuracy,
      ledgerAccuracy: ledger.metrics?.inDistribution?.intentAccuracy ?? null,
      perIntentF1,
      confusion,
      regressions,
    },
    heldout,
  };
}
