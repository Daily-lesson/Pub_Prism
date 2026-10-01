/**
 * `createEngine` — registry + optional model bundle + host hooks → one object that answers
 * queries through the ladder and the planner. A missing or invalid bundle is keyword-only
 * mode (never a throw at answer time); a sha256 mismatch is recorded as `modelLoadError`.
 */

import { validateRegistry, normalizeRegistry } from '../registry/validate';
import { RegistryValidationError } from '../registry/load';
import type { NormalizedRegistry, Registry } from '../registry/types';
import { buildKeywordBags, type KeywordBag } from './keyword';
import { resolveLadder, DEFAULT_CONFIDENCE_THRESHOLD, type LadderResolution } from './ladder';
import { autoDetectSlots } from '../dataset/paraphrases';
import { resolveSlotId } from './session';
import { plan, type IsAllowed, type PlannerChip, type StatusProvider } from './planner';
import { classify as classifyWith, loadArtifacts, verifyArtifacts, type ClassifyResult, type LoadedArtifacts } from './session';
import type { EvidenceChain, LadderRung } from './evidence';

export interface EngineOptions {
  registry: Registry | NormalizedRegistry;
  /** Directory holding `ledger.json` + artifacts. Omit for keyword-only mode. */
  artifactsDir?: string;
  /** Model confidence gate, default 0.6. */
  confidenceGate?: number;
  status?: StatusProvider;
  isAllowed?: IsAllowed;
}

export interface Answer {
  answerText: string;
  chips: PlannerChip[];
  evidence: EvidenceChain;
  intent: string | null;
  intentConf: number | null;
  ladderRung: LadderRung;
  /** null in keyword-only mode. */
  modelVersion: string | null;
}

export interface Engine {
  readonly registry: NormalizedRegistry;
  readonly bags: readonly KeywordBag[];
  readonly mode: 'model' | 'keyword-only';
  /** Why the model is unavailable (missing dir, sha mismatch, load failure); undefined in model mode. */
  readonly modelLoadError: string | undefined;
  /** Ledger version once the bundle is verified; null in keyword-only mode. */
  readonly modelVersion: string | null;
  readonly modelSha256: string | null;
  answer(query: string): Promise<Answer>;
  /** Model only. Rejects with a clear error in keyword-only mode. */
  classify(query: string): Promise<ClassifyResult>;
  /** Load the ONNX session now (otherwise it loads lazily on the first answer). */
  warm(): Promise<void>;
}

export function createEngine(opts: EngineOptions): Engine {
  const validation = validateRegistry(opts.registry);
  if (!validation.ok || !validation.registry) throw new RegistryValidationError(validation.errors);
  const registry: NormalizedRegistry = validation.registry ?? normalizeRegistry(opts.registry);
  const bags = buildKeywordBags(registry);
  const threshold = opts.confidenceGate ?? DEFAULT_CONFIDENCE_THRESHOLD;

  let mode: 'model' | 'keyword-only' = 'keyword-only';
  let modelLoadError: string | undefined;
  let modelVersion: string | null = null;
  let modelSha256: string | null = null;
  let artifacts: LoadedArtifacts | null = null;
  let loading: Promise<void> | null = null;

  if (opts.artifactsDir === undefined) {
    modelLoadError = 'no artifactsDir supplied';
  } else {
    try {
      const v = verifyArtifacts(opts.artifactsDir);
      mode = 'model';
      modelVersion = v.ledger.version;
      modelSha256 = v.ledger.onnx.sha256;
    } catch (e) {
      modelLoadError = (e as Error).message;
    }
  }

  function degrade(reason: string): void {
    mode = 'keyword-only';
    modelLoadError = reason;
    modelVersion = null;
    modelSha256 = null;
    artifacts = null;
  }

  function warm(): Promise<void> {
    if (mode !== 'model' || artifacts) return Promise.resolve();
    if (!loading) {
      loading = loadArtifacts(opts.artifactsDir as string, registry)
        .then((a) => {
          artifacts = a;
        })
        .catch((e: Error) => {
          degrade(`model load failed: ${e.message}`);
        })
        .finally(() => {
          loading = null;
        });
    }
    return loading;
  }

  async function classify(query: string): Promise<ClassifyResult> {
    await warm();
    if (mode !== 'model' || !artifacts) {
      throw new Error(`no model loaded (keyword-only mode): ${modelLoadError ?? 'unknown reason'}`);
    }
    return classifyWith(query, artifacts, registry);
  }

  async function answer(query: string): Promise<Answer> {
    await warm();
    const res: LadderResolution = await resolveLadder(query, {
      classify: mode === 'model' && artifacts ? (q) => classifyWith(q, artifacts as LoadedArtifacts, registry) : undefined,
      bags,
      threshold,
      detectSlots: (q, intentId) => {
        const intent = registry.intents.find((i) => i.id === intentId);
        if (!intent) return [];
        return autoDetectSlots(registry, q, intent.slots).map((s) => ({ ...s, resolvedId: resolveSlotId(registry, s.name, s.value) }));
      },
    });
    const planned = await plan(registry, res.intent, res.slots, { status: opts.status, isAllowed: opts.isAllowed });
    const evidence: EvidenceChain = {
      modelVersion: mode === 'model' ? modelVersion : null,
      modelSha256: mode === 'model' ? modelSha256 : null,
      intent: planned.evidence.intent,
      intentConf: res.intentConf,
      ladderRung: res.rung,
      plannerTemplateId: planned.evidence.plannerTemplateId,
      registryEntriesUsed: planned.evidence.registryEntriesUsed,
    };
    return {
      answerText: planned.answerText,
      chips: planned.chips,
      evidence,
      intent: planned.evidence.intent,
      intentConf: res.intentConf,
      ladderRung: res.rung,
      modelVersion: evidence.modelVersion,
    };
  }

  return {
    registry,
    bags,
    get mode() {
      return mode;
    },
    get modelLoadError() {
      return modelLoadError;
    },
    get modelVersion() {
      return modelVersion;
    },
    get modelSha256() {
      return modelSha256;
    },
    answer,
    classify,
    warm,
  };
}
