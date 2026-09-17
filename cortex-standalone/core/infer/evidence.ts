/**
 * The evidence chain every answer carries (docs/CONTRACTS.md §5.4).
 */

export type LadderRung = 'cortex' | 'keyword' | 'fallback';

export interface EvidenceChain {
  /** Ledger version of the loaded model, or null in keyword-only mode. */
  modelVersion: string | null;
  /** sha256 of the loaded ONNX file (re-verified at load), or null in keyword-only mode. */
  modelSha256: string | null;
  /** Matched intent id, or null on the fallback rung. */
  intent: string | null;
  /** The model's confidence for its top intent (even when a lower rung answered), or null
   * when no model ran. */
  intentConf: number | null;
  ladderRung: LadderRung;
  /** `planner.howto.<id>` | `planner.status.<id>` | `planner.meta.greeting` |
   * `planner.meta.out_of_domain` | `planner.fallback`. */
  plannerTemplateId: string;
  /** Stable ids into the registry: `intent:<id>`, `slot.<name>:<resolvedId|text>`. */
  registryEntriesUsed: string[];
}

export function emptyEvidence(ladderRung: LadderRung, plannerTemplateId = 'planner.fallback'): EvidenceChain {
  return {
    modelVersion: null,
    modelSha256: null,
    intent: null,
    intentConf: null,
    ladderRung,
    plannerTemplateId,
    registryEntriesUsed: [],
  };
}
