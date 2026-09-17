/**
 * Registry types — the host-authored description of an app (docs/CONTRACTS.md §1).
 *
 * `Registry` is the raw on-disk shape (optional fields may be absent). `NormalizedRegistry` is
 * what every other module consumes: every optional intent field is filled with its default,
 * `heldout` is always an array, `assistantName` is always set, and the two built-in meta
 * intents (`greeting`, `out_of_domain`) are present in the taxonomy (§1.4/§1.6).
 */

export type IntentFamily = 'howto' | 'status' | 'meta';

export interface VocabEntry {
  id: string;
  label: string;
}

export interface SlotDef {
  label?: string;
  vocab: VocabEntry[];
}

export interface LinkDef {
  label: string;
  /** Opaque JSON handed to the host's `navigate(target)` (§1.5). */
  target: unknown;
  /** Opaque string handed to the host's `isAllowed(need)`; a refused link is dropped (§1.5). */
  need?: string;
}

export interface HowtoAnswer {
  steps: string[];
  links?: LinkDef[];
}

export interface StatusAnswer {
  unavailable: string;
  links?: LinkDef[];
}

export interface IntentDef {
  id: string;
  family: IntentFamily;
  label: string;
  slots?: string[];
  keywords?: string;
  templates?: string[];
  paraphrases?: string[];
  answer?: HowtoAnswer;
  status?: StatusAnswer;
}

export interface HeldoutExample {
  utterance: string;
  intent: string;
}

export interface AppInfo {
  slug: string;
  name: string;
  assistantName?: string;
}

export interface Registry {
  registryVersion: '1';
  app: AppInfo;
  slots: Record<string, SlotDef>;
  intents: IntentDef[];
  heldout?: HeldoutExample[];
}

// ── normalized form ─────────────────────────────────────────────────────────────────────

export interface NormalizedIntentDef extends IntentDef {
  slots: string[];
  keywords: string;
  templates: string[];
  paraphrases: string[];
}

export interface NormalizedAppInfo extends AppInfo {
  assistantName: string;
}

export interface NormalizedRegistry extends Registry {
  app: NormalizedAppInfo;
  intents: NormalizedIntentDef[];
  heldout: HeldoutExample[];
}

export const BUILTIN_INTENT_IDS = ['greeting', 'out_of_domain'] as const;
export type BuiltinIntentId = (typeof BUILTIN_INTENT_IDS)[number];

/** Taxonomy order = declaration order (§1.6). */
export function intentIds(registry: NormalizedRegistry): string[] {
  return registry.intents.map((i) => i.id);
}

export function getIntent(registry: NormalizedRegistry, id: string): NormalizedIntentDef | undefined {
  return registry.intents.find((i) => i.id === id);
}

/** `["O", "B-<slot>", "I-<slot>", …]` in declared slot order (§2.8). */
export function slotNamesOf(registry: NormalizedRegistry): string[] {
  return Object.keys(registry.slots);
}

export function slotLabelsOf(registry: NormalizedRegistry): string[] {
  return ['O', ...slotNamesOf(registry).flatMap((s) => [`B-${s}`, `I-${s}`])];
}
