/**
 * The deterministic Answer Planner (docs/CONTRACTS.md §5.4–§5.5). Every sentence is either
 * registry copy the host wrote, a fixed app-neutral meta/fallback line, or a string the host's
 * status provider returned. No free text is generated, and no number is ever invented.
 */

import type { LinkDef, NormalizedRegistry } from '../registry/types';
import type { ClassifiedSlot } from './session';

export interface PlannerChip {
  label: string;
  target: unknown;
}

export interface PlannerEvidence {
  intent: string | null;
  plannerTemplateId: string;
  registryEntriesUsed: string[];
}

export interface PlannerResult {
  answerText: string;
  chips: PlannerChip[];
  evidence: PlannerEvidence;
}

export type StatusProvider = (intentId: string, slots: ClassifiedSlot[]) => string | undefined | Promise<string | undefined>;
export type IsAllowed = (need: string) => boolean;

export interface PlanOptions {
  status?: StatusProvider;
  /** Absent = allow everything (fail-open, UX-only — §1.5). */
  isAllowed?: IsAllowed;
}

export const GREETING_COPY = 'Hi! Ask me how to do something in this app, or what its current state is.';
export const OUT_OF_DOMAIN_COPY = "That's outside what I can help with here — I only know this app. Try asking how to do something in it.";
export const FALLBACK_COPY = "I'm not sure what you're asking.";
export const UNAVAILABLE_FALLBACK_COPY = "I can't read that from here.";

function gateLinks(links: readonly LinkDef[] | undefined, isAllowed?: IsAllowed): PlannerChip[] {
  const out: PlannerChip[] = [];
  for (const l of links ?? []) {
    if (l.need !== undefined && isAllowed && !isAllowed(l.need)) continue;
    out.push({ label: l.label, target: l.target });
  }
  return out;
}

function slotEntries(slots: readonly ClassifiedSlot[]): string[] {
  return slots.map((s) => `slot.${s.name}:${s.resolvedId ?? s.value}`);
}

function slotPrefix(slots: readonly ClassifiedSlot[]): string {
  if (!slots.length) return '';
  return `Noted — you mentioned ${slots.map((s) => `"${s.value}"`).join(', ')}. `;
}

/** The guide list: every non-meta intent's label, in taxonomy order. */
export function guideList(registry: NormalizedRegistry): string[] {
  return registry.intents.filter((i) => i.family !== 'meta').map((i) => i.label);
}

function planFallback(registry: NormalizedRegistry): PlannerResult {
  const guide = guideList(registry);
  const list = guide.length ? ` Here's what I can help with: ${guide.join('; ')}.` : '';
  return {
    answerText: `${FALLBACK_COPY}${list}`,
    chips: [],
    evidence: { intent: null, plannerTemplateId: 'planner.fallback', registryEntriesUsed: [] },
  };
}

/**
 * Compose the answer for `intent`. `intent === null` (or an id the registry doesn't know)
 * yields the honest fallback.
 */
export async function plan(
  registry: NormalizedRegistry,
  intent: string | null,
  slots: readonly ClassifiedSlot[],
  opts: PlanOptions = {},
): Promise<PlannerResult> {
  if (intent === 'greeting') {
    return { answerText: GREETING_COPY, chips: [], evidence: { intent, plannerTemplateId: 'planner.meta.greeting', registryEntriesUsed: ['intent:greeting'] } };
  }
  if (intent === 'out_of_domain') {
    return {
      answerText: OUT_OF_DOMAIN_COPY,
      chips: [],
      evidence: { intent, plannerTemplateId: 'planner.meta.out_of_domain', registryEntriesUsed: ['intent:out_of_domain'] },
    };
  }
  const def = intent === null ? undefined : registry.intents.find((i) => i.id === intent);
  if (!def || intent === null) return planFallback(registry);

  const used = [`intent:${intent}`, ...slotEntries(slots)];

  if (def.family === 'howto' && def.answer) {
    const numbered = def.answer.steps.map((s, i) => `${i + 1}. ${s}`).join(' ');
    return {
      answerText: `${slotPrefix(slots)}${numbered}`,
      chips: gateLinks(def.answer.links, opts.isAllowed),
      evidence: { intent, plannerTemplateId: `planner.howto.${intent}`, registryEntriesUsed: used },
    };
  }

  if (def.family === 'status') {
    let text: string | undefined;
    if (opts.status) {
      try {
        const r = await opts.status(intent, [...slots]);
        if (typeof r === 'string' && r.length > 0) text = r;
      } catch {
        text = undefined;
      }
    }
    if (text === undefined) text = def.status?.unavailable ?? UNAVAILABLE_FALLBACK_COPY;
    return {
      answerText: text,
      chips: gateLinks(def.status?.links, opts.isAllowed),
      evidence: { intent, plannerTemplateId: `planner.status.${intent}`, registryEntriesUsed: used },
    };
  }

  // A host-declared meta intent other than the two built-ins has no copy of its own.
  return planFallback(registry);
}
