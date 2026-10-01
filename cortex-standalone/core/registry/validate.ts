/**
 * Registry validation + normalization (docs/CONTRACTS.md §1.1–§1.8).
 *
 * `validateRegistry` is a hand-written implementation of `registry/schema.json`'s structural
 * rules (no JSON-Schema dependency) plus the rules the schema cannot express (§1.2 placeholders
 * name declared slots, §1.3 family-specific required copy, held-out intents exist). One
 * human-readable line per error, naming the intent/slot it concerns.
 *
 * `normalizeRegistry` fills defaults and appends the built-in meta intents (§1.4/§1.6). It is
 * idempotent and structural — it does not validate; call `validateRegistry` first on untrusted
 * input (`loadRegistryFile` does both).
 */

import { BUILTIN_INTENT_IDS, type NormalizedIntentDef, type NormalizedRegistry, type Registry } from './types';
import { builtinIntent } from './builtins';

export interface ValidationResult {
  ok: boolean;
  errors: string[];
  /** The normalized registry, present only when `ok`. */
  registry?: NormalizedRegistry;
}

const SLUG_RE = /^[a-z0-9][a-z0-9-]*$/;
const SLOT_NAME_RE = /^[a-z][a-zA-Z0-9]*$/;
const INTENT_ID_RE = /^[a-z][a-z0-9_]*$/;
const PLACEHOLDER_RE = /\{([^{}]*)\}/g;
const FAMILIES = ['howto', 'status', 'meta'] as const;

type Obj = Record<string, unknown>;

function isObj(v: unknown): v is Obj {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

function extraKeys(obj: Obj, allowed: readonly string[]): string[] {
  return Object.keys(obj).filter((k) => !allowed.includes(k));
}

function checkString(
  errors: string[],
  where: string,
  value: unknown,
  opts: { minLength?: number; maxLength?: number; pattern?: RegExp; patternText?: string } = {},
): value is string {
  if (typeof value !== 'string') {
    errors.push(`${where}: must be a string`);
    return false;
  }
  if (opts.minLength !== undefined && value.length < opts.minLength) {
    errors.push(`${where}: must be at least ${opts.minLength} character(s)`);
    return false;
  }
  if (opts.maxLength !== undefined && value.length > opts.maxLength) {
    errors.push(`${where}: must be at most ${opts.maxLength} characters`);
    return false;
  }
  if (opts.pattern && !opts.pattern.test(value)) {
    errors.push(`${where}: must match ${opts.patternText ?? String(opts.pattern)}`);
    return false;
  }
  return true;
}

function checkStringArray(errors: string[], where: string, value: unknown, item: { minLength: number; maxLength: number }): void {
  if (!Array.isArray(value)) {
    errors.push(`${where}: must be an array of strings`);
    return;
  }
  value.forEach((s, i) => checkString(errors, `${where}[${i}]`, s, item));
}

function checkLinks(errors: string[], where: string, value: unknown): void {
  if (value === undefined) return;
  if (!Array.isArray(value)) {
    errors.push(`${where}: must be an array`);
    return;
  }
  if (value.length > 8) errors.push(`${where}: at most 8 links allowed (got ${value.length})`);
  value.forEach((link, i) => {
    const w = `${where}[${i}]`;
    if (!isObj(link)) {
      errors.push(`${w}: must be an object {label, target, need?}`);
      return;
    }
    for (const k of extraKeys(link, ['label', 'target', 'need'])) errors.push(`${w}: unknown property "${k}"`);
    checkString(errors, `${w}.label`, link.label, { minLength: 1, maxLength: 80 });
    if (!('target' in link)) errors.push(`${w}: missing required "target"`);
    if (link.need !== undefined) checkString(errors, `${w}.need`, link.need, { maxLength: 80 });
  });
}

/** Every `{placeholder}` in a template, in order (duplicates kept). */
export function templatePlaceholders(template: string): string[] {
  const out: string[] = [];
  let m: RegExpExecArray | null;
  const re = new RegExp(PLACEHOLDER_RE.source, 'g');
  while ((m = re.exec(template))) out.push(m[1]);
  return out;
}

/**
 * Validate a raw JSON value against the registry contract. Never throws on malformed input —
 * every problem becomes one line in `errors`.
 */
export function validateRegistry(raw: unknown): ValidationResult {
  const errors: string[] = [];

  if (!isObj(raw)) return { ok: false, errors: ['registry: must be a JSON object'] };

  for (const k of extraKeys(raw, ['registryVersion', 'app', 'slots', 'intents', 'heldout'])) {
    errors.push(`registry: unknown top-level property "${k}"`);
  }
  for (const k of ['registryVersion', 'app', 'slots', 'intents']) {
    if (!(k in raw)) errors.push(`registry: missing required "${k}"`);
  }
  if ('registryVersion' in raw && raw.registryVersion !== '1') {
    errors.push(`registryVersion: must be "1" (got ${JSON.stringify(raw.registryVersion)})`);
  }

  // ── app ──
  if ('app' in raw) {
    if (!isObj(raw.app)) {
      errors.push('app: must be an object {slug, name, assistantName?}');
    } else {
      for (const k of extraKeys(raw.app, ['slug', 'name', 'assistantName'])) errors.push(`app: unknown property "${k}"`);
      if (!('slug' in raw.app)) errors.push('app: missing required "slug"');
      else checkString(errors, 'app.slug', raw.app.slug, { maxLength: 64, pattern: SLUG_RE, patternText: '^[a-z0-9][a-z0-9-]*$' });
      if (!('name' in raw.app)) errors.push('app: missing required "name"');
      else checkString(errors, 'app.name', raw.app.name, { minLength: 1, maxLength: 120 });
      if (raw.app.assistantName !== undefined) checkString(errors, 'app.assistantName', raw.app.assistantName, { minLength: 1, maxLength: 120 });
    }
  }

  // ── slots ──
  const slotVocabSize = new Map<string, number>();
  if ('slots' in raw) {
    if (!isObj(raw.slots)) {
      errors.push('slots: must be an object mapping slot name -> { label?, vocab }');
    } else {
      for (const [name, def] of Object.entries(raw.slots)) {
        if (!SLOT_NAME_RE.test(name)) errors.push(`slot "${name}": name must match ^[a-z][a-zA-Z0-9]*$`);
        if (!isObj(def)) {
          errors.push(`slot "${name}": must be an object { label?, vocab }`);
          continue;
        }
        for (const k of extraKeys(def, ['label', 'vocab'])) errors.push(`slot "${name}": unknown property "${k}"`);
        if (def.label !== undefined) checkString(errors, `slot "${name}".label`, def.label, { maxLength: 120 });
        if (!('vocab' in def)) {
          errors.push(`slot "${name}": missing required "vocab"`);
          continue;
        }
        if (!Array.isArray(def.vocab)) {
          errors.push(`slot "${name}".vocab: must be an array of { id, label }`);
          continue;
        }
        slotVocabSize.set(name, def.vocab.length);
        const seenIds = new Set<string>();
        def.vocab.forEach((entry, i) => {
          const w = `slot "${name}".vocab[${i}]`;
          if (!isObj(entry)) {
            errors.push(`${w}: must be an object { id, label }`);
            return;
          }
          for (const k of extraKeys(entry, ['id', 'label'])) errors.push(`${w}: unknown property "${k}"`);
          if (!('id' in entry)) errors.push(`${w}: missing required "id"`);
          else if (checkString(errors, `${w}.id`, entry.id, { minLength: 1, maxLength: 120 })) {
            if (seenIds.has(entry.id)) errors.push(`${w}: duplicate vocab id "${entry.id}" in slot "${name}"`);
            seenIds.add(entry.id);
          }
          if (!('label' in entry)) errors.push(`${w}: missing required "label"`);
          else checkString(errors, `${w}.label`, entry.label, { minLength: 1, maxLength: 200 });
        });
      }
    }
  }

  // ── intents ──
  const intentIds = new Set<string>();
  if ('intents' in raw) {
    if (!Array.isArray(raw.intents)) {
      errors.push('intents: must be an array');
    } else {
      if (raw.intents.length === 0) errors.push('intents: at least one intent is required');
      raw.intents.forEach((intent, i) => {
        const idText = isObj(intent) && typeof intent.id === 'string' ? intent.id : `#${i}`;
        const w = `intent "${idText}"`;
        if (!isObj(intent)) {
          errors.push(`intent #${i}: must be an object`);
          return;
        }
        for (const k of extraKeys(intent, ['id', 'family', 'label', 'slots', 'keywords', 'templates', 'paraphrases', 'answer', 'status'])) {
          errors.push(`${w}: unknown property "${k}"`);
        }
        if (!('id' in intent)) errors.push(`intent #${i}: missing required "id"`);
        else if (checkString(errors, `${w}.id`, intent.id, { maxLength: 64, pattern: INTENT_ID_RE, patternText: '^[a-z][a-z0-9_]*$' })) {
          if (intentIds.has(intent.id)) errors.push(`${w}: duplicate intent id "${intent.id}" (rule 1.1)`);
          intentIds.add(intent.id);
        }
        if (!('family' in intent)) errors.push(`${w}: missing required "family"`);
        else if (!FAMILIES.includes(intent.family as (typeof FAMILIES)[number])) {
          errors.push(`${w}.family: must be one of howto|status|meta (got ${JSON.stringify(intent.family)})`);
        }
        if (!('label' in intent)) errors.push(`${w}: missing required "label"`);
        else checkString(errors, `${w}.label`, intent.label, { minLength: 1, maxLength: 200 });

        const declaredSlots: string[] = [];
        if (intent.slots !== undefined) {
          if (!Array.isArray(intent.slots)) errors.push(`${w}.slots: must be an array of slot names`);
          else {
            intent.slots.forEach((s, j) => {
              if (typeof s !== 'string') errors.push(`${w}.slots[${j}]: must be a string`);
              else {
                declaredSlots.push(s);
                if (!slotVocabSize.has(s)) errors.push(`${w}: slot "${s}" is not declared in registry.slots (rule 1.2)`);
              }
            });
          }
        }
        if (intent.keywords !== undefined) checkString(errors, `${w}.keywords`, intent.keywords, { maxLength: 2000 });
        if (intent.templates !== undefined) checkStringArray(errors, `${w}.templates`, intent.templates, { minLength: 1, maxLength: 300 });
        if (intent.paraphrases !== undefined) checkStringArray(errors, `${w}.paraphrases`, intent.paraphrases, { minLength: 1, maxLength: 300 });

        // §1.2 placeholders
        if (Array.isArray(intent.templates)) {
          intent.templates.forEach((t, j) => {
            if (typeof t !== 'string') return;
            for (const ph of templatePlaceholders(t)) {
              if (!declaredSlots.includes(ph)) {
                errors.push(`${w}.templates[${j}]: placeholder {${ph}} is not listed in the intent's slots (rule 1.2)`);
              } else if ((slotVocabSize.get(ph) ?? 0) === 0) {
                errors.push(`${w}.templates[${j}]: placeholder {${ph}} names slot "${ph}" whose vocab is empty (rule 1.2)`);
              }
            }
          });
        }

        // answer / status shapes
        if (intent.answer !== undefined) {
          if (!isObj(intent.answer)) errors.push(`${w}.answer: must be an object { steps, links? }`);
          else {
            for (const k of extraKeys(intent.answer, ['steps', 'links'])) errors.push(`${w}.answer: unknown property "${k}"`);
            if (!('steps' in intent.answer)) errors.push(`${w}.answer: missing required "steps"`);
            else if (!Array.isArray(intent.answer.steps) || intent.answer.steps.length === 0) {
              errors.push(`${w}.answer.steps: must be a non-empty array of strings`);
            } else checkStringArray(errors, `${w}.answer.steps`, intent.answer.steps, { minLength: 1, maxLength: 500 });
            checkLinks(errors, `${w}.answer.links`, intent.answer.links);
          }
        }
        if (intent.status !== undefined) {
          if (!isObj(intent.status)) errors.push(`${w}.status: must be an object { unavailable, links? }`);
          else {
            for (const k of extraKeys(intent.status, ['unavailable', 'links'])) errors.push(`${w}.status: unknown property "${k}"`);
            if (!('unavailable' in intent.status)) errors.push(`${w}.status: missing required "unavailable"`);
            else checkString(errors, `${w}.status.unavailable`, intent.status.unavailable, { minLength: 1, maxLength: 500 });
            checkLinks(errors, `${w}.status.links`, intent.status.links);
          }
        }

        // §1.3 family requirements
        if (intent.family === 'howto') {
          const steps = isObj(intent.answer) ? intent.answer.steps : undefined;
          if (!Array.isArray(steps) || steps.length === 0) errors.push(`${w}: a howto intent must have answer.steps with at least one step (rule 1.3)`);
        } else if (intent.family === 'status') {
          const unavailable = isObj(intent.status) ? intent.status.unavailable : undefined;
          if (typeof unavailable !== 'string' || unavailable.length === 0) errors.push(`${w}: a status intent must have status.unavailable copy (rule 1.3)`);
        }

        // §1.4 built-ins may be overridden, but they stay meta
        if (typeof intent.id === 'string' && (BUILTIN_INTENT_IDS as readonly string[]).includes(intent.id) && intent.family !== 'meta') {
          errors.push(`${w}: the built-in intent "${intent.id}" must have family "meta" (rule 1.4)`);
        }

        // An intent with nothing to compile can never appear in a split (§2.5) — fail here, loudly.
        const nTemplates = Array.isArray(intent.templates) ? intent.templates.length : 0;
        const nParaphrases = Array.isArray(intent.paraphrases) ? intent.paraphrases.length : 0;
        const isBuiltin = typeof intent.id === 'string' && (BUILTIN_INTENT_IDS as readonly string[]).includes(intent.id);
        if (nTemplates + nParaphrases === 0 && !isBuiltin) {
          errors.push(`${w}: needs at least one template or paraphrase to be trainable`);
        }
      });
    }
  }

  // ── heldout ──
  if ('heldout' in raw && raw.heldout !== undefined) {
    if (!Array.isArray(raw.heldout)) errors.push('heldout: must be an array of { utterance, intent }');
    else {
      const known = new Set<string>([...intentIds, ...BUILTIN_INTENT_IDS]);
      raw.heldout.forEach((ex, i) => {
        const w = `heldout[${i}]`;
        if (!isObj(ex)) {
          errors.push(`${w}: must be an object { utterance, intent }`);
          return;
        }
        for (const k of extraKeys(ex, ['utterance', 'intent'])) errors.push(`${w}: unknown property "${k}"`);
        if (!('utterance' in ex)) errors.push(`${w}: missing required "utterance"`);
        else checkString(errors, `${w}.utterance`, ex.utterance, { minLength: 1, maxLength: 300 });
        if (!('intent' in ex)) errors.push(`${w}: missing required "intent"`);
        else if (typeof ex.intent !== 'string') errors.push(`${w}.intent: must be a string`);
        else if (!known.has(ex.intent)) errors.push(`${w}: intent "${ex.intent}" is not declared in intents[]`);
      });
    }
  }

  if (errors.length > 0) return { ok: false, errors };
  return { ok: true, errors: [], registry: normalizeRegistry(raw as unknown as Registry) };
}

/**
 * Fill defaults and append the built-in meta intents (§1.4/§1.6). Idempotent: normalizing an
 * already-normalized registry yields an equal object, which is what makes `registryHash`
 * declaration-independent (§1.7).
 */
export function normalizeRegistry(raw: Registry | NormalizedRegistry): NormalizedRegistry {
  const intents: NormalizedIntentDef[] = raw.intents.map((i) => {
    const norm: NormalizedIntentDef = {
      id: i.id,
      family: i.family,
      label: i.label,
      slots: [...(i.slots ?? [])],
      keywords: i.keywords ?? '',
      templates: [...(i.templates ?? [])],
      paraphrases: [...(i.paraphrases ?? [])],
    };
    if (i.answer) norm.answer = { steps: [...i.answer.steps], ...(i.answer.links ? { links: i.answer.links.map((l) => ({ ...l })) } : {}) };
    if (i.status) norm.status = { unavailable: i.status.unavailable, ...(i.status.links ? { links: i.status.links.map((l) => ({ ...l })) } : {}) };
    // A declared built-in with no utterances of its own keeps the built-in utterance sets.
    if ((BUILTIN_INTENT_IDS as readonly string[]).includes(i.id) && norm.templates.length + norm.paraphrases.length === 0) {
      const b = builtinIntent(i.id as 'greeting' | 'out_of_domain');
      norm.templates = b.templates;
      norm.paraphrases = b.paraphrases;
      if (!norm.keywords) norm.keywords = b.keywords;
    }
    return norm;
  });
  for (const id of BUILTIN_INTENT_IDS) {
    if (!intents.some((i) => i.id === id)) intents.push(builtinIntent(id));
  }
  return {
    registryVersion: '1',
    app: {
      slug: raw.app.slug,
      name: raw.app.name,
      assistantName: raw.app.assistantName ?? `${raw.app.name} Assistant`,
    },
    slots: Object.fromEntries(
      Object.entries(raw.slots).map(([name, def]) => [
        name,
        { ...(def.label !== undefined ? { label: def.label } : {}), vocab: def.vocab.map((v) => ({ id: v.id, label: v.label })) },
      ]),
    ),
    intents,
    heldout: (raw.heldout ?? []).map((h) => ({ utterance: h.utterance, intent: h.intent })),
  };
}
