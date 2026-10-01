import { describe, expect, it } from 'vitest';
import path from 'node:path';
import {
  validateRegistry,
  normalizeRegistry,
  registryHash,
  canonicalJson,
  loadRegistryFile,
  RegistryValidationError,
  BUILTIN_META_INTENTS,
  type Registry,
} from '../core/index';

const EXAMPLE = path.resolve(__dirname, '..', 'registry', 'examples', 'ops-dashboard.registry.json');

function minimal(): Registry {
  return {
    registryVersion: '1',
    app: { slug: 'demo-app', name: 'Demo App' },
    slots: {
      region: { label: 'Region', vocab: [{ id: 'north', label: 'North campus' }, { id: 'south', label: 'South campus' }] },
      empty: { vocab: [] },
    },
    intents: [
      {
        id: 'open_reports',
        family: 'howto',
        label: 'Find the reports',
        slots: ['region'],
        keywords: 'reports report',
        templates: ['how do i find the reports', 'show me the {region} reports'],
        paraphrases: ['where are my numbers'],
        answer: { steps: ['Open Reports.'], links: [{ label: 'Reports', target: { route: '/reports' }, need: 'reports' }] },
      },
      {
        id: 'report_count',
        family: 'status',
        label: 'How many reports',
        templates: ['how many reports are there'],
        status: { unavailable: 'Cannot read counts here.' },
      },
    ],
  };
}

function errorsFor(mutate: (r: Registry) => void): string[] {
  const r = minimal();
  mutate(r);
  const v = validateRegistry(r);
  expect(v.ok).toBe(false);
  return v.errors;
}

describe('validateRegistry — §1 rules', () => {
  it('accepts the example registry and the minimal one', () => {
    expect(validateRegistry(minimal()).ok).toBe(true);
    const reg = loadRegistryFile(EXAMPLE);
    expect(reg.intents.map((i) => i.id)).toContain('greeting');
    expect(reg.intents.map((i) => i.id)).toContain('out_of_domain');
  });

  it('rejects a non-object with one readable line', () => {
    expect(validateRegistry('nope')).toEqual({ ok: false, errors: ['registry: must be a JSON object'] });
  });

  it('requires registryVersion "1" and the four required keys', () => {
    const v = validateRegistry({});
    expect(v.ok).toBe(false);
    for (const k of ['registryVersion', 'app', 'slots', 'intents']) expect(v.errors.join('\n')).toContain(`missing required "${k}"`);
    expect(errorsFor((r) => ((r as { registryVersion: string }).registryVersion = '2'))).toContain('registryVersion: must be "1" (got "2")');
  });

  it('refuses unknown properties at every level (additionalProperties: false)', () => {
    expect(errorsFor((r) => ((r as unknown as Record<string, unknown>).extra = 1))).toContain('registry: unknown top-level property "extra"');
    expect(errorsFor((r) => ((r.app as unknown as Record<string, unknown>).logo = 'x'))).toContain('app: unknown property "logo"');
    expect(errorsFor((r) => ((r.intents[0] as unknown as Record<string, unknown>).foo = 1))).toContain('intent "open_reports": unknown property "foo"');
    expect(errorsFor((r) => ((r.intents[0].answer!.links![0] as unknown as Record<string, unknown>).icon = 'x'))).toContain(
      'intent "open_reports".answer.links[0]: unknown property "icon"',
    );
  });

  it('checks app.slug pattern and name length', () => {
    expect(errorsFor((r) => (r.app.slug = 'Bad Slug'))[0]).toContain('app.slug: must match');
    expect(errorsFor((r) => (r.app.name = ''))[0]).toContain('app.name: must be at least 1');
  });

  it('checks slot names, vocab shape and duplicate vocab ids', () => {
    expect(errorsFor((r) => (r.slots['Bad-Name'] = { vocab: [] })).join('\n')).toContain('slot "Bad-Name": name must match');
    expect(errorsFor((r) => ((r.slots.region as unknown as Record<string, unknown>).vocab = 'x')).join('\n')).toContain('slot "region".vocab: must be an array');
    expect(errorsFor((r) => r.slots.region.vocab.push({ id: 'north', label: 'Again' })).join('\n')).toContain('duplicate vocab id "north" in slot "region"');
  });

  it('1.1 — intent ids are unique and families are howto|status|meta', () => {
    expect(errorsFor((r) => r.intents.push({ ...r.intents[0] })).join('\n')).toContain('duplicate intent id "open_reports" (rule 1.1)');
    expect(errorsFor((r) => ((r.intents[0] as { family: string }).family = 'chat')).join('\n')).toContain(
      'intent "open_reports".family: must be one of howto|status|meta (got "chat")',
    );
    expect(errorsFor((r) => (r.intents[0].id = 'Bad-Id')).join('\n')).toContain('intent "Bad-Id".id: must match ^[a-z][a-z0-9_]*$');
  });

  it('1.2 — placeholders must name a declared slot with a non-empty vocab', () => {
    expect(errorsFor((r) => r.intents[0].templates!.push('open {period} reports')).join('\n')).toContain(
      'intent "open_reports".templates[2]: placeholder {period} is not listed in the intent\'s slots (rule 1.2)',
    );
    expect(errorsFor((r) => (r.intents[0].slots = ['ghost'])).join('\n')).toContain('intent "open_reports": slot "ghost" is not declared in registry.slots (rule 1.2)');
    expect(
      errorsFor((r) => {
        r.intents[0].slots = ['region', 'empty'];
        r.intents[0].templates!.push('show {empty}');
      }).join('\n'),
    ).toContain('placeholder {empty} names slot "empty" whose vocab is empty (rule 1.2)');
  });

  it('1.3 — howto needs answer.steps, status needs status.unavailable', () => {
    expect(errorsFor((r) => delete r.intents[0].answer).join('\n')).toContain('intent "open_reports": a howto intent must have answer.steps with at least one step (rule 1.3)');
    expect(errorsFor((r) => (r.intents[0].answer = { steps: [] })).join('\n')).toContain('a howto intent must have answer.steps');
    expect(errorsFor((r) => delete r.intents[1].status).join('\n')).toContain('intent "report_count": a status intent must have status.unavailable copy (rule 1.3)');
    const meta = minimal();
    meta.intents.push({ id: 'chit_chat', family: 'meta', label: 'Chit chat', templates: ['how are you'] });
    expect(validateRegistry(meta).ok).toBe(true);
  });

  it('1.4 — a declared built-in must stay meta', () => {
    expect(errorsFor((r) => r.intents.push({ id: 'greeting', family: 'howto', label: 'Hi', templates: ['hi'], answer: { steps: ['x'] } })).join('\n')).toContain(
      'the built-in intent "greeting" must have family "meta" (rule 1.4)',
    );
  });

  it('links: max 8, label/target required, need optional string', () => {
    expect(
      errorsFor((r) => (r.intents[0].answer!.links = Array.from({ length: 9 }, (_, i) => ({ label: `L${i}`, target: i })))).join('\n'),
    ).toContain('intent "open_reports".answer.links: at most 8 links allowed (got 9)');
    expect(errorsFor((r) => (r.intents[0].answer!.links = [{ label: 'x' } as never])).join('\n')).toContain('answer.links[0]: missing required "target"');
  });

  it('heldout entries must reference declared intents (built-ins count)', () => {
    expect(errorsFor((r) => (r.heldout = [{ utterance: 'hey', intent: 'nope' }])).join('\n')).toContain('heldout[0]: intent "nope" is not declared in intents[]');
    const ok = minimal();
    ok.heldout = [{ utterance: 'hey', intent: 'greeting' }];
    expect(validateRegistry(ok).ok).toBe(true);
  });

  it('an intent with no templates and no paraphrases is refused (it could never be trained)', () => {
    expect(errorsFor((r) => r.intents.push({ id: 'silent', family: 'howto', label: 'Silent', answer: { steps: ['x'] } })).join('\n')).toContain(
      'intent "silent": needs at least one template or paraphrase',
    );
  });

  it('loadRegistryFile throws with every error line', () => {
    const bad = path.resolve(__dirname, 'fixtures', 'does-not-exist.registry.json');
    expect(() => loadRegistryFile(bad)).toThrow(RegistryValidationError);
  });
});

describe('normalizeRegistry + builtins (§1.4, §1.6)', () => {
  it('fills defaults and appends greeting + out_of_domain last', () => {
    const n = normalizeRegistry(minimal());
    expect(n.intents.map((i) => i.id)).toEqual(['open_reports', 'report_count', 'greeting', 'out_of_domain']);
    expect(n.intents[1].slots).toEqual([]);
    expect(n.intents[1].keywords).toBe('');
    expect(n.intents[1].paraphrases).toEqual([]);
    expect(n.heldout).toEqual([]);
    expect(n.app.assistantName).toBe('Demo App Assistant');
  });

  it('built-ins carry ≥25 templates and ≥25 paraphrases each', () => {
    for (const id of ['greeting', 'out_of_domain'] as const) {
      expect(BUILTIN_META_INTENTS[id].templates.length).toBeGreaterThanOrEqual(25);
      expect(BUILTIN_META_INTENTS[id].paraphrases.length).toBeGreaterThanOrEqual(25);
      expect(BUILTIN_META_INTENTS[id].family).toBe('meta');
    }
  });

  it('a declared built-in keeps its position and its own utterances', () => {
    const r = minimal();
    r.intents.unshift({ id: 'greeting', family: 'meta', label: 'Hello', templates: ['hello', 'hi'] });
    const n = normalizeRegistry(r);
    expect(n.intents.map((i) => i.id)).toEqual(['greeting', 'open_reports', 'report_count', 'out_of_domain']);
    expect(n.intents[0].templates).toEqual(['hello', 'hi']);
  });

  it('a declared built-in with no utterances inherits the built-in sets', () => {
    const r = minimal();
    r.intents.push({ id: 'out_of_domain', family: 'meta', label: 'Off topic' });
    const n = normalizeRegistry(r);
    const ood = n.intents.find((i) => i.id === 'out_of_domain')!;
    expect(ood.templates).toEqual(BUILTIN_META_INTENTS.out_of_domain.templates);
    expect(ood.label).toBe('Off topic');
  });

  it('is idempotent', () => {
    const once = normalizeRegistry(minimal());
    expect(normalizeRegistry(once)).toEqual(once);
  });
});

describe('registryHash (§1.7)', () => {
  it('is identical whether or not the built-ins were declared verbatim', () => {
    const undeclared = minimal();
    const declared = minimal();
    declared.intents.push({ ...BUILTIN_META_INTENTS.greeting }, { ...BUILTIN_META_INTENTS.out_of_domain });
    expect(registryHash(declared)).toBe(registryHash(undeclared));
  });

  it('is identical whether or not defaults were spelled out', () => {
    const implicit = minimal();
    const explicit = minimal();
    explicit.intents[1].slots = [];
    explicit.intents[1].keywords = '';
    explicit.intents[1].paraphrases = [];
    explicit.heldout = [];
    explicit.app.assistantName = 'Demo App Assistant';
    expect(registryHash(explicit)).toBe(registryHash(implicit));
  });

  it('changes when content changes, and is a sha256 hex', () => {
    const a = registryHash(minimal());
    const b = minimal();
    b.intents[0].answer!.steps[0] = 'Open Reports from the nav.';
    expect(registryHash(b)).not.toBe(a);
    expect(a).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('canonicalJson', () => {
  it('sorts keys recursively and keeps array order', () => {
    expect(canonicalJson({ b: 1, a: { z: [3, { y: 1, x: 2 }], m: null } })).toBe('{"a":{"m":null,"z":[3,{"x":2,"y":1}]},"b":1}');
    expect(canonicalJson([2, 1])).toBe('[2,1]');
    expect(canonicalJson('s')).toBe('"s"');
  });
});
