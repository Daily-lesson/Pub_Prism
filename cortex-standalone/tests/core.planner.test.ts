import { describe, expect, it } from 'vitest';
import { plan, normalizeRegistry, GREETING_COPY, OUT_OF_DOMAIN_COPY, FALLBACK_COPY, type Registry } from '../core/index';

function reg(): Registry {
  return {
    registryVersion: '1',
    app: { slug: 'p', name: 'P' },
    slots: { region: { vocab: [{ id: 'north', label: 'North campus' }] } },
    intents: [
      {
        id: 'open_reports',
        family: 'howto',
        label: 'Find the reports',
        slots: ['region'],
        templates: ['x'],
        answer: {
          steps: ['Open Reports from the left nav.', 'Pick a region.'],
          links: [
            { label: 'Reports', target: { route: '/reports' }, need: 'reports' },
            { label: 'Help', target: { route: '/help' } },
          ],
        },
      },
      {
        id: 'report_count',
        family: 'status',
        label: 'How many reports',
        templates: ['y'],
        status: { unavailable: "I can't read report counts here.", links: [{ label: 'Reports', target: { route: '/reports' }, need: 'reports' }] },
      },
    ],
  };
}
const R = normalizeRegistry(reg());

describe('planner (§5.4–§5.5)', () => {
  it('howto: numbered steps joined by a space, links become chips', async () => {
    const r = await plan(R, 'open_reports', []);
    expect(r.answerText).toBe('1. Open Reports from the left nav. 2. Pick a region.');
    expect(r.chips).toEqual([
      { label: 'Reports', target: { route: '/reports' } },
      { label: 'Help', target: { route: '/help' } },
    ]);
    expect(r.evidence).toEqual({ intent: 'open_reports', plannerTemplateId: 'planner.howto.open_reports', registryEntriesUsed: ['intent:open_reports'] });
  });

  it('howto: slot prefix when a span was tagged; slot entry cites resolvedId', async () => {
    const r = await plan(R, 'open_reports', [{ name: 'region', value: 'north campus', start: 0, end: 12, resolvedId: 'north' }]);
    expect(r.answerText.startsWith('Noted — you mentioned "north campus". 1. Open')).toBe(true);
    expect(r.evidence.registryEntriesUsed).toEqual(['intent:open_reports', 'slot.region:north']);
  });

  it('need gate drops a refused link (never renders it disabled); absent isAllowed = allow all', async () => {
    const gated = await plan(R, 'open_reports', [], { isAllowed: (need) => need !== 'reports' });
    expect(gated.chips).toEqual([{ label: 'Help', target: { route: '/help' } }]);
    const open = await plan(R, 'open_reports', [], {});
    expect(open.chips.length).toBe(2);
  });

  it('status: provider string wins; non-string/undefined/throw ⇒ unavailable copy', async () => {
    const withProvider = await plan(R, 'report_count', [], { status: (id, slots) => `${id}:${slots.length}:12 open` });
    expect(withProvider.answerText).toBe('report_count:0:12 open');
    expect(withProvider.evidence.plannerTemplateId).toBe('planner.status.report_count');
    const undef = await plan(R, 'report_count', [], { status: () => undefined });
    expect(undef.answerText).toBe("I can't read report counts here.");
    const num = await plan(R, 'report_count', [], { status: (() => 12) as unknown as () => string });
    expect(num.answerText).toBe("I can't read report counts here.");
    const throws = await plan(R, 'report_count', [], {
      status: () => {
        throw new Error('boom');
      },
    });
    expect(throws.answerText).toBe("I can't read report counts here.");
    const none = await plan(R, 'report_count', []);
    expect(none.answerText).toBe("I can't read report counts here.");
    expect(none.chips).toEqual([{ label: 'Reports', target: { route: '/reports' } }]);
  });

  it('status: async provider is awaited', async () => {
    const r = await plan(R, 'report_count', [], { status: async () => 'three open' });
    expect(r.answerText).toBe('three open');
  });

  it('meta + fallback copy and plannerTemplateIds', async () => {
    const g = await plan(R, 'greeting', []);
    expect(g.answerText).toBe(GREETING_COPY);
    expect(g.evidence.plannerTemplateId).toBe('planner.meta.greeting');
    const o = await plan(R, 'out_of_domain', []);
    expect(o.answerText).toBe(OUT_OF_DOMAIN_COPY);
    expect(o.evidence.plannerTemplateId).toBe('planner.meta.out_of_domain');
    const f = await plan(R, null, []);
    expect(f.answerText.startsWith(FALLBACK_COPY)).toBe(true);
    expect(f.answerText).toContain('Find the reports');
    expect(f.answerText).toContain('How many reports');
    expect(f.chips).toEqual([]);
    expect(f.evidence).toEqual({ intent: null, plannerTemplateId: 'planner.fallback', registryEntriesUsed: [] });
    const unknown = await plan(R, 'no_such_intent', []);
    expect(unknown.evidence.plannerTemplateId).toBe('planner.fallback');
  });
});
