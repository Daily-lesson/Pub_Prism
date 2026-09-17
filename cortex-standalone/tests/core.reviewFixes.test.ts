/**
 * Regression pins for the adversarial-review fix batch (correctness MF-1, SF-5/6/7, nit 12;
 * security nit 7). Each case reproduces the reviewer's exact finding and would fail on the
 * pre-fix code.
 */
import { describe, expect, it } from 'vitest';
import { autoDetectSlots } from '../core/dataset/paraphrases';
import { normalizeRegistry, type Registry } from '../core/registry';
import { matchKeywords, buildKeywordBags } from '../core/infer/keyword';
import { tokenize, type TokenizerConfig } from '../core/tokenizer';
import { resolveLadder } from '../core/infer/ladder';
import { plan } from '../core/infer/planner';

const R = normalizeRegistry({
  registryVersion: '1',
  app: { slug: 's', name: 'S' },
  slots: {
    department: { vocab: [{ id: 'it', label: 'IT' }, { id: 'hr', label: 'HR' }, { id: 'fac', label: 'Facilities' }] },
    region: { vocab: [{ id: 'north', label: 'North campus' }] },
  },
  intents: [
    { id: 'alpha', family: 'howto', label: 'Alpha', keywords: 'reports', slots: ['department', 'region'], templates: ['alpha'], answer: { steps: ['a'] } },
    { id: 'beta', family: 'howto', label: 'Beta', keywords: 'rooms booking', slots: [], templates: ['beta'], answer: { steps: ['b'] } },
  ],
} as Registry);

describe('autoDetectSlots matches whole words only (§2.3, review MF-1)', () => {
  it('a two-letter label never tags the inside of a word', () => {
    for (const u of ['anything still waiting in the queue', 'there is a leak in the kitchen', 'submit an issue', 'is the queue long for editing']) {
      expect(autoDetectSlots(R, u, ['department', 'region'])).toEqual([]);
    }
  });
  it('still tags the label as a standalone word, at the first whole-word occurrence', () => {
    // NB "it" (the pronoun) IS a whole-word, case-insensitive match for the label "IT" — an
    // inherent property of label matching, so the fixture avoids the pronoun on purpose.
    expect(autoDetectSlots(R, 'submit that to IT today', ['department'])).toEqual([{ name: 'department', value: 'IT', start: 15, end: 17 }]);
    expect(autoDetectSlots(R, 'tell hr about the north campus', ['department', 'region'])).toEqual([
      { name: 'department', value: 'hr', start: 5, end: 7 },
      { name: 'region', value: 'north campus', start: 18, end: 30 },
    ]);
  });
  it('a label at the very start or end of the utterance counts', () => {
    expect(autoDetectSlots(R, 'IT is slow', ['department'])[0]?.value).toBe('IT');
    expect(autoDetectSlots(R, 'escalate to hr', ['department'])[0]?.value).toBe('hr');
  });
});

describe('keyword rung uses set semantics (§5.2, review SF-5)', () => {
  it('a repeated query word scores once — matches the widget', () => {
    const bags = buildKeywordBags(R);
    expect(matchKeywords('reports reports rooms booking', bags)?.intent).toBe('beta');
    expect(matchKeywords('reports', bags)?.intent).toBe('alpha');
  });
});

describe('tokenize reads own vocab keys only (review SF-7)', () => {
  it('"constructor" maps to unkId, never to Object.prototype.constructor', () => {
    const cfg: TokenizerConfig = { version: '1', lower: true, maxLen: 4, padId: 0, unkId: 1, padToken: '<pad>', unkToken: '<unk>', vocab: { open: 5, the: 6 } };
    expect(tokenize('open the constructor', cfg)).toEqual([5, 6, 1, 0]);
    expect(tokenize('open the toString', cfg)).toEqual([5, 6, 1, 0]);
  });
});

describe('keyword-rung slots + plural slot prefix (§5.5, review SF-6 / nit 12)', () => {
  it('the keyword rung reports whole-word vocab matches when detectSlots is supplied, none otherwise', async () => {
    const bags = buildKeywordBags(R);
    const bare = await resolveLadder('reports for hr in the north campus', { bags });
    expect(bare.rung).toBe('keyword');
    expect(bare.slots).toEqual([]);
    const withSlots = await resolveLadder('reports for hr in the north campus', {
      bags,
      detectSlots: (q, intent) => autoDetectSlots(R, q, R.intents.find((i) => i.id === intent)!.slots),
    });
    expect(withSlots.slots.map((s) => `${s.name}:${s.value}`)).toEqual(['department:hr', 'region:north campus']);
    const planned = await plan(R, withSlots.intent, withSlots.slots, {});
    expect(planned.answerText.startsWith('Noted — you mentioned "hr", "north campus". ')).toBe(true);
  });
});
