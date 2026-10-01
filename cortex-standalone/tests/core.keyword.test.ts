import { describe, expect, it } from 'vitest';
import { STOPWORDS, keywordTokens, buildKeywordBags, matchKeywords, normalizeRegistry, type Registry } from '../core/index';

const CONTRACT_STOPWORDS =
  'the a an and or of to in on for with is are was were be been do does did how what where when which who whom why can could would should will shall may might must i me my we our you your it its this that these those there here from by at as into onto than then so if not no yes please just about over under up down out off again more most some any all';

function reg(): Registry {
  return {
    registryVersion: '1',
    app: { slug: 'k', name: 'K' },
    slots: {},
    intents: [
      { id: 'alpha', family: 'howto', label: 'Alpha thing', keywords: 'reports chart', templates: ['a'], answer: { steps: ['x'] } },
      { id: 'beta', family: 'howto', label: 'Beta thing', keywords: 'reports export', templates: ['b'], answer: { steps: ['x'] } },
      { id: 'gamma', family: 'status', label: 'Gamma count', keywords: 'export chart tables', templates: ['c'], status: { unavailable: 'n' } },
    ],
  };
}

describe('keyword rung (§5.2–§5.3)', () => {
  it('STOPWORDS is exactly the contract list, in order', () => {
    expect([...STOPWORDS]).toEqual(CONTRACT_STOPWORDS.split(' '));
  });

  it('keywordTokens: lowercase, non-alnum split, length ≥ 3, stopwords removed', () => {
    expect(keywordTokens('How DO I find the Reports, please? #42 ok')).toEqual(['find', 'reports']);
    expect(keywordTokens('')).toEqual([]);
  });

  it('bags = keywords ∪ label words; out_of_domain has an empty bag', () => {
    const bags = buildKeywordBags(normalizeRegistry(reg()));
    expect(bags.map((b) => b.intent)).toEqual(['alpha', 'beta', 'gamma', 'greeting', 'out_of_domain']);
    expect([...bags[0].words].sort()).toEqual(['alpha', 'chart', 'reports', 'thing']);
    expect(bags[4].words.size).toBe(0);
  });

  it('highest score wins; ties break on taxonomy order; 0 ⇒ null', () => {
    const bags = buildKeywordBags(normalizeRegistry(reg()));
    expect(matchKeywords('export the chart tables', bags)).toEqual({ intent: 'gamma', score: 3 });
    expect(matchKeywords('reports', bags)).toEqual({ intent: 'alpha', score: 1 });
    expect(matchKeywords('the and of', bags)).toBeNull();
    expect(matchKeywords('unrelated words entirely', bags)).toBeNull();
    expect(matchKeywords('weather joke', bags)).toBeNull();
  });
});
