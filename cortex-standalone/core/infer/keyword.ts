/**
 * The keyword rung (docs/CONTRACTS.md §5.2–§5.3): whole-word overlap between the query and
 * each intent's keyword bag. Highest score wins; ties break on taxonomy order; 0 is no match.
 */

import type { NormalizedRegistry } from '../registry/types';

/** Exact, shared by every copy (§5.2). Order is load-bearing for the widget drift test. */
export const STOPWORDS: readonly string[] = [
  'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'is', 'are', 'was', 'were', 'be', 'been',
  'do', 'does', 'did', 'how', 'what', 'where', 'when', 'which', 'who', 'whom', 'why', 'can', 'could', 'would',
  'should', 'will', 'shall', 'may', 'might', 'must', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'it', 'its',
  'this', 'that', 'these', 'those', 'there', 'here', 'from', 'by', 'at', 'as', 'into', 'onto', 'than', 'then',
  'so', 'if', 'not', 'no', 'yes', 'please', 'just', 'about', 'over', 'under', 'up', 'down', 'out', 'off',
  'again', 'more', 'most', 'some', 'any', 'all',
];

const STOPWORD_SET: ReadonlySet<string> = new Set(STOPWORDS);

/** lowercase → split on `/[^a-z0-9]+/` → keep length ≥ 3 and not a stopword. */
export function keywordTokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length >= 3 && !STOPWORD_SET.has(t));
}

export interface KeywordBag {
  intent: string;
  words: Set<string>;
}

/** One bag per intent in taxonomy order: `keywords` ∪ `label` words. `out_of_domain` gets an
 * empty bag — it is the "nothing matched" bucket, never a keyword target (§5.1). */
export function buildKeywordBags(registry: NormalizedRegistry): KeywordBag[] {
  return registry.intents.map((intent) => ({
    intent: intent.id,
    words: intent.id === 'out_of_domain' ? new Set<string>() : new Set([...keywordTokens(intent.keywords), ...keywordTokens(intent.label)]),
  }));
}

export function matchKeywords(query: string, bags: readonly KeywordBag[]): { intent: string; score: number } | null {
  // Set semantics (§5.2): a repeated query word counts once, so "reports reports" scores 1.
  const tokens = [...new Set(keywordTokens(query))];
  if (!tokens.length) return null;
  let best: { intent: string; score: number } | null = null;
  for (const bag of bags) {
    let score = 0;
    for (const tok of tokens) if (bag.words.has(tok)) score++;
    if (score > 0 && (!best || score > best.score)) best = { intent: bag.intent, score };
  }
  return best;
}
