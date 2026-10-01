/**
 * The tokenizer — the byte-identical contract (docs/CONTRACTS.md §3).
 *
 * Three copies of this algorithm exist on purpose (this TypeScript core, the Python trainer,
 * the browser widget). Tests pin all three against `tests/fixtures/tokenizer.parity.json`.
 *
 *   1. Lowercase (when `lower`).
 *   2. Split into maximal runs of Unicode Letter/Number codepoints; everything else separates.
 *   3. Map each word to `vocab[word]`, else `unkId`.
 *   4. Truncate to `maxLen`, pad with `padId`.
 */

import { readFileSync } from 'node:fs';

export interface TokenizerConfig {
  version: string;
  lower: boolean;
  maxLen: number;
  padId: number;
  unkId: number;
  padToken: string;
  unkToken: string;
  vocab: Record<string, number>;
}

/** The word-character class. Exported by name: a drift test in the widget reads it. */
export const WORD_CHAR_RE = /[\p{L}\p{N}]/u;

/** True for any Unicode Letter or Number codepoint. */
export function isWordChar(ch: string): boolean {
  return WORD_CHAR_RE.test(ch);
}

/** Maximal runs of word characters, iterating by codepoint so surrogate pairs never split. */
export function wordSplit(text: string): string[] {
  const tokens: string[] = [];
  let cur = '';
  for (const ch of text) {
    if (isWordChar(ch)) {
      cur += ch;
    } else if (cur) {
      tokens.push(cur);
      cur = '';
    }
  }
  if (cur) tokens.push(cur);
  return tokens;
}

export interface OffsetToken {
  text: string;
  /** UTF-16 code-unit offsets into the ORIGINAL (un-lowercased) string, `[start, end)`. */
  start: number;
  end: number;
}

/** Same split as `wordSplit`, keeping each token's offsets into the original-cased text (§3.5). */
export function wordSplitWithOffsets(text: string): OffsetToken[] {
  const tokens: OffsetToken[] = [];
  let cur = '';
  let start = -1;
  let i = 0;
  for (const ch of text) {
    if (isWordChar(ch)) {
      if (start === -1) start = i;
      cur += ch;
    } else if (cur) {
      tokens.push({ text: cur, start, end: i });
      cur = '';
      start = -1;
    }
    i += ch.length;
  }
  if (cur) tokens.push({ text: cur, start, end: i });
  return tokens;
}

/** text -> exactly `config.maxLen` ids (truncated / padded), OOV words -> `config.unkId`. */
export function tokenize(text: string, config: TokenizerConfig): number[] {
  const lowered = config.lower ? text.toLowerCase() : text;
  const words = wordSplit(lowered);
  // Own-property lookup only: a word like "constructor" must map to unkId, never to an
  // inherited Object.prototype member (the Python side is `dict.get`, the widget hasOwnProperty).
  const ids = words
    .slice(0, config.maxLen)
    .map((w) => (Object.prototype.hasOwnProperty.call(config.vocab, w) ? config.vocab[w] : config.unkId));
  while (ids.length < config.maxLen) ids.push(config.padId);
  return ids;
}

/** Load a `*.tokenizer.json` (§3.3) from disk, checking the shape. */
export function loadTokenizerConfig(path: string): TokenizerConfig {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<TokenizerConfig> | null;
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    typeof parsed.vocab !== 'object' ||
    parsed.vocab === null ||
    typeof parsed.maxLen !== 'number' ||
    typeof parsed.padId !== 'number' ||
    typeof parsed.unkId !== 'number' ||
    typeof parsed.lower !== 'boolean'
  ) {
    throw new Error(`loadTokenizerConfig: malformed tokenizer config at ${path}`);
  }
  return parsed as TokenizerConfig;
}
