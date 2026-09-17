import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { tokenize, wordSplit, wordSplitWithOffsets, WORD_CHAR_RE, isWordChar, type TokenizerConfig } from '../core/index';

const fixture = JSON.parse(fs.readFileSync(path.resolve(__dirname, 'fixtures', 'tokenizer.parity.json'), 'utf8')) as {
  config: TokenizerConfig;
  cases: Array<{ text: string; words: string[]; ids: number[] }>;
};

describe('tokenizer (§3)', () => {
  it('WORD_CHAR_RE is exactly /[\\p{L}\\p{N}]/u', () => {
    expect(WORD_CHAR_RE.source).toBe('[\\p{L}\\p{N}]');
    expect(WORD_CHAR_RE.flags).toBe('u');
    expect(isWordChar('é')).toBe(true);
    expect(isWordChar('7')).toBe(true);
    expect(isWordChar('-')).toBe(false);
    expect(isWordChar('😀')).toBe(false);
  });

  it('fixture has ≥25 cases and every case matches', () => {
    expect(fixture.cases.length).toBeGreaterThanOrEqual(25);
    for (const c of fixture.cases) {
      expect(wordSplit(fixture.config.lower ? c.text.toLowerCase() : c.text), c.text).toEqual(c.words);
      const ids = tokenize(c.text, fixture.config);
      expect(ids, c.text).toEqual(c.ids);
      expect(ids.length).toBe(fixture.config.maxLen);
    }
  });

  it('hand-checked cases', () => {
    const cfg = fixture.config;
    expect(tokenize('', cfg)).toEqual(new Array(cfg.maxLen).fill(cfg.padId));
    expect(tokenize('Café STRASSE', cfg).slice(0, 3)).toEqual([cfg.vocab['café'], cfg.unkId, cfg.padId]);
    expect(tokenize('ticket #3 is open', cfg).slice(0, 5)).toEqual([26, 24, 28, 27, 0]);
    expect(tokenize('the '.repeat(14), cfg)).toEqual(new Array(12).fill(6));
    expect(wordSplit('hello 😀 thanks')).toEqual(['hello', 'thanks']);
  });

  it('wordSplitWithOffsets keeps original-cased offsets; casing never moves boundaries', () => {
    const text = 'Show me the North Campus, now!';
    const toks = wordSplitWithOffsets(text);
    expect(toks.map((t) => t.text)).toEqual(['Show', 'me', 'the', 'North', 'Campus', 'now']);
    for (const t of toks) expect(text.slice(t.start, t.end)).toBe(t.text);
    expect(toks.map((t) => t.text.toLowerCase())).toEqual(wordSplit(text.toLowerCase()));
    const emoji = wordSplitWithOffsets('a 😀 b');
    expect(emoji).toEqual([
      { text: 'a', start: 0, end: 1 },
      { text: 'b', start: 5, end: 6 },
    ]);
  });
});
