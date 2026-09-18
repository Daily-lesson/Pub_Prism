/* tests/widgetDrift.test.ts — CONTRACTS.md §6.6: the widget's copies of STOPWORDS, the
 * tokenizer's word-char regex, the keyword-token split regex and the confidence default are
 * read out of widget/cortex-widget.js AS TEXT and compared byte-for-byte to the core's exports
 * (`STOPWORDS` from core/infer/keyword, `WORD_CHAR_RE` from core/tokenizer) and to the literal
 * list in docs/CONTRACTS.md §5.2. When the core files are not present the core comparison is
 * reported as NOT RUN (skipped with a reason) rather than weakened; the CONTRACTS comparison
 * always runs.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';

const PKG = path.resolve(__dirname, '..');
const WIDGET = path.join(PKG, 'widget', 'cortex-widget.js');
const CONTRACTS = path.join(PKG, 'docs', 'CONTRACTS.md');

function resolveCore(rel: string): string | null {
  for (const cand of [rel + '.ts', rel + '.js', rel + '/index.ts', rel + '/index.js', rel + '.mjs']) {
    const p = path.join(PKG, cand);
    if (fs.existsSync(p)) return p;
  }
  return null;
}
const CORE_KEYWORD = resolveCore('core/infer/keyword');
const CORE_TOKENIZER = resolveCore('core/tokenizer');
const CORE_PRESENT = !!(CORE_KEYWORD && CORE_TOKENIZER);

const src = fs.readFileSync(WIDGET, 'utf8');
function extract(re: RegExp, what: string): RegExpMatchArray {
  const m = src.match(re);
  if (!m) throw new Error('widget/cortex-widget.js: could not find ' + what);
  return m;
}
const widget = {
  stopwords: JSON.parse(extract(/^\s*var STOPWORDS = (\[[^\]]*\]);/m, 'the STOPWORDS array literal')[1]) as string[],
  wordCharSource: extract(/^\s*var WORD_CHAR_RE = \/(.+?)\/([a-z]*);/m, 'WORD_CHAR_RE')[1],
  wordCharFlags: extract(/^\s*var WORD_CHAR_RE = \/(.+?)\/([a-z]*);/m, 'WORD_CHAR_RE')[2],
  keywordSplitSource: extract(/^\s*var KEYWORD_SPLIT_RE = \/(.+?)\/([a-z]*);/m, 'KEYWORD_SPLIT_RE')[1],
  keywordSplitFlags: extract(/^\s*var KEYWORD_SPLIT_RE = \/(.+?)\/([a-z]*);/m, 'KEYWORD_SPLIT_RE')[2],
  confidenceGate: Number(extract(/^\s*var DEFAULT_CONFIDENCE_GATE = ([0-9.]+);/m, 'DEFAULT_CONFIDENCE_GATE')[1]),
  keywordMinLen: Number(extract(/^\s*var KEYWORD_MIN_LEN = (\d+);/m, 'KEYWORD_MIN_LEN')[1]),
};

function contractStopwords(): string[] {
  const doc = fs.readFileSync(CONTRACTS, 'utf8');
  const m = doc.match(/STOPWORDS \(exact, shared by all copies\):\s*`([^`]+)`/);
  if (!m) throw new Error('CONTRACTS.md: STOPWORDS literal not found');
  return m[1].trim().split(/\s+/);
}

describe('widget drift — against docs/CONTRACTS.md (always runs)', () => {
  it('STOPWORDS are byte-identical to the §5.2 list, in order', () => {
    expect(widget.stopwords).toEqual(contractStopwords());
    expect(new Set(widget.stopwords).size).toBe(widget.stopwords.length);
  });
  it('word-char regex is the §3.1 class with the u flag', () => {
    expect(widget.wordCharSource).toBe('[\\p{L}\\p{N}]');
    expect(widget.wordCharFlags).toBe('u');
  });
  it('keyword split regex is the §5.2 class, min token length 3, gate 0.6', () => {
    expect(widget.keywordSplitSource).toBe('[^a-z0-9]+');
    expect(widget.keywordSplitFlags).toBe('');
    expect(widget.keywordMinLen).toBe(3);
    expect(widget.confidenceGate).toBe(0.6);
  });
});

describe.skipIf(!CORE_PRESENT)('widget drift — against the core exports', () => {
  it('STOPWORDS byte-identical to core/infer/keyword STOPWORDS', async () => {
    const core = await import(CORE_KEYWORD!);
    expect(core.STOPWORDS, 'core/infer/keyword must export STOPWORDS').toBeDefined();
    const coreList = Array.from(core.STOPWORDS as Iterable<string>);
    expect(JSON.stringify(widget.stopwords)).toBe(JSON.stringify(coreList));
  });
  it('WORD_CHAR_RE.source byte-identical to core/tokenizer WORD_CHAR_RE', async () => {
    const core = await import(CORE_TOKENIZER!);
    expect(core.WORD_CHAR_RE, 'core/tokenizer must export WORD_CHAR_RE').toBeInstanceOf(RegExp);
    expect(widget.wordCharSource).toBe((core.WORD_CHAR_RE as RegExp).source);
    expect(widget.wordCharFlags).toBe((core.WORD_CHAR_RE as RegExp).flags);
  });
  it('keyword-token regex and the confidence default match the core (or the CONTRACTS literal when the core does not export them)', async () => {
    const kw = await import(CORE_KEYWORD!);
    const splitRe: RegExp | undefined = kw.KEYWORD_SPLIT_RE || kw.KEYWORD_TOKEN_RE || kw.TOKEN_SPLIT_RE || kw.SPLIT_RE;
    if (splitRe instanceof RegExp) {
      expect(widget.keywordSplitSource).toBe(splitRe.source);
      expect(widget.keywordSplitFlags).toBe(splitRe.flags);
    } else {
      expect(widget.keywordSplitSource).toBe('[^a-z0-9]+');
    }
    let gate: unknown = kw.DEFAULT_CONFIDENCE_THRESHOLD ?? kw.DEFAULT_CONFIDENCE_GATE ?? kw.CONFIDENCE_GATE ?? kw.DEFAULT_THRESHOLD;
    if (gate === undefined) {
      const ladder = resolveCore('core/infer/ladder');
      if (ladder) {
        const l = await import(ladder);
        gate = l.DEFAULT_CONFIDENCE_THRESHOLD ?? l.DEFAULT_CONFIDENCE_GATE ?? l.CONFIDENCE_GATE ?? l.DEFAULT_THRESHOLD;
      }
    }
    expect(widget.confidenceGate).toBe(typeof gate === 'number' ? gate : 0.6);
  });
});

describe.skipIf(!CORE_PRESENT)('widget drift — built-in meta intents and copy against the core', () => {
  // The widget carries its own BUILTIN_META (label + keywords) and META_COPY so a registry
  // answers identically locally and via serverUrl. Read AS TEXT from the widget source.
  const greetingBlock = extract(/greeting: \{\s*id: 'greeting', family: 'meta', label: '([^']*)', slots: \[\],\s*keywords: '([^']*)'/m, 'BUILTIN_META.greeting');
  const metaCopy = extract(/var META_COPY = \{\s*greeting: '([^']*)',\s*out_of_domain: "([^"]*)",\s*fallback: "([^"]*)"/m, 'META_COPY');
  it('greeting label + keyword bag are byte-identical to core/registry/builtins', async () => {
    const b = await import(resolveCore('core/registry/builtins')!);
    expect(greetingBlock[1]).toBe(b.BUILTIN_META_INTENTS.greeting.label);
    expect(greetingBlock[2]).toBe(b.BUILTIN_META_INTENTS.greeting.keywords);
  });
  it('greeting / out_of_domain / fallback copy are byte-identical to core/infer/planner', async () => {
    const pl = await import(resolveCore('core/infer/planner')!);
    expect(metaCopy[1]).toBe(pl.GREETING_COPY);
    expect(metaCopy[2]).toBe(pl.OUT_OF_DOMAIN_COPY);
    expect(metaCopy[3]).toBe(pl.FALLBACK_COPY);
  });
});

describe.skipIf(CORE_PRESENT)('widget drift — core comparison NOT RUN', () => {
  it('reports the missing core files instead of weakening the test', () => {
    // eslint-disable-next-line no-console
    console.warn('[widgetDrift] NOT RUN against core: missing ' +
      [!CORE_KEYWORD && 'core/infer/keyword', !CORE_TOKENIZER && 'core/tokenizer'].filter(Boolean).join(', ') +
      ' — only the CONTRACTS.md comparison ran.');
    expect(CORE_PRESENT).toBe(false);
  });
});
