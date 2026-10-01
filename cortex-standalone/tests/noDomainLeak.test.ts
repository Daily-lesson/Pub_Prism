/**
 * Guard: this package is host-agnostic and is published outside the product it was
 * extracted from. No file in it may carry that product's domain vocabulary — not in
 * code, comments, tests, fixtures, or the example registry. The only files allowed to
 * name the origin product (by its name, never its domain) are the three attribution/
 * design documents listed in ALLOW_ORIGIN_NAME.
 *
 * Non-vacuity: `tests/noDomainLeak.test.ts` itself contains the words (in the list below),
 * so it is excluded by path — if you widen the exclusions, re-run the ritual of dropping a
 * forbidden word into any source file and watching this test name the file and line.
 */
import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(__dirname, '..');

const EXCLUDED_DIRS = new Set(['node_modules', 'build', '.git', 'runtime', '.vitest', '__pycache__', '.venv']);
const EXCLUDED_FILES = new Set([path.join('tests', 'noDomainLeak.test.ts')]);
/** Files that may name the origin product (attribution/design record), still never its domain words. */
const ALLOW_ORIGIN_NAME = new Set(['README.md', 'NOTICE.md', path.join('docs', 'ARCHITECTURE.md')]);

const DOMAIN_WORDS = ['robot', 'robots', 'mission', 'missions', 'fleet', 'geofence', 'geofences', 'telemetry'];
const ORIGIN_NAME = ['prism'];

const TEXT_EXT = new Set(['.ts', '.js', '.mjs', '.py', '.json', '.jsonl', '.md', '.html', '.css', '.yml', '.yaml', '.txt', '.xml']);

function walk(dir: string, out: string[]): void {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!EXCLUDED_DIRS.has(entry.name)) walk(path.join(dir, entry.name), out);
      continue;
    }
    if (TEXT_EXT.has(path.extname(entry.name))) out.push(path.join(dir, entry.name));
  }
}

function findHits(file: string, words: string[]): string[] {
  const hits: string[] = [];
  const lines = fs.readFileSync(file, 'utf8').split('\n');
  const re = new RegExp(`\\b(${words.join('|')})\\b`, 'i');
  lines.forEach((line, i) => {
    const m = re.exec(line);
    if (m) hits.push(`${path.relative(ROOT, file)}:${i + 1}: "${m[1]}"`);
  });
  return hits;
}

describe('no domain vocabulary from the origin product leaks into the package', () => {
  const files: string[] = [];
  walk(ROOT, files);
  const scanned = files.filter((f) => !EXCLUDED_FILES.has(path.relative(ROOT, f)));

  it('scans a non-trivial set of files (non-vacuity)', () => {
    expect(scanned.length).toBeGreaterThan(20);
    expect(scanned.some((f) => f.endsWith('.py'))).toBe(true);
    expect(scanned.some((f) => f.endsWith('.js'))).toBe(true);
  });

  it('contains none of the origin product domain words anywhere', () => {
    const hits = scanned.flatMap((f) => findHits(f, DOMAIN_WORDS));
    expect(hits, hits.join('\n')).toEqual([]);
  });

  it('names the origin product only in the attribution/design documents', () => {
    const hits = scanned
      .filter((f) => !ALLOW_ORIGIN_NAME.has(path.relative(ROOT, f)))
      .flatMap((f) => findHits(f, ORIGIN_NAME));
    expect(hits, hits.join('\n')).toEqual([]);
  });
});
