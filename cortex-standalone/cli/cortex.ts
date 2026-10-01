#!/usr/bin/env node
/**
 * cortex CLI — validate / compile / eval-gate / learn.
 *
 *   cortex validate  <registry.json>
 *   cortex compile   <registry.json> --out <dir> [--seed 42]
 *   cortex eval-gate <registry.json> --artifacts <dir> [--epsilon 0.03]
 *   cortex learn     <url | file.html | sitemap.xml> --out <registry.json> [--crawl] [--depth N] [--name X] [--slug Y]
 */

import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { compileDataset, writeDataset } from '../core/dataset/index';
import { loadRegistryFile, RegistryValidationError, registryHash } from '../core/registry/index';
import { runRegressionGate } from '../core/eval/index';

interface Parsed {
  positional: string[];
  flags: Record<string, string | true>;
}

function parseArgs(argv: string[]): Parsed {
  const positional: string[] = [];
  const flags: Record<string, string | true> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const eq = key.indexOf('=');
      if (eq !== -1) {
        flags[key.slice(0, eq)] = key.slice(eq + 1);
      } else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
        flags[key] = argv[++i];
      } else {
        flags[key] = true;
      }
    } else {
      positional.push(a);
    }
  }
  return { positional, flags };
}

function flagString(flags: Parsed['flags'], name: string): string | undefined {
  const v = flags[name];
  return typeof v === 'string' ? v : undefined;
}

function usage(): string {
  return [
    'usage:',
    '  cortex validate  <registry.json>',
    '  cortex compile   <registry.json> --out <dir> [--seed 42]',
    '  cortex eval-gate <registry.json> --artifacts <dir> [--epsilon 0.03]',
    '  cortex learn     <url | file.html | sitemap.xml> --out <registry.json> [--crawl] [--depth N] [--name X] [--slug Y]',
  ].join('\n');
}

function fail(message: string, code = 1): never {
  process.stderr.write(`${message}\n`);
  process.exit(code);
}

function requireRegistryArg(positional: string[], cmd: string): string {
  const p = positional[0];
  if (!p) fail(`cortex ${cmd}: missing <registry.json>\n${usage()}`);
  return path.resolve(p);
}

function loadOrFail(registryPath: string) {
  try {
    return loadRegistryFile(registryPath);
  } catch (e) {
    if (e instanceof RegistryValidationError) fail(e.message);
    fail(`could not load ${registryPath}: ${(e as Error).message}`);
  }
}

async function cmdValidate(p: Parsed): Promise<void> {
  const registryPath = requireRegistryArg(p.positional, 'validate');
  const registry = loadOrFail(registryPath);
  const slots = Object.keys(registry.slots).length;
  process.stdout.write(
    `OK ${registryPath}\n  app: ${registry.app.slug} (${registry.app.name})\n  intents: ${registry.intents.length} (incl. built-ins)\n` +
      `  slots: ${slots}\n  heldout: ${registry.heldout.length}\n  registryHash: ${registryHash(registry)}\n`,
  );
}

async function cmdCompile(p: Parsed): Promise<void> {
  const registryPath = requireRegistryArg(p.positional, 'compile');
  const out = flagString(p.flags, 'out');
  if (!out) fail(`cortex compile: --out <dir> is required\n${usage()}`);
  const seedRaw = flagString(p.flags, 'seed') ?? '42';
  const seed = Number(seedRaw);
  if (!Number.isFinite(seed) || !Number.isInteger(seed)) fail(`cortex compile: --seed must be an integer (got ${JSON.stringify(seedRaw)})`);
  const registry = loadOrFail(registryPath);
  let ds;
  try {
    ds = compileDataset(registry, seed);
  } catch (e) {
    fail(`cortex compile: ${(e as Error).message}`);
  }
  const outDir = path.resolve(out);
  writeDataset(ds, outDir);
  const c = ds.manifest.counts;
  process.stdout.write(
    `compiled ${registryPath} seed=${seed} -> ${outDir}\n  train=${c.train} val=${c.val} test=${c.test} total=${c.total}\n` +
      `  intents=${ds.manifest.intents.length} slots=${ds.manifest.slotNames.length}\n  registryHash=${ds.manifest.registryHash}\n  datasetHash=${ds.manifest.datasetHash}\n`,
  );
}

async function cmdEvalGate(p: Parsed): Promise<void> {
  const registryPath = requireRegistryArg(p.positional, 'eval-gate');
  const artifacts = flagString(p.flags, 'artifacts');
  if (!artifacts) fail(`cortex eval-gate: --artifacts <dir> is required\n${usage()}`);
  const epsilonRaw = flagString(p.flags, 'epsilon');
  const epsilon = epsilonRaw === undefined ? undefined : Number(epsilonRaw);
  if (epsilon !== undefined && !Number.isFinite(epsilon)) fail(`cortex eval-gate: --epsilon must be a number`);
  const registry = loadOrFail(registryPath);
  const report = await runRegressionGate({ registry, artifactsDir: path.resolve(artifacts), epsilon });
  const lines: string[] = [];
  lines.push(`eval-gate ${report.passed ? 'PASSED' : 'FAILED'} (ledger ${report.ledgerVersion ?? 'n/a'})`);
  lines.push(`  artifact sha256: ${report.artifact.sha256Verified ? 'verified' : 'NOT verified'}`);
  lines.push(`  registryHash: ${report.registry.registryHash}${report.registry.matches ? ' (matches ledger)' : ' (differs from ledger — answer copy may have changed; not a gate failure)'}`);
  lines.push(`  dataset: ${report.dataset.deterministic ? 'deterministic' : 'DRIFTED'} (${report.dataset.datasetHash ?? 'n/a'})`);
  if (report.inDistribution) {
    lines.push(`  in-distribution: n=${report.inDistribution.n} accuracy=${report.inDistribution.accuracy.toFixed(4)} (ledger ${report.inDistribution.ledgerAccuracy ?? 'n/a'}) regressions=${report.inDistribution.regressions.length}`);
  }
  if (report.heldout) {
    lines.push(
      report.heldout.enforced
        ? `  heldout: n=${report.heldout.n} accuracy=${report.heldout.accuracy?.toFixed(4)} floor=${report.heldout.floor}`
        : `  heldout: not enforced (ledger metrics.heldout is null; ${report.heldout.n} declared)`,
    );
  }
  for (const r of report.reasons) lines.push(`  ✗ ${r}`);
  process.stdout.write(lines.join('\n') + '\n');
  if (!report.passed) process.exit(1);
}

async function cmdLearn(p: Parsed): Promise<void> {
  const input = p.positional[0];
  if (!input) fail(`cortex learn: missing <url | file.html | sitemap.xml>\n${usage()}`);
  const out = flagString(p.flags, 'out');
  if (!out) fail(`cortex learn: --out <registry.json> is required\n${usage()}`);
  const depthRaw = flagString(p.flags, 'depth');
  const depth = depthRaw === undefined ? undefined : Number(depthRaw);
  if (depth !== undefined && (!Number.isInteger(depth) || depth < 0)) fail('cortex learn: --depth must be a non-negative integer');
  type LearnFn = (
    input: string,
    opts: { crawl?: boolean; depth?: number; name?: string; slug?: string },
  ) => Promise<{ registry: unknown; report: { pages: number; intents: number; warnings: string[] } }>;
  let learn: LearnFn;
  try {
    // Resolved only when this command runs, so the rest of the CLI works without the learn module.
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    learn = (require('../learn') as { learn: LearnFn }).learn;
  } catch (e) {
    fail(`cortex learn: the learn module is not available: ${(e as Error).message}`);
  }
  const result = await learn(input, {
    crawl: p.flags.crawl === true || p.flags.crawl === 'true',
    depth,
    name: flagString(p.flags, 'name'),
    slug: flagString(p.flags, 'slug'),
  });
  const outPath = path.resolve(out);
  writeFileSync(outPath, JSON.stringify(result.registry, null, 2) + '\n', 'utf8');
  process.stdout.write(`wrote DRAFT registry -> ${outPath}\n  pages: ${result.report.pages}\n  intents: ${result.report.intents}\n`);
  for (const w of result.report.warnings) process.stdout.write(`  ! ${w}\n`);
  process.stdout.write('  edit labels/keywords/steps before training.\n');
}

async function main(): Promise<void> {
  const [cmd, ...rest] = process.argv.slice(2);
  const parsed = parseArgs(rest);
  switch (cmd) {
    case 'validate':
      return cmdValidate(parsed);
    case 'compile':
      return cmdCompile(parsed);
    case 'eval-gate':
      return cmdEvalGate(parsed);
    case 'learn':
      return cmdLearn(parsed);
    case undefined:
    case '--help':
    case '-h':
    case 'help':
      process.stdout.write(usage() + '\n');
      return;
    default:
      fail(`unknown command "${cmd}"\n${usage()}`);
  }
}

main().catch((e: Error) => fail(`cortex: ${e.message}`));
