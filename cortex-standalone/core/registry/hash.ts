/**
 * Canonical JSON + registry hash (docs/CONTRACTS.md §1.7).
 */

import { createHash } from 'node:crypto';
import { normalizeRegistry } from './validate';
import type { NormalizedRegistry, Registry } from './types';

function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value !== null && typeof value === 'object') {
    const src = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(src).sort()) out[key] = canonicalize(src[key]);
    return out;
  }
  return value;
}

/** Recursive key-sorted `JSON.stringify`; arrays keep their order. */
export function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

export function sha256Hex(input: string | Buffer): string {
  return createHash('sha256').update(input).digest('hex');
}

/** sha256 of `canonicalJson(registry)` AFTER normalization (built-ins appended, defaults filled). */
export function registryHash(registry: Registry | NormalizedRegistry): string {
  return sha256Hex(canonicalJson(normalizeRegistry(registry)));
}
