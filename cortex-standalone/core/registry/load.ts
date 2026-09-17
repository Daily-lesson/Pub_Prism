import { readFileSync } from 'node:fs';
import { validateRegistry } from './validate';
import type { NormalizedRegistry } from './types';

export class RegistryValidationError extends Error {
  constructor(
    public readonly errors: string[],
    public readonly path?: string,
  ) {
    super(`${path ? `${path}: ` : ''}registry is invalid (${errors.length} error${errors.length === 1 ? '' : 's'}):\n  - ${errors.join('\n  - ')}`);
    this.name = 'RegistryValidationError';
  }
}

/** Read + parse + validate + normalize a registry file. Throws with every error line on failure. */
export function loadRegistryFile(path: string): NormalizedRegistry {
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, 'utf8'));
  } catch (e) {
    throw new RegistryValidationError([`could not read/parse JSON: ${(e as Error).message}`], path);
  }
  const result = validateRegistry(raw);
  if (!result.ok || !result.registry) throw new RegistryValidationError(result.errors, path);
  return result.registry;
}

/** Validate + normalize an in-memory value. Throws with every error line on failure. */
export function loadRegistry(raw: unknown): NormalizedRegistry {
  const result = validateRegistry(raw);
  if (!result.ok || !result.registry) throw new RegistryValidationError(result.errors);
  return result.registry;
}
