/**
 * Deterministic seeded PRNG (mulberry32) + seed derivation (docs/CONTRACTS.md §2.10).
 * All randomness in the compiler flows through this — never `Math.random()`.
 */

export type Rng = () => number;

/** mulberry32 seeded with a 32-bit unsigned integer; yields floats in [0,1). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return function next(): number {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** (baseSeed, salt) -> 32-bit sub-seed. FNV-1a-style mixing over the salt's UTF-16 code units,
 * starting from `baseSeed ^ FNV_OFFSET`. */
export function deriveSeed(baseSeed: number, salt: string): number {
  let h = (baseSeed >>> 0) ^ 0x811c9dc5;
  for (let i = 0; i < salt.length; i++) {
    h ^= salt.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Random integer in [0, maxExclusive). */
export function randInt(rng: Rng, maxExclusive: number): number {
  return Math.floor(rng() * maxExclusive);
}

/** Deterministic Fisher-Yates shuffle; returns a new array. */
export function shuffle<T>(items: readonly T[], rng: Rng): T[] {
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = randInt(rng, i + 1);
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}

/** Deterministically pick `count` distinct items (or all, if fewer). */
export function sampleDistinct<T>(items: readonly T[], count: number, rng: Rng): T[] {
  return shuffle(items, rng).slice(0, Math.min(count, items.length));
}
