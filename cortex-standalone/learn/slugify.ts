/**
 * Identifier helpers for `learn` (CONTRACTS §8.2).
 *
 * `toIntentId` turns free link text into a registry intent id
 * (`^[a-z][a-z0-9_]*$`, ≤ 64 chars), ASCII-folding accents, deduping with
 * `_2`/`_3`… against `taken`, and prefixing `go_` when the text would start
 * with a digit. `toAppSlug` does the same for the app slug shape
 * (`^[a-z0-9][a-z0-9-]*$`).
 */

const INTENT_ID_MAX = 64;
const APP_SLUG_MAX = 64;

/** Lowercase + strip combining marks so "Café" → "cafe". */
function asciiFold(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase();
}

export function toIntentId(text: string, taken: Set<string>): string {
  let base = asciiFold(text)
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .replace(/_{2,}/g, '_');
  if (base === '') base = 'destination';
  if (/^[0-9]/.test(base)) base = `go_${base}`;
  if (base.length > INTENT_ID_MAX) base = base.slice(0, INTENT_ID_MAX).replace(/_+$/g, '');

  let candidate = base;
  let n = 2;
  while (taken.has(candidate)) {
    const suffix = `_${n}`;
    candidate = base.slice(0, INTENT_ID_MAX - suffix.length).replace(/_+$/g, '') + suffix;
    n += 1;
  }
  taken.add(candidate);
  return candidate;
}

export function toAppSlug(text: string): string {
  let slug = asciiFold(text)
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
  if (slug === '') slug = 'site';
  if (slug.length > APP_SLUG_MAX) slug = slug.slice(0, APP_SLUG_MAX).replace(/-+$/g, '');
  return slug;
}
