/**
 * Best-effort query redaction: emails, id-shaped tokens, digit runs, length cap. Deliberately
 * conservative (over-redact rather than under-redact). Pure.
 */

const EMAIL_RE = /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/gi;
// Long alphanumeric runs (>=10 chars) mixing letters and digits — serials, order ids, UUID
// fragments, key-shaped strings. Ordinary words never match (they're all-letters).
const LONG_ID_TOKEN_RE = /\b(?=[a-z0-9]*[0-9])(?=[a-z0-9]*[a-z])[a-z0-9-]{10,}\b/gi;
const DIGIT_RE = /\d+/g;

/** Emails + id-shaped tokens only; digit runs are preserved. */
export function redactIdentifiers(input: string, maxLen = 300): string {
  let out = input.replace(EMAIL_RE, '[email]').replace(LONG_ID_TOKEN_RE, '[id]');
  out = out.trim().replace(/\s+/g, ' ');
  if (out.length > maxLen) out = `${out.slice(0, maxLen)}…`;
  return out;
}

/** `redactIdentifiers` + a generic digit strip — the strongest scrub. Emails and ids are
 * replaced as whole units before digits are stripped. */
export function redactText(input: string, maxLen = 300): string {
  const out = redactIdentifiers(input, Number.MAX_SAFE_INTEGER).replace(DIGIT_RE, '#');
  return out.length > maxLen ? `${out.slice(0, maxLen)}…` : out;
}
