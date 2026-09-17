/**
 * A small hand-written HTML extractor for `learn` — no dependency.
 *
 * Good enough for: `<title>`, `<h1>`/`<h2>` text, and `<a href>` links inside
 * the `<nav>`, `<header>` and `<aside>` landmarks (plus the full link list, for
 * the no-landmark fallback). It is NOT a general HTML parser: nested landmarks
 * of the same tag and exotic markup are handled loosely, which is fine for a
 * draft-time crawler whose output a human edits anyway.
 */

export interface ExtractedLink {
  /** Raw `href` attribute value, entity-decoded, untrimmed of scheme. */
  href: string;
  /** Visible text, tags stripped, whitespace collapsed. */
  text: string;
}

export interface ExtractedPage {
  title: string;
  h1: string[];
  h2: string[];
  /** Links found inside <nav>/<header>/<aside>. */
  landmarkLinks: ExtractedLink[];
  /** Every link on the page. */
  allLinks: ExtractedLink[];
  /** True when at least one landmark element exists on the page. */
  hasLandmarks: boolean;
}

const NAMED_ENTITIES: Record<string, string> = {
  amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ', ndash: '–', mdash: '—',
  hellip: '…', copy: '©', reg: '®', trade: '™', rsquo: '’', lsquo: '‘',
  rdquo: '”', ldquo: '“', bull: '•', middot: '·', laquo: '«', raquo: '»',
  // Latin-1 accented letters (the ones that show up in nav labels); names are case-sensitive in HTML
  agrave: 'à', aacute: 'á', acirc: 'â', atilde: 'ã', auml: 'ä', aring: 'å', aelig: 'æ',
  ccedil: 'ç', egrave: 'è', eacute: 'é', ecirc: 'ê', euml: 'ë', igrave: 'ì', iacute: 'í',
  icirc: 'î', iuml: 'ï', ntilde: 'ñ', ograve: 'ò', oacute: 'ó', ocirc: 'ô', otilde: 'õ',
  ouml: 'ö', oslash: 'ø', ugrave: 'ù', uacute: 'ú', ucirc: 'û', uuml: 'ü', yacute: 'ý',
  yuml: 'ÿ', szlig: 'ß',
  Agrave: 'À', Aacute: 'Á', Acirc: 'Â', Atilde: 'Ã', Auml: 'Ä', Aring: 'Å', AElig: 'Æ',
  Ccedil: 'Ç', Egrave: 'È', Eacute: 'É', Ecirc: 'Ê', Euml: 'Ë', Igrave: 'Ì', Iacute: 'Í',
  Icirc: 'Î', Iuml: 'Ï', Ntilde: 'Ñ', Ograve: 'Ò', Oacute: 'Ó', Ocirc: 'Ô', Otilde: 'Õ',
  Ouml: 'Ö', Oslash: 'Ø', Ugrave: 'Ù', Uacute: 'Ú', Ucirc: 'Û', Uuml: 'Ü', Yacute: 'Ý',
};

export function decodeEntities(s: string): string {
  return s.replace(/&(#x[0-9a-f]+|#[0-9]+|[a-zA-Z]+);/g, (m, body: string) => {
    const b = body.toLowerCase();
    if (b.startsWith('#x')) { const cp = parseInt(b.slice(2), 16); return Number.isFinite(cp) ? safeFromCodePoint(cp, m) : m; }
    if (b.startsWith('#')) { const cp = parseInt(b.slice(1), 10); return Number.isFinite(cp) ? safeFromCodePoint(cp, m) : m; }
    if (Object.prototype.hasOwnProperty.call(NAMED_ENTITIES, body)) return NAMED_ENTITIES[body];
    return Object.prototype.hasOwnProperty.call(NAMED_ENTITIES, b) ? NAMED_ENTITIES[b] : m;
  });
}

function safeFromCodePoint(cp: number, fallback: string): string {
  try { return String.fromCodePoint(cp); } catch { return fallback; }
}

/** Strip tags, decode entities, collapse whitespace. */
export function textOf(html: string): string {
  return decodeEntities(html.replace(/<[^>]*>/g, ' ')).replace(/\s+/g, ' ').trim();
}

/** Remove comments, scripts, styles and templates so their contents never look like links or headings. */
function stripNoise(html: string): string {
  return html
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script\s*>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style\s*>/gi, ' ')
    .replace(/<template\b[^>]*>[\s\S]*?<\/template\s*>/gi, ' ');
}

function attr(tagAttrs: string, name: string): string | undefined {
  const re = new RegExp(`(?:^|\\s)${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'>]+))`, 'i');
  const m = re.exec(tagAttrs);
  if (!m) return undefined;
  return decodeEntities(m[1] ?? m[2] ?? m[3] ?? '');
}

export function extractLinks(html: string): ExtractedLink[] {
  const out: ExtractedLink[] = [];
  const re = /<a\b([^>]*)>([\s\S]*?)<\/a\s*>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) {
    const href = attr(m[1], 'href');
    if (href === undefined) continue;
    let text = textOf(m[2]);
    if (text === '') {
      // Image-only or icon links: fall back to aria-label / title / img alt.
      text = attr(m[1], 'aria-label') ?? attr(m[1], 'title') ?? '';
      if (text === '') {
        const img = /<img\b([^>]*)>/i.exec(m[2]);
        if (img) text = attr(img[1], 'alt') ?? '';
      }
    }
    out.push({ href: href.trim(), text: text.trim() });
  }
  return out;
}

function sections(html: string, tag: string): string[] {
  const out: string[] = [];
  const re = new RegExp(`<${tag}\\b[^>]*>([\\s\\S]*?)<\\/${tag}\\s*>`, 'gi');
  let m: RegExpExecArray | null;
  while ((m = re.exec(html)) !== null) out.push(m[1]);
  return out;
}

function headings(html: string, tag: string): string[] {
  return sections(html, tag).map(textOf).filter((t) => t !== '');
}

export function extractPage(rawHtml: string): ExtractedPage {
  const html = stripNoise(rawHtml);
  const titleMatch = /<title\b[^>]*>([\s\S]*?)<\/title\s*>/i.exec(html);
  const title = titleMatch ? textOf(titleMatch[1]) : '';
  const landmarks = [...sections(html, 'nav'), ...sections(html, 'header'), ...sections(html, 'aside')];
  const landmarkLinks = landmarks.flatMap(extractLinks);
  return {
    title,
    h1: headings(html, 'h1'),
    h2: headings(html, 'h2'),
    landmarkLinks,
    allLinks: extractLinks(html),
    hasLandmarks: landmarks.length > 0,
  };
}

/** `<loc>` entries of a sitemap (or sitemap index). */
export function extractSitemapLocs(xml: string): { locs: string[]; isIndex: boolean } {
  const locs: string[] = [];
  const re = /<loc\b[^>]*>([\s\S]*?)<\/loc\s*>/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml)) !== null) {
    const v = decodeEntities(m[1]).trim();
    if (v !== '') locs.push(v);
  }
  return { locs, isIndex: /<sitemapindex\b/i.test(xml) };
}
