const fs = require('fs');
const html = fs.readFileSync(require('path').join(__dirname, '../PRISM-v10-complete.html'), 'utf8');
const re = /<button([^>]*)>([\s\S]*?)<\/button>/gi;
let m;
while ((m = re.exec(html))) {
  const attrs = m[1];
  const content = m[2].trim();
  const hasAria = /aria-label\s*=/.test(attrs);
  if (!hasAria && (!content || /^[^A-Za-z0-9]+$/.test(content))) {
    console.log('BUTTON', attrs.trim(), 'CONTENT', JSON.stringify(content));
  }
}
