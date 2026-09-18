const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');
const axeCore = require('axe-core');

(async () => {
  const htmlPath = process.argv[2] || path.join(__dirname, '..', 'PRISM-v10-complete.html');
  const absHtml = path.resolve(htmlPath);
  if (!fs.existsSync(absHtml)) {
    console.error('HTML file not found:', absHtml);
    process.exit(2);
  }

  const browser = await puppeteer.launch({args:['--no-sandbox','--disable-setuid-sandbox']});
  try {
    const page = await browser.newPage();
    const url = 'file://' + absHtml;
    console.log('Loading', url);
    await page.goto(url, {waitUntil: 'networkidle2', timeout: 60000});
    // inject axe
    await page.evaluate(axeCore.source);
    // run axe
    const results = await page.evaluate(async () => {
      return await axe.run(document, { runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] } });
    });
    const outPath = path.join(__dirname, 'axe-report.json');
    fs.writeFileSync(outPath, JSON.stringify(results, null, 2));
    console.log('Axe results saved to', outPath);
    console.log(`${results.violations.length} violations, ${results.incomplete.length} incomplete, ${results.passes.length} passes, ${results.inapplicable.length} inapplicable`);
    if(results.violations.length>0){
      console.log('Top violations:');
      results.violations.slice(0,5).forEach(v=>{
        console.log('-', v.id, `(${v.impact})`, v.help, `
  nodes: ${v.nodes.length}`);
      });
    }
  } catch (err) {
    console.error('Audit failed:', err);
    process.exitCode = 3;
  } finally {
    await browser.close();
  }
})();
