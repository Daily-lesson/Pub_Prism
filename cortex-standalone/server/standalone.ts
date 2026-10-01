/**
 * A runnable local server (CONTRACTS §7.3): the example dashboard as static
 * files plus the advisor router at `/api/advisor`.
 *
 *   PORT=8787 npx tsx server/standalone.ts
 *
 * This is a local end-to-end harness, not a deployment: it serves the package
 * root statically (so `/examples/demo-dashboard/` works with relative asset
 * paths) and has no auth or rate limiting — see server/router.ts.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import express from 'express';
import { loadRegistryFile } from '../core/index';
import { createAdvisorRouter } from './router';

const PKG_ROOT = path.resolve(__dirname, '..');
const EXAMPLE_REGISTRY = path.join(PKG_ROOT, 'registry', 'examples', 'ops-dashboard.registry.json');
const DEMO_ARTIFACTS = path.join(PKG_ROOT, 'examples', 'demo-dashboard', 'cortex', 'model');
const RUNTIME_DIR = path.join(PKG_ROOT, 'runtime');

export function createStandaloneApp(): express.Express {
  const registry = loadRegistryFile(EXAMPLE_REGISTRY);
  const app = express();
  app.disable('x-powered-by');
  app.use('/api/advisor', createAdvisorRouter({
    registry,
    // Keyword-only until someone trains a model into the demo folder.
    artifactsDir: fs.existsSync(path.join(DEMO_ARTIFACTS, 'ledger.json')) ? DEMO_ARTIFACTS : undefined,
    runtimeDir: fs.existsSync(RUNTIME_DIR) ? RUNTIME_DIR : undefined,
    // Demo status provider: there is no live data behind this server, so it returns undefined
    // for every status intent and the registry's honest `unavailable` copy is shown (§5.5/§9).
    status: () => undefined,
  }));
  // Only the four trees the demo page references are served — never tests/, train/, build/
  // (the compiled dataset) or the lockfile — so a copy-paste deploy of this harness can't
  // expose them.
  for (const sub of ['examples', 'widget', 'registry', 'runtime']) {
    app.use(`/${sub}`, express.static(path.join(PKG_ROOT, sub), { dotfiles: 'ignore', index: ['index.html'] }));
  }
  app.get('/', (_req, res) => res.redirect('/examples/demo-dashboard/'));
  return app;
}

if (require.main === module) {
  const port = Number(process.env.PORT ?? 8787);
  const app = createStandaloneApp();
  app.listen(port, () => {
    console.log(`cortex standalone: http://localhost:${port}/examples/demo-dashboard/  (advisor at /api/advisor)`);
  });
}
