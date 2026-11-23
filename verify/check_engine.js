/* Cross-checks the JavaScript engine embedded in docs/index.html against Python.

   node verify/check_engine.js trace.json genomes.json
       trace.json    from  python verify/physics_trace.py
       genomes.json  from  python verify/export_genomes.py
   Options:  --engine path/to/index.html|engine.js   --runs N --generations G [--seed S] [--json out.json]
             (runs N seeded trainings in JS and prints the same summary as
              python headless_train.py --runs N)
*/
const fs = require('fs'), path = require('path'), vm = require('vm');
const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf('--' + name); return i < 0 ? def : args[i + 1]; };
const pos = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const root = path.resolve(__dirname, '..');
const enginePath = opt('engine', path.join(root, 'docs', 'index.html'));
let src = fs.readFileSync(enginePath, 'utf8');
let cfgText = fs.readFileSync(path.join(root, 'config.txt'), 'utf8');
if (enginePath.endsWith('.html')) {
  const m = src.match(/<script id="engine">([\s\S]*?)<\/script>/);
  const c = src.match(/<script type="text\/plain" id="config-txt">\n?([\s\S]*?)<\/script>/);
  if (!m) throw new Error('engine script not found');
  src = m[1];
  console.log('config.txt embedded in the page is identical to config.txt:', c && c[1].trim() === cfgText.trim());
}
const ctx = { module: { exports: {} }, console }; vm.createContext(ctx);
vm.runInContext(src + '\nthis.Engine = Engine;', ctx);
const E = ctx.Engine;
let bad = 0;

/* 1. physics */
if (pos[0]) {
  const trace = JSON.parse(fs.readFileSync(pos[0], 'utf8'));
  const H = {
    mid: Array(16).fill(300),
    mixed: [300, 120, 400, 250, 60, 440, 180, 350, 90, 420, 200, 310, 140, 380, 260, 70],
    hard: [50, 449, 50, 449, 50, 449, 50, 449]
  };
  const mkEvery = n => { let i = 0; return () => ++i % n === 0; };
  const P = {
    never: () => () => false, low: () => y => y > 400,
    every17: () => mkEvery(17), every9: () => mkEvery(9),
    gap: () => (y, top, bottom) => y + 48 > bottom - 20
  };
  let ok = 0;
  for (const t of trace) {
    const r = E.playPolicy(P[t.policy](), H[t.heights], 3000);
    const same = r.frames === t.frames && r.score === t.score;
    if (same) ok++; else { bad++; console.log('PHYSICS MISMATCH', t, r); }
  }
  console.log(`physics traces identical to Python: ${ok}/${trace.length}`);
}

/* 2. network outputs */
if (pos[1]) {
  const gs = JSON.parse(fs.readFileSync(pos[1], 'utf8'));
  let n = 0, maxErr = 0;
  const inK = [-1, -2, -3], outK = [0];
  for (const j of gs) {
    const g = E.Genome.fromJSON({ id: 0, fitness: 0, nodes: j.nodes, conns: j.conns });
    const net = E.Network.create(g, inK, outK);
    j.inputs.forEach((inp, k) => { const o = net.activate(inp)[0]; maxErr = Math.max(maxErr, Math.abs(o - j.outputs[k])); n++; });
  }
  console.log(`network outputs vs neat-python: ${n} activations over ${gs.length} genomes, max abs difference ${maxErr.toExponential(2)}`);
  if (maxErr > 1e-9) bad++;
}

/* 3. seeded training runs */
const runs = +opt('runs', 0);
if (runs) {
  const gens = +opt('generations', 20), seed0 = +opt('seed', 0);
  const cfg = E.parseConfig(cfgText); const reached = [];
  for (let s = 0; s < runs; s++) {
    const rng = E.makeRng(seed0 + s), pop = new E.Population(cfg, rng);
    let hit = null;
    for (let g = 0; g < gens; g++) {
      const run = new E.GenerationRun(pop, rng); run.runToEnd();
      const row = pop.record();
      if (row.best >= cfg.NEAT.fitness_threshold) { hit = row.gen; break; }
      pop.advance();
    }
    if (hit !== null) reached.push(hit);
  }
  reached.sort((a, b) => a - b);
  const med = reached.length ? (reached.length % 2 ? reached[(reached.length - 1) / 2] : (reached[reached.length / 2 - 1] + reached[reached.length / 2]) / 2) : null;
  const mean = reached.length ? reached.reduce((a, b) => a + b, 0) / reached.length : null;
  if (opt('json', null)) fs.writeFileSync(opt('json'), JSON.stringify({ runs, generations: gens, seed: seed0, reached }));
  console.log(`${runs} runs, seeds ${seed0}..${seed0 + runs - 1}, up to ${gens} generations`);
  console.log(`reached fitness ${cfg.NEAT.fitness_threshold}: ${reached.length}/${runs} runs`);
  if (reached.length) console.log(`generation reached (0-based): median ${med} mean ${mean.toFixed(2)} min ${reached[0]} max ${reached[reached.length - 1]}`);
}
process.exit(bad ? 1 : 0);
