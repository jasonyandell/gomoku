export const meta = {
  name: 'wiki-refine-loop',
  description: 'Grind the gomoku wiki to ship-shape: loop [adversarial review -> apply actionable findings] until an adversarial pass comes back clean (or a cap).',
  whenToUse: 'After a wiki restructure, or any time the hub-of-hubs wiki has drifted and you want it driven to ship-shape hands-off. Edits only the 12 synthesis pages; leaves evidence/leaf pages alone.',
  phases: [
    { title: 'Review', detail: 'adversarial newcomer review + link-check -> structured verdict' },
    { title: 'Refine', detail: 'one agent applies the actionable findings to synthesis pages only' },
  ],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const MAX = A.maxIterations || 4
const WIKI = '/Users/jason/code/gomoku/wiki'

// The ONLY files the fixer may edit (synthesis layer). Everything else is evidence.
const SYNTH = 'index.md, ops.md, alphazero.md, experiments.md, seek-vct.md, derby.md, ' +
  'm5-mainframe.md, reference.md, training-timeline.md, train-a-model.md, eval-a-model.md, ' +
  `publish-a-model.md (all directly under ${WIKI}/).`

const LINKCHECK = [
  `python3 - <<'PY'`,
  `import re,os,glob`,
  `os.chdir(${JSON.stringify(WIKI)})`,
  `mds=[f for f in glob.glob('*.md')+glob.glob('topics/*.md')+glob.glob('ops/*.md')+glob.glob('sources/*.md')+glob.glob('cards/*.md') if not f.startswith('_archive')]`,
  `bad=[]`,
  `for f in mds:`,
  `    d=os.path.dirname(f) or '.'`,
  `    for ln,line in enumerate(open(f),1):`,
  `        for m in re.finditer(r'\\]\\(([^)]+)\\)',line):`,
  `            t=m.group(1).split('#')[0].strip()`,
  `            if not t or t.startswith('http') or t.startswith('mailto'): continue`,
  `            if not os.path.exists(os.path.normpath(os.path.join(d,t))): bad.append(f+':'+str(ln)+' -> '+t)`,
  `print('BROKEN' if bad else 'CLEAN', len(bad))`,
  `[print(' ',b) for b in bad[:40]]`,
  `PY`,
].join('\n')

const REVIEW_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ship_shape', 'link_check_clean', 'one_fetch_ok', 'findings', 'verdict'],
  properties: {
    ship_shape: { type: 'boolean', description: 'true ONLY if there are zero must/should findings AND links clean AND index+hubs each fit one fetch' },
    link_check_clean: { type: 'boolean' },
    one_fetch_ok: { type: 'boolean', description: 'index.md and each hub page are each under ~24k tokens (~18k words)' },
    findings: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['severity', 'file', 'summary', 'fix'],
      properties: {
        severity: { type: 'string', enum: ['must', 'should', 'nice'] },
        file: { type: 'string' },
        line: { type: 'string' },
        summary: { type: 'string', description: 'the concrete defect (mis-route, stale claim, contradiction, unreachable page, essay-bloat, broken link)' },
        fix: { type: 'string', description: 'the specific edit that resolves it' },
      } } },
    verdict: { type: 'string', description: 'one line: ship-shape, or the top thing still wrong' },
  },
}

let iteration = 0
const history = []

while (iteration < MAX) {
  iteration++

  phase('Review')
  const review = await agent(
    `Adversarially review the gomoku hub-of-hubs wiki at ${WIKI} as a skeptical NEWCOMER — hunt for what's WRONG, do not cheerlead.\n\n` +
    `Read: index.md, ops.md, the 5 hubs (alphazero/experiments/seek-vct/derby/m5-mainframe/reference), training-timeline.md, and the 3 workflow pages (train/eval/publish-a-model). Then judge:\n` +
    `- NAVIGATION: can a newcomer answer "train?/eval?/current state?/where's the VCT work?/what next?" in <=2 hops? Any page reachable from no hub?\n` +
    `- ACCURACY: spot-check 3-4 bold claims against the actual topic/leaf page they link to; flag any claim not supported by its own link, or any internal contradiction between two hubs.\n` +
    `- ALTITUDE: are hubs still hubs (start->now->learned + links), or has essay-bloat crept back? Anything stale/misleading?\n` +
    `- ONE-FETCH: index.md and each hub must each fit one fetch (~<=18k words). Report one_fetch_ok.\n` +
    `- LINKS: run this exact command and report link_check_clean based on whether it prints CLEAN:\n\`\`\`\n${LINKCHECK}\n\`\`\`\n\n` +
    `Return findings ranked, each tagged severity must/should/nice with a concrete file + fix. Set ship_shape=true ONLY if there are zero must AND zero should findings, links are CLEAN, and one_fetch_ok. Iteration ${iteration}/${MAX}. Do NOT edit any files.`,
    { label: `review#${iteration}`, phase: 'Review', agentType: 'general-purpose', effort: 'high', schema: REVIEW_SCHEMA }
  )

  const actionable = (review.findings || []).filter(f => f.severity === 'must' || f.severity === 'should')
  history.push({ iteration, ship_shape: review.ship_shape, link_clean: review.link_check_clean, actionable: actionable.length, verdict: review.verdict })
  log(`review#${iteration}: ship_shape=${review.ship_shape} links=${review.link_check_clean ? 'clean' : 'BROKEN'} actionable=${actionable.length} — ${review.verdict}`)

  if (review.ship_shape && review.link_check_clean && review.one_fetch_ok) { log(`SHIP-SHAPE at iteration ${iteration}`); break }
  if (!actionable.length && review.link_check_clean) { log(`no actionable findings left (only nice-to-haves) — stopping`); break }

  phase('Refine')
  const fixReport = await agent(
    `Apply these adversarial review findings to the gomoku wiki. You may ONLY edit these synthesis pages: ${SYNTH}\n` +
    `Do NOT edit any topic/ops/source/card leaf page, TRAINING_WIKI.md, or anything in _archive/ — those are evidence.\n\n` +
    `FINDINGS (JSON):\n${JSON.stringify(actionable)}\n\n` +
    `Rules: make the MINIMAL accurate edit per finding; every factual claim must be supported by the page it links to (verify by reading that page); keep hubs at hub-altitude (pointers, not essays); do not introduce broken links. After editing, re-run this link check and confirm it prints CLEAN:\n\`\`\`\n${LINKCHECK}\n\`\`\`\n` +
    `Report concisely what you changed per finding and the final link-check result. Iteration ${iteration}.`,
    { label: `fix#${iteration}`, phase: 'Refine', agentType: 'general-purpose', effort: 'high' }
  )
  log(`fix#${iteration} applied`)
  history[history.length - 1].fix = String(fixReport).slice(0, 400)
}

return {
  iterations: iteration,
  converged: history[history.length - 1]?.ship_shape === true,
  history,
}
