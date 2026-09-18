// PR triage in the browser: the same typed judgments as examples/pr_review.py, run on the
// in-page engine. Fetches public pull requests straight from the GitHub API (CORS-enabled,
// 60 requests/hour unauthenticated).

export const PR_QUESTIONS = {
  kind: { type: "choice", instructions: "What kind of change is this pull request, judging from the title, description and the files touched?",
    criteria: { feature: "adds new user-facing behaviour or capability", bugfix: "corrects incorrect behaviour", refactor: "restructures code without changing behaviour", docs: "documentation, comments, changelog only", tests: "adds or changes tests only", chore: "build, CI, dependencies, formatting, release plumbing" } },
  description_matches: { type: "noul", instructions: "Does the `body` accurately describe what the changed `files` and `stats` suggest the PR does?",
    criteria: { true: "the description explains the change and nothing important is left unsaid", false: "the description is missing, generic, or does not match the files touched" } },
  breaking_change: { type: "noul", instructions: "Is this likely a breaking change for users of this project (public API, config format, CLI flags, behaviour that callers rely on)?" },
  needs_migration: { type: "noul", instructions: "Does this change need a data or schema migration, or a coordinated rollout (e.g. database, stored formats, protocol versions)?" },
  risk: { type: "score", instructions: "How risky is merging this PR, considering scope, the areas touched and how easy it is to reason about?",
    criteria: ["trivial: docs, comments, formatting, isolated tests", "low: small, local, easy to reason about", "medium: touches shared logic or several files; a careful review is warranted", "high: core behaviour, data handling, security, or concurrency; could break users"] },
};

export const HUNK_QUESTIONS = {
  sensitive_area: { type: "noul", instructions: "Does this hunk touch a sensitive area: authentication, authorization, permissions, secrets or credentials, payments, cryptography, or personal data?" },
  removes_error_handling: { type: "noul", instructions: "Does this hunk remove or weaken error handling, validation, or safety checks (e.g. deleting a try/except, an assert, a bounds check, or broadening a caught exception)?" },
  leftover_debug: { type: "noul", instructions: "Does this hunk add leftover debugging or temporary code: print/console.log statements, commented-out code, TODO/FIXME/HACK markers, hard-coded test values?" },
  public_api_change: { type: "noul", instructions: "Does this hunk change a public interface: a function or class signature, an exported name, a CLI flag, a config key, a wire or file format?" },
  behaviour_change: { type: "noul", instructions: "Does this hunk change runtime behaviour (as opposed to a pure rename, formatting, comments, or type annotations)?",
    criteria: { true: "control flow, values, or side effects differ after the change", false: "purely cosmetic, comments, docs, renames, or typing" } },
  needs_eyes: { type: "score", instructions: "How much would a senior reviewer want to look closely at this hunk?",
    criteria: ["no: trivial or mechanical; skim at most", "a glance: straightforward change, low chance of a bug", "a careful read: non-trivial logic, edge cases possible", "definitely: subtle, risky, or hard to reason about; discuss before merging"] },
};

const SKIP = [/\.lock$/, /package-lock\.json$/, /yarn\.lock$/, /pnpm-lock\.yaml$/, /go\.sum$/, /\.min\.(js|css)$/, /\.(svg|png|jpg|gif|pdf|snap|ipynb)$/, /(^|\/)(dist|build|vendor|node_modules)\//];
export const TEST_RE = /(^|\/)(tests?|__tests__|spec|specs)(\/|$)|_test\.|\.test\.|\.spec\.|test_[^/]*\.py$/;

export function parseDiff(diff) {
  const hunks = [];
  let path = null, cur = null;
  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git")) {
      const m = line.match(/^diff --git a\/(.*?) b\/(.*)$/);
      path = m ? m[2] : line.split(" ").pop();
      cur = null;
    } else if (line.startsWith("@@")) {
      cur = { path: path ?? "?", header: line, lines: [line], added: 0, removed: 0 };
      hunks.push(cur);
    } else if (cur && !/^(\+\+\+|---|index |similarity|rename |new file|deleted file|Binary files)/.test(line)) {
      cur.lines.push(line);
      if (line.startsWith("+")) cur.added++; else if (line.startsWith("-")) cur.removed++;
    }
  }
  return hunks.map((h) => ({ ...h, text: h.lines.join("\n") }));
}

const skipPath = (p) => SKIP.some((re) => re.test(p));
const trim = (t, n) => (t.length <= n ? t : `${t.slice(0, (n * 2) / 3)}\n... [${t.length - n} chars omitted] ...\n${t.slice(-n / 3)}`);

export async function fetchPR(repo, number) {
  const base = `https://api.github.com/repos/${repo}/pulls/${number}`;
  const [meta, diff, files] = await Promise.all([
    fetch(base, { headers: { Accept: "application/vnd.github+json" } }).then((r) => { if (!r.ok) throw new Error(`GitHub: ${r.status} ${r.statusText}`); return r.json(); }),
    fetch(base, { headers: { Accept: "application/vnd.github.diff" } }).then((r) => r.text()),
    fetch(`${base}/files?per_page=100`, { headers: { Accept: "application/vnd.github+json" } }).then((r) => (r.ok ? r.json() : [])),
  ]);
  return { meta, diff, files };
}

const pAtLeast = (probs, level) => Object.entries(probs).reduce((a, [k, v]) => a + (Number(k) >= level ? v : 0), 0);

export async function reviewPR(engine, pr, { maxHunks = 12, maxChars = 3000, temperature = 1.0, onProgress } = {}) {
  const { meta, diff, files } = pr;
  const paths = files.map((f) => f.filename);
  const testFiles = paths.filter((p) => TEST_RE.test(p));
  const prState = {
    title: meta.title, body: trim(meta.body || "(no description)", 3000), author: meta.user?.login,
    labels: (meta.labels || []).map((l) => l.name), base_branch: meta.base?.ref,
    stats: { files: meta.changed_files, additions: meta.additions, deletions: meta.deletions },
    files: files.slice(0, 60).map((f) => ({ path: f.filename, "+": f.additions, "-": f.deletions })),
  };
  onProgress?.("PR-level questions…", 0);
  const prAnswers = (await engine.answer({ state: prState, questions: PR_QUESTIONS }, { temperature })).answers;

  const all = parseDiff(diff);
  let hunks = all.filter((h) => !skipPath(h.path));
  const skipped = all.length - hunks.length;
  hunks.sort((a, b) => (b.added + b.removed) - (a.added + a.removed));
  hunks = hunks.slice(0, maxHunks);
  const rows = [];
  for (let i = 0; i < hunks.length; i++) {
    const h = hunks[i];
    onProgress?.(`hunk ${i + 1} of ${hunks.length}: ${h.path}`, (i + 1) / (hunks.length + 1));
    const a = (await engine.answer({ state: { file: h.path, is_test_file: TEST_RE.test(h.path), hunk: trim(h.text, maxChars) }, questions: HUNK_QUESTIONS }, { temperature })).answers;
    const row = { file: h.path, header: h.header, added: h.added, removed: h.removed, is_test: TEST_RE.test(h.path) };
    for (const [k, v] of Object.entries(a)) row[k] = v.type === "noul" ? v.noul : v.score;
    row.p_needs_eyes = pAtLeast(a.needs_eyes.probabilities, 2);
    rows.push(row);
  }
  const byEyes = [...rows].sort((x, y) => y.p_needs_eyes - x.p_needs_eyes);
  const logic = rows.filter((r) => r.behaviour_change > 0.5 && !r.is_test);
  const summary = {
    kind: prAnswers.kind.choice, kind_probs: prAnswers.kind.probabilities, risk: prAnswers.risk.score,
    breaking_change: prAnswers.breaking_change.noul, needs_migration: prAnswers.needs_migration.noul, description_matches: prAnswers.description_matches.noul,
    sensitive: rows.filter((r) => r.sensitive_area > 0.5).length, weakens_errors: rows.filter((r) => r.removes_error_handling > 0.5).length, debug: rows.filter((r) => r.leftover_debug > 0.5).length,
    logic_hunks: logic.length, test_files: testFiles.length, needs_tests: logic.length > 0 && testFiles.length === 0 && !["docs", "chore"].includes(prAnswers.kind.choice),
    escalate: byEyes.filter((r) => r.p_needs_eyes >= 0.5 && !r.is_test).slice(0, 10),
    escalate_tests: byEyes.filter((r) => r.p_needs_eyes >= 0.5 && r.is_test).slice(0, 5),
    reviewed: rows.length, skipped, total_hunks: all.length,
  };
  return { pr: { url: meta.html_url, title: meta.title, number: meta.number }, prAnswers, rows, summary };
}
