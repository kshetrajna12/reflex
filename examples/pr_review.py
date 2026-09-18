"""AI PR triage with a System One model: typed judgments over a pull request, no prose.

Two levels of state:
  * PR level  – title, description, file list, stats  -> kind, risk, breaking, needs migration…
  * hunk level – one state per diff hunk               -> sensitive area, removed error handling,
                                                          leftover debug, public API, needs eyes…
Everything is aggregated in code (max over hunks, thresholds you choose) and the hunks
that score highest are the ones to hand to a reasoning model or a human reviewer.

    uv run python examples/pr_review.py --repo pydantic/pydantic --pr 12345
    uv run python examples/pr_review.py --diff my.patch --title "..." --body "..."
    uv run python examples/pr_review.py --repo o/r --pr N --http     # against reflex-serve
    uv run python examples/pr_review.py --repo o/r --pr N --json out.json
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------- questions

PR_QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What kind of change is this pull request, judging from the title, description and the files touched?",
        "criteria": {
            "feature": "adds new user-facing behaviour or capability",
            "bugfix": "corrects incorrect behaviour",
            "refactor": "restructures code without changing behaviour",
            "docs": "documentation, comments, changelog only",
            "tests": "adds or changes tests only",
            "chore": "build, CI, dependencies, formatting, release plumbing",
        },
    },
    "description_matches": {
        "type": "noul",
        "instructions": "Does the `body` accurately describe what the changed `files` and `stats` suggest the PR does?",
        "criteria": {
            "true": "the description explains the change and nothing important is left unsaid",
            "false": "the description is missing, generic, or does not match the files touched",
        },
    },
    "breaking_change": {
        "type": "noul",
        "instructions": "Is this likely a breaking change for users of this project (public API, config format, CLI flags, behaviour that callers rely on)?",
    },
    "needs_migration": {
        "type": "noul",
        "instructions": "Does this change need a data or schema migration, or a coordinated rollout (e.g. database, stored formats, protocol versions)?",
    },
    "risk": {
        "type": "score",
        "instructions": "How risky is merging this PR, considering scope, the areas touched and how easy it is to reason about?",
        "criteria": [
            "trivial: docs, comments, formatting, isolated tests",
            "low: small, local, easy to reason about",
            "medium: touches shared logic or several files; a careful review is warranted",
            "high: core behaviour, data handling, security, or concurrency; could break users",
        ],
    },
}

HUNK_QUESTIONS = {
    "sensitive_area": {
        "type": "noul",
        "instructions": "Does this hunk touch a sensitive area: authentication, authorization, permissions, secrets or credentials, payments, cryptography, or personal data?",
    },
    "removes_error_handling": {
        "type": "noul",
        "instructions": "Does this hunk remove or weaken error handling, validation, or safety checks (e.g. deleting a try/except, an assert, a bounds check, or broadening a caught exception)?",
    },
    "leftover_debug": {
        "type": "noul",
        "instructions": "Does this hunk add leftover debugging or temporary code: print/console.log statements, commented-out code, TODO/FIXME/HACK markers, hard-coded test values?",
    },
    "public_api_change": {
        "type": "noul",
        "instructions": "Does this hunk change a public interface: a function or class signature, an exported name, a CLI flag, a config key, a wire or file format?",
    },
    "behaviour_change": {
        "type": "noul",
        "instructions": "Does this hunk change runtime behaviour (as opposed to a pure rename, formatting, comments, or type annotations)?",
        "criteria": {
            "true": "control flow, values, or side effects differ after the change",
            "false": "purely cosmetic, comments, docs, renames, or typing",
        },
    },
    "generated": {
        "type": "noul",
        "instructions": "Does this hunk look machine-generated or boilerplate (lockfiles, snapshots, vendored code, auto-formatted output) rather than hand-written logic?",
    },
    "needs_eyes": {
        "type": "score",
        "instructions": "How much would a senior reviewer want to look closely at this hunk?",
        "criteria": [
            "no: trivial or mechanical; skim at most",
            "a glance: straightforward change, low chance of a bug",
            "a careful read: non-trivial logic, edge cases possible",
            "definitely: subtle, risky, or hard to reason about; discuss before merging",
        ],
    },
}

SKIP_GLOBS = [
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "go.sum",
    "Cargo.lock",
    "*.min.js",
    "*.min.css",
    "*.svg",
    "*.png",
    "*.jpg",
    "*.gif",
    "*.pdf",
    "*.snap",
    "*.ipynb",
    "dist/*",
    "build/*",
    "vendor/*",
    "node_modules/*",
]
TEST_PATTERNS = re.compile(
    r"(^|/)(tests?|__tests__|spec|specs)(/|$)|_test\.|\.test\.|\.spec\.|test_[^/]*\.py$"
)

# ----------------------------------------------------------------------------- diff parsing


@dataclass
class Hunk:
    path: str
    header: str
    lines: list[str] = field(default_factory=list)
    added: int = 0
    removed: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def parse_diff(diff: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    path = None
    cur: Hunk | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            m = re.match(r"diff --git a/(.*?) b/(.*)$", line)
            path = m.group(2) if m else line.split()[-1].lstrip("b/")
            cur = None
        elif line.startswith("@@"):
            cur = Hunk(path=path or "?", header=line, lines=[line])
            hunks.append(cur)
        elif cur is not None and not line.startswith(
            (
                "+++",
                "---",
                "index ",
                "similarity",
                "rename ",
                "new file",
                "deleted file",
                "Binary files",
            )
        ):
            cur.lines.append(line)
            if line.startswith("+"):
                cur.added += 1
            elif line.startswith("-"):
                cur.removed += 1
    return hunks


def skip_path(path: str) -> bool:
    return any(
        fnmatch.fnmatch(path, g) or fnmatch.fnmatch(path.split("/")[-1], g) for g in SKIP_GLOBS
    )


def trim(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: max_chars * 2 // 3]
    tail = text[-max_chars // 3 :]
    return f"{head}\n... [{len(text) - max_chars} chars omitted] ...\n{tail}"


# ----------------------------------------------------------------------------- fetch


def gh_json(args: list[str]):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def fetch_pr(repo: str, number: int) -> dict:
    meta = gh_json(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "title,body,author,labels,files,additions,deletions,changedFiles,baseRefName,url",
        ]
    )
    diff = subprocess.check_output(["gh", "pr", "diff", str(number), "--repo", repo], text=True)
    return {"meta": meta, "diff": diff}


# ----------------------------------------------------------------------------- run


class Client:
    def __init__(self, model: str | None, http: bool, permutations: int):
        self.permutations = permutations
        if http:
            from reflex.client import Reflex

            self._http = Reflex()
            self.engine = None
        else:
            from reflex import Engine

            self.engine = Engine.load(model)
            self._http = None

    def ask(self, state, questions) -> dict:
        if self.engine is not None:
            from reflex import SystemOneRequest

            return self.engine.answer(
                SystemOneRequest(state=state, questions=questions, permutations=self.permutations)
            ).model_dump()["answers"]
        return self._http.systemone(state, questions, permutations=self.permutations)["answers"]


def review(pr: dict, client: Client, max_hunks: int, max_chars: int, verbose: bool) -> dict:
    meta, diff = pr["meta"], pr["diff"]
    files = meta.get("files") or []
    changed_paths = [f["path"] for f in files]
    test_files = [p for p in changed_paths if TEST_PATTERNS.search(p)]

    # --- PR level
    pr_state = {
        "title": meta.get("title"),
        "body": trim(meta.get("body") or "(no description)", 4000),
        "author": (meta.get("author") or {}).get("login"),
        "labels": [lb["name"] for lb in meta.get("labels") or []],
        "base_branch": meta.get("baseRefName"),
        "stats": {
            "files": meta.get("changedFiles"),
            "additions": meta.get("additions"),
            "deletions": meta.get("deletions"),
        },
        "files": [
            {"path": f["path"], "+": f.get("additions"), "-": f.get("deletions")}
            for f in files[:80]
        ],
    }
    t0 = time.perf_counter()
    pr_answers = client.ask(pr_state, PR_QUESTIONS)
    t_pr = time.perf_counter() - t0

    # --- hunk level
    hunks = [h for h in parse_diff(diff) if not skip_path(h.path)]
    skipped = len(parse_diff(diff)) - len(hunks)
    hunks.sort(key=lambda h: -(h.added + h.removed))
    hunks = hunks[:max_hunks]
    hunk_rows = []
    t1 = time.perf_counter()
    for i, h in enumerate(hunks):
        state = {
            "file": h.path,
            "is_test_file": bool(TEST_PATTERNS.search(h.path)),
            "hunk": trim(h.text, max_chars),
        }
        a = client.ask(state, HUNK_QUESTIONS)
        row = {
            "file": h.path,
            "header": h.header,
            "added": h.added,
            "removed": h.removed,
            **{k: (v["noul"] if v["type"] == "noul" else v["score"]) for k, v in a.items()},
            "needs_eyes_probs": a["needs_eyes"]["probabilities"],
        }
        hunk_rows.append(row)
        if verbose:
            print(
                f"  [{i + 1}/{len(hunks)}] {h.path} {h.header[:30]}  eyes={row['needs_eyes']:.2f}",
                file=sys.stderr,
            )
    t_hunks = time.perf_counter() - t1

    # --- aggregate in code
    def p_at_least(probs: dict, level: int) -> float:
        return sum(v for k, v in probs.items() if int(k) >= level)

    for r in hunk_rows:
        r["p_needs_eyes"] = p_at_least(r["needs_eyes_probs"], 2)
    # production code first; test hunks are listed separately (they need eyes for other reasons)
    escalate = sorted(
        hunk_rows, key=lambda r: (bool(TEST_PATTERNS.search(r["file"])), -r["p_needs_eyes"])
    )
    logic_hunks = [
        r for r in hunk_rows if r["behaviour_change"] > 0.5 and not TEST_PATTERNS.search(r["file"])
    ]
    summary = {
        "kind": pr_answers["kind"]["choice"],
        "kind_probs": pr_answers["kind"]["probabilities"],
        "risk": pr_answers["risk"]["score"],
        "risk_max_over_hunks": max((r["needs_eyes"] for r in hunk_rows), default=0.0),
        "breaking_change": pr_answers["breaking_change"]["noul"],
        "needs_migration": pr_answers["needs_migration"]["noul"],
        "description_matches": pr_answers["description_matches"]["noul"],
        "sensitive_hunks": sum(r["sensitive_area"] > 0.5 for r in hunk_rows),
        "hunks_removing_error_handling": sum(r["removes_error_handling"] > 0.5 for r in hunk_rows),
        "hunks_with_debug_leftovers": sum(r["leftover_debug"] > 0.5 for r in hunk_rows),
        "logic_hunks": len(logic_hunks),
        "test_files_changed": len(test_files),
        "needs_tests": bool(logic_hunks)
        and not test_files
        and pr_answers["kind"]["choice"] not in ("docs", "chore"),
        "escalate": [
            r for r in escalate if r["p_needs_eyes"] >= 0.5 and not TEST_PATTERNS.search(r["file"])
        ][:10],
        "escalate_tests": [
            r for r in escalate if r["p_needs_eyes"] >= 0.5 and TEST_PATTERNS.search(r["file"])
        ][:5],
        "hunks_reviewed": len(hunk_rows),
        "hunks_skipped": skipped,
        "timing_s": {"pr": round(t_pr, 2), "hunks": round(t_hunks, 2)},
    }
    return {
        "pr": {"url": meta.get("url"), "title": meta.get("title")},
        "pr_answers": pr_answers,
        "hunks": hunk_rows,
        "summary": summary,
    }


def print_report(out: dict) -> None:
    s = out["summary"]
    pct = lambda p: f"{100 * p:4.0f}%"
    print(f"\n{out['pr']['title']}\n{out['pr']['url'] or ''}\n")
    print(
        f"kind            {s['kind']:<10} "
        + "  ".join(
            f"{k} {pct(v)}" for k, v in sorted(s["kind_probs"].items(), key=lambda kv: -kv[1])[:3]
        )
    )
    print(
        f"risk            {s['risk']:.2f} / 3   (max hunk needs-eyes {s['risk_max_over_hunks']:.2f} / 3)"
    )
    print(
        f"breaking        {pct(s['breaking_change'])}      needs migration {pct(s['needs_migration'])}      description matches {pct(s['description_matches'])}"
    )
    print(
        f"hunks           {s['hunks_reviewed']} reviewed, {s['hunks_skipped']} skipped (lockfiles, binaries)   logic hunks {s['logic_hunks']}   test files changed {s['test_files_changed']}"
    )
    flags = []
    if s["needs_tests"]:
        flags.append("NEEDS TESTS: behaviour changes with no test files touched")
    if s["sensitive_hunks"]:
        flags.append(f"{s['sensitive_hunks']} hunk(s) in sensitive areas")
    if s["hunks_removing_error_handling"]:
        flags.append(f"{s['hunks_removing_error_handling']} hunk(s) weaken error handling")
    if s["hunks_with_debug_leftovers"]:
        flags.append(f"{s['hunks_with_debug_leftovers']} hunk(s) with debug leftovers")
    if s["breaking_change"] > 0.5:
        flags.append("likely breaking change")
    print("flags           " + ("; ".join(flags) if flags else "none"))
    print("\nescalate to a reasoning model / human (P(needs a careful read) >= 50%):")
    if not s["escalate"]:
        print("  nothing; all hunks look routine")
    for r in s["escalate"]:
        why = [
            k
            for k in (
                "sensitive_area",
                "removes_error_handling",
                "public_api_change",
                "leftover_debug",
            )
            if r[k] > 0.5
        ]
        print(
            f"  {pct(r['p_needs_eyes'])}  {r['file']}  {r['header'][:28]:<28} +{r['added']}/-{r['removed']}  {', '.join(why)}"
        )
    if s["escalate_tests"]:
        print("test hunks worth a look:")
        for r in s["escalate_tests"]:
            print(
                f"  {pct(r['p_needs_eyes'])}  {r['file']}  {r['header'][:28]:<28} +{r['added']}/-{r['removed']}"
            )
    print(
        f"\n{s['timing_s']['pr']:.1f}s PR level, {s['timing_s']['hunks']:.1f}s for {s['hunks_reviewed']} hunks"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--diff", help="local unified diff instead of a GitHub PR")
    ap.add_argument("--title", default="")
    ap.add_argument("--body", default="")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--http", action="store_true", help="use a running reflex-serve")
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--max-hunks", type=int, default=60)
    ap.add_argument("--max-chars", type=int, default=6000, help="per-hunk diff text cap")
    ap.add_argument("--json", help="write full results here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.diff:
        with open(args.diff) as f:
            diff = f.read()
        paths = sorted({h.path for h in parse_diff(diff)})
        pr = {
            "meta": {
                "title": args.title,
                "body": args.body,
                "files": [{"path": p} for p in paths],
                "changedFiles": len(paths),
            },
            "diff": diff,
        }
    elif args.repo and args.pr:
        pr = fetch_pr(args.repo, args.pr)
    else:
        ap.error("give --repo and --pr, or --diff")

    client = Client(args.model, args.http, args.permutations)
    out = review(pr, client, args.max_hunks, args.max_chars, args.verbose)
    print_report(out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
