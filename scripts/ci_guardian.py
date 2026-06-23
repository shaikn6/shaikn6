#!/usr/bin/env python3
"""
CI Guardian — scans all shaikn6 repos for CI failures, classifies them,
and auto-fixes code/config issues via Claude API, opening a PR with the patch.

Requires env vars:
  GH_TOKEN      — PAT with repo + workflow scopes
  ANTHROPIC_API_KEY — Claude API key
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import anthropic

GH_OWNER = "shaikn6"
MAX_LOG_CHARS = 8000
MAX_REPOS_PER_RUN = 50


# ── helpers ──────────────────────────────────────────────────────────────────

def gh(*args, check=True) -> str:
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True,
        env={**os.environ}
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def gh_json(*args) -> list | dict:
    return json.loads(gh(*args))


def list_repos() -> list[str]:
    repos = gh_json("repo", "list", GH_OWNER, "--limit", "100", "--json", "name")
    return [r["name"] for r in repos]


def latest_failed_run(repo: str) -> dict | None:
    try:
        runs = gh_json(
            "run", "list", "--repo", f"{GH_OWNER}/{repo}",
            "--limit", "5", "--json", "status,conclusion,name,databaseId,workflowName"
        )
    except Exception:
        return None
    for run in runs:
        if run.get("conclusion") in ("failure", "timed_out"):
            return run
    return None


def get_run_logs(repo: str, run_id: int) -> str:
    try:
        result = subprocess.run(
            ["gh", "run", "view", str(run_id), "--repo", f"{GH_OWNER}/{repo}", "--log-failed"],
            capture_output=True, text=True, timeout=30
        )
        log = result.stdout
    except Exception:
        log = ""
    return log[-MAX_LOG_CHARS:] if len(log) > MAX_LOG_CHARS else log


def classify_failure(log: str) -> str:
    billing_markers = [
        "recent account payments have failed",
        "spending limit needs to be increased",
        "job was not started because",
    ]
    if any(m in log for m in billing_markers):
        return "billing"
    if not log.strip():
        return "unknown"
    return "code"


def get_workflow_file(repo: str, workflow_name: str) -> str:
    try:
        files = gh_json(
            "api", f"repos/{GH_OWNER}/{repo}/contents/.github/workflows"
        )
        for f in files:
            content_b64 = gh_json(
                "api", f"repos/{GH_OWNER}/{repo}/contents/{f['path']}"
            ).get("content", "")
            import base64
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            if workflow_name.lower() in f["name"].lower() or "ci" in f["name"].lower():
                return f"# File: {f['path']}\n{content}"
    except Exception:
        pass
    return ""


def call_claude(log: str, workflow: str, repo: str) -> str:
    client = anthropic.Anthropic()
    prompt = textwrap.dedent(f"""
        You are an expert DevOps engineer. A GitHub Actions CI workflow in repo `{repo}` is failing.

        ## Failing workflow file
        {workflow[:3000] if workflow else "(not available)"}

        ## Failure log (tail)
        {log}

        ## Task
        1. Identify the root cause in one sentence.
        2. Provide the minimal fix as a JSON object with this shape:
           {{
             "root_cause": "...",
             "files": [
               {{"path": "relative/path/to/file", "content": "full new file content"}}
             ]
           }}

        Rules:
        - Fix only what is broken. No refactoring.
        - If the fix requires a secret or manual action, set "files": [] and explain in root_cause.
        - Return ONLY the JSON object, no markdown fences.
    """).strip()

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def apply_fix(repo: str, fix: dict) -> str | None:
    """Create a branch and PR with the fix. Returns PR URL or None."""
    files = fix.get("files", [])
    if not files:
        return None

    branch = f"ci-guardian/fix-{int(time.time())}"

    # Create branch from default branch
    try:
        default = gh_json("api", f"repos/{GH_OWNER}/{repo}")["default_branch"]
        ref_sha = gh_json(
            "api", f"repos/{GH_OWNER}/{repo}/git/ref/heads/{default}"
        )["object"]["sha"]

        gh("api", "--method", "POST",
           f"repos/{GH_OWNER}/{repo}/git/refs",
           "--field", f"ref=refs/heads/{branch}",
           "--field", f"sha={ref_sha}")
    except Exception as e:
        print(f"  [guardian] branch create failed: {e}")
        return None

    # Push each file
    import base64
    for f in files:
        path = f["path"]
        content_b64 = base64.b64encode(f["content"].encode()).decode()
        # Get existing SHA if file exists
        try:
            existing = gh_json("api", f"repos/{GH_OWNER}/{repo}/contents/{path}?ref={branch}")
            sha_arg = ["--field", f"sha={existing['sha']}"]
        except Exception:
            sha_arg = []

        try:
            gh("api", "--method", "PUT",
               f"repos/{GH_OWNER}/{repo}/contents/{path}",
               "--field", f"message=fix(ci): auto-fix by CI Guardian",
               "--field", f"content={content_b64}",
               "--field", f"branch={branch}",
               *sha_arg)
        except Exception as e:
            print(f"  [guardian] file push failed {path}: {e}")
            return None

    # Open PR
    root_cause = fix.get("root_cause", "CI failure detected by guardian")
    try:
        pr = gh_json(
            "api", "--method", "POST",
            f"repos/{GH_OWNER}/{repo}/pulls",
            "--field", f"title=fix(ci): {root_cause[:72]}",
            "--field", f"body=## CI Guardian Auto-Fix\n\n**Root cause:** {root_cause}\n\nThis PR was opened automatically by the CI Guardian agent.",
            "--field", f"head={branch}",
            "--field", f"base={default}",
        )
        return pr.get("html_url")
    except Exception as e:
        print(f"  [guardian] PR create failed: {e}")
        return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    repos = list_repos()[:MAX_REPOS_PER_RUN]
    print(f"Scanning {len(repos)} repos...")

    billing_blocked = []
    fixed = []
    skipped = []
    errors = []

    for repo in repos:
        run = latest_failed_run(repo)
        if not run:
            continue

        run_id = run["databaseId"]
        workflow = run.get("workflowName", "")
        print(f"\n[{repo}] FAIL: {workflow} (run {run_id})")

        log = get_run_logs(repo, run_id)
        kind = classify_failure(log)
        print(f"  classified: {kind}")

        if kind == "billing":
            billing_blocked.append(repo)
            continue

        if kind == "unknown":
            skipped.append(repo)
            continue

        # Code/config failure — call Claude
        wf_file = get_workflow_file(repo, workflow)
        try:
            raw = call_claude(log, wf_file, repo)
            fix = json.loads(raw)
        except Exception as e:
            print(f"  Claude parse error: {e}")
            errors.append(repo)
            continue

        print(f"  root cause: {fix.get('root_cause','?')}")

        if not fix.get("files"):
            print(f"  no auto-fix possible (manual action required)")
            skipped.append(repo)
            continue

        pr_url = apply_fix(repo, fix)
        if pr_url:
            print(f"  PR opened: {pr_url}")
            fixed.append((repo, pr_url))
        else:
            errors.append(repo)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Auto-fixed (PR opened): {len(fixed)}")
    for repo, url in fixed:
        print(f"    {repo}: {url}")
    print(f"  Billing-blocked (need account fix): {len(billing_blocked)}")
    print(f"  Skipped (no auto-fix): {len(skipped)}")
    print(f"  Errors: {len(errors)}")

    # Output for GH Actions summary
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "w") as f:
            f.write(f"## CI Guardian Report\n\n")
            f.write(f"| Status | Count |\n|---|---|\n")
            f.write(f"| ✅ Auto-fixed | {len(fixed)} |\n")
            f.write(f"| 💳 Billing blocked | {len(billing_blocked)} |\n")
            f.write(f"| ⏭️ Skipped | {len(skipped)} |\n")
            f.write(f"| ❌ Errors | {len(errors)} |\n\n")
            if fixed:
                f.write("### PRs Opened\n")
                for repo, url in fixed:
                    f.write(f"- [{repo}]({url})\n")
            if billing_blocked:
                f.write(f"\n### Billing Blocked\n")
                f.write(", ".join(billing_blocked) + "\n")


if __name__ == "__main__":
    main()
