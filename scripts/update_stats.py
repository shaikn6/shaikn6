"""Refreshes the public-repo count in README.md between marker comments.

Run in CI (see .github/workflows/update-stats.yml) so the footer never
goes stale again — it previously claimed "21 public repos" while the
account actually had 14.
"""
import os
import re
import urllib.request
import json

USERNAME = "shaikn6"
README_PATH = os.path.join(os.path.dirname(__file__), "..", "README.md")
START = "<!--REPO_COUNT_START-->"
END = "<!--REPO_COUNT_END-->"


def public_repo_count() -> int:
    req = urllib.request.Request(
        f"https://api.github.com/users/{USERNAME}",
        headers={"Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["public_repos"]


def main() -> None:
    count = public_repo_count()
    with open(README_PATH, encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    replacement = f"{START}{count} public repos{END}"
    new_content = pattern.sub(replacement, content)

    if new_content != content:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated repo count to {count}")
    else:
        print("No change")


if __name__ == "__main__":
    main()
