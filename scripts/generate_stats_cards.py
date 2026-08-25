"""Generates self-hosted stats + top-languages SVG cards.

Replaces github-readme-stats.vercel.app / github-readme-activity-graph /
github-profile-trophy — all three went down (503/402) under real load,
which is exactly the failure this script exists to avoid. Runs in CI
(see .github/workflows/update-stats.yml) and commits static SVGs to the
`output` branch, mirroring the snake-animation workflow's proven pattern.
"""
import os
import urllib.request
import json

USERNAME = "shaikn6"
TOKEN = os.environ.get("GH_TOKEN", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dist")

BG = "#0a0a0b"
ACCENT = "#c8b08a"
ACCENT_2 = "#b59b73"
TEXT = "#e8e6e1"
MUTED = "#8a8781"
BORDER = "#232320"

GRAPHQL_QUERY = """
{
  user(login: "%s") {
    followers { totalCount }
    contributionsCollection {
      contributionCalendar { totalContributions }
    }
    repositories(first: 100, ownerAffiliations: OWNER, isFork: false, privacy: PUBLIC) {
      totalCount
      nodes {
        stargazerCount
        languages(first: 6, orderBy: {field: SIZE, direction: DESC}) {
          edges { size node { name color } }
        }
      }
    }
  }
}
""" % USERNAME


def fetch_stats() -> dict:
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": GRAPHQL_QUERY}).encode(),
        headers={
            "Authorization": f"bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    return data["data"]["user"]


def aggregate_languages(repos: list[dict], top_n: int = 6) -> list[tuple[str, str, int]]:
    totals: dict[str, dict] = {}
    for repo in repos:
        for edge in repo["languages"]["edges"]:
            name = edge["node"]["name"]
            color = edge["node"]["color"] or MUTED
            totals.setdefault(name, {"color": color, "size": 0})
            totals[name]["size"] += edge["size"]
    ranked = sorted(totals.items(), key=lambda kv: kv[1]["size"], reverse=True)[:top_n]
    total_size = sum(v["size"] for _, v in ranked) or 1
    return [(name, v["color"], round(v["size"] / total_size * 100, 1)) for name, v in ranked]


def render_stats_card(public_repos: int, stars: int, contributions: int, followers: int) -> str:
    stats = [
        (str(public_repos), "Public Repos"),
        (str(stars), "Stars Earned"),
        (f"{contributions:,}", "Contributions"),
        (str(followers), "Followers"),
    ]
    col_w = 205
    width = col_w * len(stats)
    height = 120
    cols = "".join(
        f'''
    <g transform="translate({i * col_w}, 0)">
      <text x="{col_w / 2}" y="52" text-anchor="middle" font-family="'Segoe UI', system-ui, sans-serif"
            font-size="34" font-weight="700" fill="{ACCENT}">{value}</text>
      <text x="{col_w / 2}" y="78" text-anchor="middle" font-family="'Segoe UI', system-ui, sans-serif"
            font-size="13" fill="{MUTED}" letter-spacing="0.5">{label}</text>
    </g>'''
        for i, (value, label) in enumerate(stats)
    )
    dividers = "".join(
        f'<line x1="{i * col_w}" y1="30" x2="{i * col_w}" y2="90" stroke="{BORDER}" stroke-width="1"/>'
        for i in range(1, len(stats))
    )
    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" rx="12" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>
  {dividers}
  {cols}
</svg>'''


def render_langs_card(langs: list[tuple[str, str, float]]) -> str:
    pad = 16
    bar_w = 400
    width = bar_w + pad * 2
    bar_y = 40
    bar_h = 14
    height = 40 + bar_h + 24 + -(-len(langs) // 2) * 26 + 20
    x = 0.0
    segments = ""
    for name, color, pct in langs:
        seg_w = bar_w * (pct / 100)
        segments += f'<rect x="{x:.2f}" y="{bar_y}" width="{seg_w:.2f}" height="{bar_h}" fill="{color}"/>'
        x += seg_w
    col_w = bar_w / 2
    legend = ""
    for i, (name, color, pct) in enumerate(langs):
        col = i % 2
        row = i // 2
        lx = col * col_w
        ly = bar_y + bar_h + 30 + row * 26
        legend += f'''
    <circle cx="{lx + 7}" cy="{ly - 5}" r="5" fill="{color}"/>
    <text x="{lx + 20}" y="{ly}" font-family="'Segoe UI', system-ui, sans-serif" font-size="13" fill="{TEXT}">{name}</text>
    <text x="{lx + col_w - 8}" y="{ly}" text-anchor="end" font-family="'Segoe UI', system-ui, sans-serif" font-size="13" fill="{MUTED}">{pct}%</text>'''
    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{width}" height="{height}" rx="12" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>
  <text x="{pad}" y="24" font-family="'Segoe UI', system-ui, sans-serif" font-size="14" font-weight="700" fill="{ACCENT}">Top Languages</text>
  <g transform="translate({pad}, 0)">
    <rect x="0" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="{BORDER}"/>
    <clipPath id="barclip"><rect x="0" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="4"/></clipPath>
    <g clip-path="url(#barclip)">{segments}</g>
    {legend}
  </g>
</svg>'''


def main() -> None:
    user = fetch_stats()
    repos = user["repositories"]["nodes"]
    public_repos = user["repositories"]["totalCount"]
    stars = sum(r["stargazerCount"] for r in repos)
    contributions = user["contributionsCollection"]["contributionCalendar"]["totalContributions"]
    followers = user["followers"]["totalCount"]
    langs = aggregate_languages(repos)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "stats-card.svg"), "w") as f:
        f.write(render_stats_card(public_repos, stars, contributions, followers))
    with open(os.path.join(OUT_DIR, "langs-card.svg"), "w") as f:
        f.write(render_langs_card(langs))
    print(f"Generated cards: {public_repos} repos, {stars} stars, {contributions} contributions, {followers} followers")


if __name__ == "__main__":
    main()
