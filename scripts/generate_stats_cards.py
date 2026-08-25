"""Generates self-hosted stats + top-languages SVG cards.

Replaces github-readme-stats.vercel.app / github-readme-activity-graph /
github-profile-trophy — all three went down (503/402) under real load,
which is exactly the failure this script exists to avoid. Runs in CI
(see .github/workflows/profile-assets.yml) and commits static SVGs to the
`output` branch, mirroring the snake-animation workflow's proven pattern.

Visual language matches the "audit ledger" identity used on the portfolio
site (numbered index, hairline dividers, mono labels) instead of generic
flat stat boxes.
"""
import os
import urllib.request
import json

USERNAME = "shaikn6"
TOKEN = os.environ.get("GH_TOKEN", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "profile-assets")

BG = "#0a0a0b"
ACCENT = "#c8b08a"
ACCENT_2 = "#b59b73"
TEXT = "#e8e6e1"
MUTED = "#8a8781"
FAINT = "#5f5c56"
BORDER = "#232320"
CARD_W = 860

FONT_MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace"
FONT_DISPLAY = "'Georgia', 'Times New Roman', serif"
FONT_SANS = "-apple-system, 'Segoe UI', system-ui, sans-serif"

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

# Minimal line icons (24x24 viewBox), same set used in the portfolio's Metrics
# component, so the profile README and the portfolio site read as one system.
ICONS = {
    "repos": '<path d="M9 6 4 12l5 6M15 6l5 6-5 6" stroke-linecap="round" stroke-linejoin="round"/>',
    "stars": '<path d="M12 3.5 14.6 9.3 21 10 16.2 14.3 17.5 20.5 12 17.2 6.5 20.5 7.8 14.3 3 10 9.4 9.3 12 3.5Z" stroke-linejoin="round"/>',
    "contributions": '<path d="M4 20V13M9.5 20V9M15 20V15M20 20V5" stroke-linecap="round"/>',
    "followers": '<circle cx="9" cy="9" r="3.5"/><path d="M3.5 20c0-3.6 2.5-6 5.5-6s5.5 2.4 5.5 6" stroke-linecap="round"/><circle cx="17" cy="8" r="2.6"/><path d="M15.3 6c1.7-.6 3.8.4 4.4 2.3" stroke-linecap="round"/>',
}


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
        ("repos", str(public_repos), "Public Repos"),
        ("stars", str(stars), "Stars Earned"),
        ("contributions", f"{contributions:,}", "Contributions"),
        ("followers", str(followers), "Followers"),
    ]
    col_w = CARD_W / len(stats)
    height = 156
    cols = ""
    dividers = ""
    for i, (icon_key, value, label) in enumerate(stats):
        cx = i * col_w
        idx = str(i + 1).zfill(2)
        cols += f'''
    <g transform="translate({cx:.1f}, 0)">
      <text x="{col_w - 20:.1f}" y="24" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="{FAINT}">{idx}</text>
      <g transform="translate(28, 34)" stroke="{ACCENT_2}" stroke-width="1.4" fill="none">{ICONS[icon_key]}</g>
      <text x="28" y="102" font-family="{FONT_DISPLAY}" font-size="32" font-weight="700" fill="{ACCENT}" letter-spacing="-0.5">{value}</text>
      <text x="28" y="126" font-family="{FONT_MONO}" font-size="11" letter-spacing="1" fill="{MUTED}">{label.upper()}</text>
    </g>'''
        if i > 0:
            dividers += f'<line x1="{cx:.1f}" y1="0" x2="{cx:.1f}" y2="{height}" stroke="{BORDER}" stroke-width="1"/>'
    return f'''<svg width="{CARD_W}" height="{height}" viewBox="0 0 {CARD_W} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{CARD_W}" height="{height}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>
  <line x1="0" y1="0" x2="{CARD_W}" y2="0" stroke="{BORDER}" stroke-width="1"/>
  {dividers}
  {cols}
</svg>'''


def render_langs_card(langs: list[tuple[str, str, float]]) -> str:
    pad = 28
    bar_w = CARD_W - pad * 2
    bar_y = 56
    bar_h = 10
    rows_per_col = -(-len(langs) // 2)
    height = bar_y + bar_h + 30 + rows_per_col * 28 + 24

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
        ly = bar_y + bar_h + 32 + row * 28
        legend += f'''
    <rect x="{lx:.1f}" y="{ly - 11}" width="3" height="14" fill="{color}"/>
    <text x="{lx + 14:.1f}" y="{ly}" font-family="{FONT_MONO}" font-size="13" fill="{TEXT}">{name}</text>
    <text x="{lx + col_w - 8:.1f}" y="{ly}" text-anchor="end" font-family="{FONT_MONO}" font-size="12" fill="{MUTED}">{pct}%</text>'''

    return f'''<svg width="{CARD_W}" height="{height}" viewBox="0 0 {CARD_W} {height}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{CARD_W}" height="{height}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>
  <text x="{pad}" y="30" font-family="{FONT_MONO}" font-size="11" letter-spacing="2" fill="{ACCENT}">TOP LANGUAGES</text>
  <line x1="{pad}" y1="40" x2="{CARD_W - pad}" y2="40" stroke="{BORDER}" stroke-width="1"/>
  <g transform="translate({pad}, 0)">
    <rect x="0" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="2" fill="{BORDER}"/>
    <clipPath id="barclip"><rect x="0" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="2"/></clipPath>
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
