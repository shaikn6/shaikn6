"""Generates a self-hosted contribution activity graph SVG.

Replaces github-readme-activity-graph.vercel.app, which returns 402
(payment required) under real load — the same failure that took down
github-readme-stats.vercel.app. Runs in CI (see
.github/workflows/profile-assets.yml) and commits the SVG to the
`output` branch, mirroring the snake-animation workflow's proven pattern.
"""
import os
import urllib.request
import json
import math

USERNAME = "shaikn6"
TOKEN = os.environ.get("GH_TOKEN", "")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "profile-assets")

BG = "#0a0a0b"
LINE = "#39d353"
FILL_TOP = "rgba(57, 211, 83, 0.35)"
FILL_BOTTOM = "rgba(57, 211, 83, 0)"
TEXT = "#e8e6e1"
MUTED = "#8a8781"
BORDER = "#232320"

FONT_MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', Consolas, monospace"

WIDTH = 860
HEIGHT = 220
PAD_L = 36
PAD_R = 20
PAD_T = 56
PAD_B = 36

GRAPHQL_QUERY = """
{
  user(login: "%s") {
    contributionsCollection {
      contributionCalendar {
        totalContributions
        weeks {
          contributionDays { date contributionCount }
        }
      }
    }
  }
}
""" % USERNAME

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fetch_calendar() -> dict:
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
    return data["data"]["user"]["contributionsCollection"]["contributionCalendar"]


def render_activity_graph(calendar: dict) -> str:
    days = [
        (d["date"], d["contributionCount"])
        for week in calendar["weeks"]
        for d in week["contributionDays"]
    ]
    total = calendar["totalContributions"]
    counts = [c for _, c in days]
    peak_count = max(counts) if counts else 0
    peak_idx = counts.index(peak_count) if counts else 0
    n = len(days)

    plot_w = WIDTH - PAD_L - PAD_R
    plot_h = HEIGHT - PAD_T - PAD_B
    # sqrt scale: a handful of bulk-commit days (600+) would otherwise flatten
    # everything else in the year to near-zero on a linear axis.
    max_c = max(math.sqrt(peak_count), 1)

    def x_at(i: int) -> float:
        return PAD_L + (i / max(n - 1, 1)) * plot_w

    def y_at(c: int) -> float:
        return PAD_T + plot_h - (math.sqrt(c) / max_c) * plot_h

    points = [(x_at(i), y_at(c)) for i, (_, c) in enumerate(days)]
    line_path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    area_path = (
        f"M {points[0][0]:.1f} {PAD_T + plot_h:.1f} L "
        + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
        + f" L {points[-1][0]:.1f} {PAD_T + plot_h:.1f} Z"
    )

    # Month tick labels — one per calendar month that starts within range,
    # skipping any that would land within 28px of the previous label (the
    # first two months in a 52-week window can otherwise fall almost on
    # top of each other).
    month_ticks = ""
    seen_months = set()
    last_x = -1000.0
    for i, (d, _) in enumerate(days):
        y, m, _ = d.split("-")
        key = (y, m)
        if key in seen_months:
            continue
        x = x_at(i)
        if x - last_x < 28:
            seen_months.add(key)
            continue
        seen_months.add(key)
        last_x = x
        month_ticks += (
            f'<text x="{x:.1f}" y="{HEIGHT - 14}" font-family="{FONT_MONO}" '
            f'font-size="10" fill="{MUTED}">{MONTH_NAMES[int(m) - 1]}</text>'
        )

    peak_x, peak_y = points[peak_idx]
    peak_date = days[peak_idx][0]
    peak_label = (
        f'<circle cx="{peak_x:.1f}" cy="{peak_y:.1f}" r="3.5" fill="{LINE}"/>'
        f'<text x="{peak_x:.1f}" y="{peak_y - 12:.1f}" text-anchor="middle" '
        f'font-family="{FONT_MONO}" font-size="10" fill="{TEXT}">{peak_count} on {peak_date}</text>'
    )

    baseline_y = PAD_T + plot_h

    return f'''<svg width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{WIDTH}" height="{HEIGHT}" rx="10" fill="{BG}" stroke="{BORDER}" stroke-width="1"/>
  <text x="28" y="30" font-family="{FONT_MONO}" font-size="11" letter-spacing="2" fill="{LINE}">CONTRIBUTION ACTIVITY</text>
  <text x="{WIDTH - 20}" y="30" text-anchor="end" font-family="{FONT_MONO}" font-size="11" fill="{MUTED}">{total:,} in the last year</text>
  <line x1="{PAD_L}" y1="{baseline_y:.1f}" x2="{WIDTH - PAD_R}" y2="{baseline_y:.1f}" stroke="{BORDER}" stroke-width="1"/>
  <defs>
    <linearGradient id="fade" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{FILL_TOP}"/>
      <stop offset="100%" stop-color="{FILL_BOTTOM}"/>
    </linearGradient>
  </defs>
  <path d="{area_path}" fill="url(#fade)"/>
  <path d="{line_path}" fill="none" stroke="{LINE}" stroke-width="1.4"/>
  {peak_label}
  {month_ticks}
</svg>'''


def main() -> None:
    calendar = fetch_calendar()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "activity-graph.svg"), "w") as f:
        f.write(render_activity_graph(calendar))
    print(f"Generated activity graph: {calendar['totalContributions']} contributions")


if __name__ == "__main__":
    main()
