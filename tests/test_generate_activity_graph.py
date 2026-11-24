import xml.etree.ElementTree as ET

from scripts import generate_activity_graph


def test_render_activity_graph_uses_calendar_days_and_counts(monkeypatch):
    def unexpected_urlopen(*args, **kwargs):
        raise AssertionError("Rendering should not make network requests")

    monkeypatch.setattr(generate_activity_graph.urllib.request, "urlopen", unexpected_urlopen)
    calendar = {
        "totalContributions": 5,
        "weeks": [
            {"contributionDays": [
                {"date": "2026-01-30", "contributionCount": 0},
                {"date": "2026-01-31", "contributionCount": 1},
            ]},
            {"contributionDays": [
                {"date": "2026-02-01", "contributionCount": 4},
            ]},
        ],
    }

    svg = ET.fromstring(generate_activity_graph.render_activity_graph(calendar))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.tag == "{http://www.w3.org/2000/svg}svg"
    labels = [element.text for element in svg.findall("svg:text", namespace)]
    assert "5 in the last year" in labels
    assert "4 on 2026-02-01" in labels
    assert labels[-2:] == ["Jan", "Feb"]
    line = svg.find("svg:path[@fill='none']", namespace)
    assert line is not None
    # One point per day, left to right; a day with more contributions sits higher (smaller y).
    points = [tuple(map(float, p.split())) for p in line.attrib["d"].lstrip("M ").split(" L ")]
    xs, ys = zip(*points)
    assert len(points) == 3
    assert list(xs) == sorted(xs)
    assert ys[0] > ys[1] > ys[2]
