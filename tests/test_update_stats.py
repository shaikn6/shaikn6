import io
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import update_stats


def test_main_updates_count_once(tmp_path, monkeypatch, capsys):
    readme = tmp_path / "README.md"
    start, end = update_stats.START, update_stats.END
    readme.write_text(f"Before\n{start}\n3 public repos\n{end}\nAfter\n", encoding="utf-8")
    monkeypatch.setattr(update_stats, "README_PATH", readme)
    monkeypatch.setattr(update_stats, "public_repo_count", lambda: 7)

    update_stats.main()

    expected = f"Before\n{start}7 public repos{end}\nAfter\n"
    assert readme.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == "Updated repo count to 7\n"
    modified = readme.stat().st_mtime_ns

    update_stats.main()

    assert readme.read_text(encoding="utf-8") == expected
    assert readme.stat().st_mtime_ns == modified
    assert capsys.readouterr().out == "No change\n"


@pytest.mark.parametrize("content", [
    "Before\n3 public repos\nAfter\n",
    "<!--REPO_COUNT_START-->3 public repos\n",
    "3 public repos<!--REPO_COUNT_END-->\n",
])
def test_main_leaves_readme_without_a_marker_pair_unchanged(
    content, tmp_path, monkeypatch, capsys
):
    readme = tmp_path / "README.md"
    readme.write_text(content, encoding="utf-8")
    modified = readme.stat().st_mtime_ns
    monkeypatch.setattr(update_stats, "README_PATH", readme)
    monkeypatch.setattr(update_stats, "public_repo_count", lambda: 7)

    update_stats.main()

    assert readme.read_text(encoding="utf-8") == content
    assert readme.stat().st_mtime_ns == modified
    assert capsys.readouterr().out == "No change\n"


def test_public_repo_count_excludes_forks_and_follows_full_pages(monkeypatch):
    pages = [
        [{"fork": True}] * 99 + [{"fork": False}],
        [{"fork": False}] * 100,
        [{"fork": True}, {"fork": False}],
    ]
    requested_pages = []

    def fake_urlopen(request):
        query = parse_qs(urlsplit(request.full_url).query)
        page = int(query["page"][0])
        requested_pages.append(page)
        assert query["per_page"] == ["100"]
        return io.BytesIO(json.dumps(pages[page - 1]).encode())

    monkeypatch.setattr(update_stats.urllib.request, "urlopen", fake_urlopen)

    assert update_stats.public_repo_count() == 102
    assert requested_pages == [1, 2, 3]
