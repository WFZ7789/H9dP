"""Live documentation layout and relative-link checks; ignore run snapshots."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def test_root_markdown_contains_only_project_entry_and_instructions():
    assert {path.name for path in ROOT.glob("*.md")} == {"README.md", "AGENTS.md"}
    assert (ROOT / "docs" / "README.md").is_file()


def test_each_model_has_an_index_and_t1a_separates_plan_from_results():
    for name in ("T1A", "P3", "P3.1", "v2", "M1"):
        assert (ROOT / "docs" / name / "README.md").is_file()
    for name in ("PLAN.md", "FIT24_RESULTS.md"):
        assert (ROOT / "docs" / "T1A" / name).is_file()


def test_live_markdown_relative_file_links_resolve():
    documents = [ROOT / "README.md", ROOT / "AGENTS.md", *(ROOT / "docs").rglob("*.md")]
    missing = []
    for document in documents:
        body = re.sub(r"```[\s\S]*?```", "", document.read_text())
        for raw in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
            url = urlsplit(raw.strip("<>"))
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            target = (document.parent / unquote(url.path)).resolve()
            if not target.exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw}")
    assert not missing, "\n".join(missing)
