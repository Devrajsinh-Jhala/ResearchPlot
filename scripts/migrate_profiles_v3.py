"""One-time deterministic migration of bundled profile JSON to schema v3."""

from __future__ import annotations

import json
from pathlib import Path


def migrate(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2:
        return
    data["schema_version"] = 3
    data["namespace"] = "researchplot"
    data["status"] = "verified"
    data["license"] = "MIT"
    data["maintainers"] = ["Devrajsinh Jhala"]
    data["extends"] = []
    data["governance"] = {
        "policy": "researchplot-governance-v1",
        "reviewers": ["Devrajsinh Jhala"],
        "reviewed_on": data["verified_on"],
        "change_note": "Migrated to schema v3 without changing venue rules.",
    }
    for source in data["sources"]:
        url = source["url"].casefold()
        source["kind"] = (
            "official_template"
            if any(token in url for token in ("template", "author-kit", "style-files"))
            else "official_guideline"
        )
        source["publisher"] = data["name"]
        source["archive_url"] = None
        source["content_sha256"] = None
    for rule in data["rules"]:
        rule["expression"] = None
        rule["supersedes"] = []
        rule["rationale"] = None
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "src" / "researchplot" / "profiles"
    for profile_path in sorted(root.glob("*.json")):
        migrate(profile_path)
