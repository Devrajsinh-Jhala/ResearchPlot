from __future__ import annotations

from PIL import Image

from researchplot import target
from researchplot.html_report import render_html_report, write_html_report


def test_html_report_is_self_contained_and_escaped(tmp_path) -> None:
    artifact = tmp_path / "figure.png"
    Image.new("RGB", (1051, 600), "white").save(artifact, dpi=(300, 300))
    report = target("nature@2026.08.0", width="single").audit(artifact)
    rendered = render_html_report([("<figure>.png", report)], title="Evidence <report>")
    assert "<!doctype html>" in rendered
    assert "Evidence &lt;report&gt;" in rendered
    assert "&lt;figure&gt;.png" in rendered
    assert "application/json" in rendered
    assert "<script src=" not in rendered

    destination = write_html_report(report, tmp_path / "report.html")
    assert destination.is_file()
    assert "ResearchPlot compliance report" in destination.read_text(encoding="utf-8")
