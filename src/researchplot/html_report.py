"""Self-contained, offline HTML compliance reports."""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol, cast

from .safe_io import atomic_write_text


class SerializableReport(Protocol):
    """Structural contract accepted by :func:`render_html_report`."""

    def to_dict(self) -> dict[str, object]: ...


def _text(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _report_section(payload: Mapping[str, object], artifact: str | None) -> str:
    verdict = str(payload.get("verdict", "indeterminate"))
    summary = payload.get("summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    findings = payload.get("findings")
    finding_rows: list[str] = []
    if isinstance(findings, list):
        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            outcome = str(finding.get("outcome", finding.get("status", "skip")))
            finding_rows.append(
                "<tr>"
                f"<td><span class='status {html.escape(outcome)}'>{_text(outcome)}</span></td>"
                f"<td><code>{_text(finding.get('rule_id'))}</code></td>"
                f"<td>{_text(finding.get('phase'))}</td>"
                f"<td>{_text(finding.get('message'))}</td>"
                f"<td>{_text(finding.get('suggestion') or 'Review the cited source.')}</td>"
                "</tr>"
            )
    sources = payload.get("sources")
    source_rows: list[str] = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            url = str(source.get("url", ""))
            title = _text(source.get("title", url))
            locator = _text(source.get("locator"))
            if url.startswith(("https://", "http://")):
                title = f"<a href='{html.escape(url, quote=True)}'>{title}</a>"
            source_rows.append(f"<li>{title}{' — ' + locator if locator else ''}</li>")
    metrics = "".join(
        f"<div class='metric'><strong>{_text(summary_map.get(key, 0))}</strong>{label}</div>"
        for key, label in (
            ("findings", "findings"),
            ("failures", "failures"),
            ("warnings", "warnings"),
            ("unresolved", "unresolved"),
        )
    )
    artifact_heading = f"<p class='artifact'>{_text(artifact)}</p>" if artifact else ""
    coverage_note = (
        "Required evidence remains unresolved; this is not a compliance pass."
        if verdict == "indeterminate"
        else "Verdict calculated from the evidence phases recorded in this report."
    )
    return f"""
    <section class="report">
      <header><div>{artifact_heading}<h2>{_text(payload.get("profile"))}</h2></div>
      <span class="verdict {html.escape(verdict)}">{_text(verdict.replace("_", " "))}</span></header>
      <div class="metrics">{metrics}</div>
      <p class="coverage">{_text(coverage_note)}</p>
      <div class="table"><table><thead><tr><th>Status</th><th>Rule</th><th>Phase</th><th>Evidence</th><th>Next action</th></tr></thead>
      <tbody>{"".join(finding_rows)}</tbody></table></div>
      <details><summary>Official sources</summary><ul>{"".join(source_rows)}</ul></details>
    </section>"""


def render_html_report(
    reports: SerializableReport | Iterable[tuple[str | Path | None, SerializableReport]],
    *,
    title: str = "ResearchPlot compliance report",
) -> str:
    """Render one or more reports as a self-contained, offline HTML document."""

    items: list[tuple[str | Path | None, SerializableReport]]
    if hasattr(reports, "to_dict"):
        items = [(None, cast(SerializableReport, reports))]
    else:
        items = list(reports)
    if not items:
        raise ValueError("At least one report is required.")
    payloads = [
        (str(path) if path is not None else None, report.to_dict()) for path, report in items
    ]
    embedded = html.escape(json.dumps([payload for _, payload in payloads], ensure_ascii=False))
    sections = "".join(_report_section(payload, path) for path, payload in payloads)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="generator" content="ResearchPlot"><title>{_text(title)}</title>
<style>
:root{{--bg:#f4f7fb;--card:#fff;--text:#172033;--muted:#5e6b82;--line:#dbe3ef;--good:#08783f;--bad:#b42318;--unknown:#945d00}}
*{{box-sizing:border-box}} body{{margin:0;padding:2rem;color:var(--text);background:var(--bg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1180px;margin:auto}} h1{{font-size:clamp(2rem,5vw,3.6rem);letter-spacing:-.04em}} .intro{{color:var(--muted)}}
.report{{margin:1.5rem 0;padding:1.5rem;background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:0 10px 30px #1b2a4e10}}
.report header{{display:flex;justify-content:space-between;gap:1rem;align-items:start}} .report h2{{margin:.1rem 0}} .artifact{{margin:0;color:var(--muted)}}
.verdict{{padding:.45rem .7rem;border-radius:999px;font-weight:800;text-transform:uppercase}} .compliant{{color:var(--good);background:#ddf8e9}} .non_compliant{{color:var(--bad);background:#ffebe8}} .indeterminate{{color:var(--unknown);background:#fff2d2}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin:1rem 0}} .metric{{padding:.8rem;border:1px solid var(--line);border-radius:10px}} .metric strong{{display:block;font-size:1.5rem}}
.coverage{{padding:.8rem 1rem;border-left:4px solid var(--unknown);background:#fff8e7}} .table{{overflow:auto}} table{{width:100%;border-collapse:collapse}} th,td{{padding:.7rem;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}} th{{font-size:.75rem;text-transform:uppercase;color:var(--muted)}}
.status{{font-weight:800;text-transform:uppercase}} .status.pass{{color:var(--good)}} .status.fail{{color:var(--bad)}} .status.skip{{color:var(--unknown)}} a{{color:#3157d5}} code{{font-family:ui-monospace,monospace}}
@media(max-width:650px){{body{{padding:1rem}}.metrics{{grid-template-columns:1fr 1fr}}.report header{{display:block}}}}
</style></head><body><main><h1>{_text(title)}</h1><p class="intro">Self-contained evidence report. No scripts or external assets are required.</p>{sections}</main>
<script type="application/json" id="researchplot-report">{embedded}</script></body></html>"""


def write_html_report(
    reports: SerializableReport | Iterable[tuple[str | Path | None, SerializableReport]],
    path: str | Path,
    *,
    title: str = "ResearchPlot compliance report",
) -> Path:
    """Write a self-contained report using an atomic same-directory replacement."""

    return atomic_write_text(path, render_html_report(reports, title=title))


__all__ = ["render_html_report", "write_html_report"]
