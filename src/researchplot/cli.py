"""ResearchPlot compliance-as-code command line interface.

The CLI is intentionally a projection of the public Python APIs.  It never treats
missing evidence as success and never mutates an artifact in response to a suggested
fix.  Exit codes are stable: 0 compliant/success, 1 known failure, 2 invalid input or
missing capability, and 3 indeterminate compliance.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypedDict, cast

import matplotlib
from matplotlib import font_manager

from .artifact_security import (
    create_deterministic_archive,
    verify_deterministic_archive,
    verify_manifest,
)
from .compliance import CompliancePolicyError, Report, Verdict
from .governance import validate_profile_governance
from .html_report import write_html_report
from .inspectors import ArtifactInspectionError, inspect_artifact
from .manuscript import ManuscriptAudit, audit_manuscript_pdf
from .migration import project_from_v1, project_spec_from_v1
from .models import OutputFormat, VenueProfile
from .planning import PlanAssessment
from .profile_lock import ProfileLock, load_profile_lock, verify_profile_lock
from .project import ProjectConfig
from .project_api import Project
from .registry import list_profiles, load_profile, resolve_profile, search_profiles
from .remediation import plan_remediation
from .remote_registry import RegistryClient
from .safe_io import atomic_write_json, atomic_write_text, strict_json_dumps, validate_output_path
from .sarif import reports_to_sarif
from .specs import DeliverableSpec, FigureSpec, ManualAttestation, ProjectSpec, Waiver
from .standards import submission_manifest_to_jats, submission_manifest_to_ro_crate
from .target import coerce_format
from .target import target as make_target


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")


def _add_report_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=["text", "json", "sarif", "html"],
        default="text",
    )
    parser.add_argument("--json", action="store_true", help="Alias for --format json.")
    parser.add_argument("--sarif", action="store_true", help="Alias for --format sarif.")
    parser.add_argument(
        "--html",
        nargs="?",
        const="researchplot-report.html",
        metavar="FILE",
        help="Write a self-contained HTML report (default: researchplot-report.html).",
    )
    parser.add_argument("--output", help="Write the selected report to a file.")


def _add_target_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True)
    parser.add_argument("--width")
    parser.add_argument("--role", default="main")
    parser.add_argument(
        "--content",
        "--artwork",
        dest="content",
        default="data-visualization",
    )
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--lock", default="researchplot.lock.json")


def _profile_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    profiles = commands.add_parser(
        "profile",
        aliases=["profiles", "venues"],
        help="Browse, validate, govern, lock, and synchronize profiles.",
    )
    subcommands = profiles.add_subparsers(dest="profile_command", required=True)

    profile_list = subcommands.add_parser("list", help="List installed profile revisions.")
    profile_list.add_argument("--kind", choices=["journal", "conference", "publisher"])
    profile_list.add_argument("--year", type=int)
    _add_json_flag(profile_list)

    profile_search = subcommands.add_parser("search", help="Search profile IDs and aliases.")
    profile_search.add_argument("query")
    _add_json_flag(profile_search)

    profile_show = subcommands.add_parser(
        "show", aliases=["info"], help="Show rules and official sources."
    )
    profile_show.add_argument("profile")
    _add_json_flag(profile_show)

    profile_diff = subcommands.add_parser("diff", help="Compare two immutable revisions.")
    profile_diff.add_argument("left")
    profile_diff.add_argument("right")
    _add_json_flag(profile_diff)

    profile_validate = subcommands.add_parser(
        "validate", help="Validate an external schema-v2 or schema-v3 profile JSON file."
    )
    profile_validate.add_argument("file")
    _add_json_flag(profile_validate)

    profile_status = subcommands.add_parser(
        "status", help="Report profile governance and evidence status."
    )
    profile_status.add_argument("profile", nargs="?")
    _add_json_flag(profile_status)

    profile_lock = subcommands.add_parser("lock", help="Write a deterministic profile lock.")
    profile_lock.add_argument("profile")
    profile_lock.add_argument("--output", default="researchplot.lock.json")
    profile_lock.add_argument("--force", action="store_true")
    _add_json_flag(profile_lock)

    profile_verify = subcommands.add_parser("verify", help="Verify a profile lock and all digests.")
    profile_verify.add_argument("--lock", default="researchplot.lock.json")
    profile_verify.add_argument("--profile")
    _add_json_flag(profile_verify)

    profile_sync = subcommands.add_parser(
        "sync", help="Opt in to a TUF-verified remote profile refresh."
    )
    profile_sync.add_argument("--base-url", required=True)
    profile_sync.add_argument("--trusted-root", required=True)
    profile_sync.add_argument("--cache-dir", required=True)
    profile_sync.add_argument("--coordinate")
    _add_json_flag(profile_sync)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchplot",
        description="Source-backed preflight and submission compliance for research figures.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"researchplot {__import__('researchplot').__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _profile_parser(commands)

    init = commands.add_parser("init", help="Create a pinned schema-v3 project and profile lock.")
    init.add_argument("--profile", required=True)
    init.add_argument("--figure", action="append", required=True, metavar="FILE")
    init.add_argument("--width")
    init.add_argument("--role", default="main")
    init.add_argument("--content", default="data-visualization")
    init.add_argument("--output", default="researchplot.toml")
    init.add_argument("--lock", default="researchplot.lock.json")
    init.add_argument("--force", action="store_true")
    _add_json_flag(init)

    migrate = commands.add_parser("migrate", help="Convert an explicit v1 project to schema v3.")
    migrate.add_argument("--config", default="researchplot.toml")
    migrate.add_argument("--output", default="researchplot.v3.toml")
    migrate.add_argument("--lock", default="researchplot.lock.json")
    migrate.add_argument("--force", action="store_true")
    _add_json_flag(migrate)

    audit = commands.add_parser("audit", help="Audit one or more existing figure artifacts.")
    audit.add_argument("paths", nargs="+")
    _add_target_options(audit)
    _add_report_options(audit)

    check = commands.add_parser("check", help="Check a strict project or audit explicit files.")
    check.add_argument("paths", nargs="*")
    check.add_argument("--config")
    check.add_argument("--profile")
    check.add_argument("--width")
    check.add_argument("--role", default="main")
    check.add_argument("--content", "--artwork", dest="content", default="data-visualization")
    check.add_argument("--frozen", action="store_true")
    check.add_argument("--lock", default="researchplot.lock.json")
    _add_report_options(check)

    bundle = commands.add_parser("bundle", help="Build and verify submission bundles.")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_build = bundle_commands.add_parser("build", help="Build from researchplot.toml.")
    bundle_build.add_argument("--config", default="researchplot.toml")
    bundle_build.add_argument("--output", default="submission")
    bundle_build.add_argument("--policy", choices=["off", "violations", "complete"])
    bundle_build.add_argument("--frozen", action="store_true")
    _add_json_flag(bundle_build)

    bundle_verify = bundle_commands.add_parser("verify", help="Verify a bundle directory or ZIP.")
    bundle_verify.add_argument("path")
    bundle_verify.add_argument("--manifest", default="researchplot-manifest.json")
    bundle_verify.add_argument("--no-strict", action="store_true")
    _add_json_flag(bundle_verify)

    bundle_archive = bundle_commands.add_parser(
        "archive", help="Create a deterministic verified ZIP or TAR."
    )
    bundle_archive.add_argument("source")
    bundle_archive.add_argument("output")
    bundle_archive.add_argument("--manifest", default="researchplot-manifest.json")
    bundle_archive.add_argument("--no-strict", action="store_true")
    _add_json_flag(bundle_archive)

    bundle_jats = bundle_commands.add_parser(
        "jats", aliases=["JATS"], help="Export a JATS 1.4 figure-group fragment."
    )
    bundle_jats.add_argument("manifest")
    bundle_jats.add_argument("--output")
    bundle_jats.add_argument("--group-id", default="researchplot-figures")

    bundle_ro = bundle_commands.add_parser(
        "ro-crate",
        aliases=["rocrate", "RO-Crate"],
        help="Export RO-Crate 1.3 JSON-LD metadata.",
    )
    bundle_ro.add_argument("manifest")
    bundle_ro.add_argument("--output", default="ro-crate-metadata.json")
    bundle_ro.add_argument("--name", default="ResearchPlot submission evidence")
    bundle_ro.add_argument("--description", default="Reproducible ResearchPlot submission.")
    bundle_ro.add_argument("--license")
    bundle_ro.add_argument("--creator", action="append", default=[])
    bundle_ro.add_argument("--force", action="store_true")

    manuscript = commands.add_parser("manuscript", help="Audit a compiled manuscript PDF.")
    manuscript_commands = manuscript.add_subparsers(dest="manuscript_command", required=True)
    manuscript_check = manuscript_commands.add_parser(
        "check", help="Run passive manuscript structure checks."
    )
    manuscript_check.add_argument("path", nargs="?")
    manuscript_check.add_argument("--config")
    manuscript_check.add_argument("--max-pages", type=int, default=2000)
    manuscript_check.add_argument("--output")
    _add_json_flag(manuscript_check)

    fix = commands.add_parser("fix", help="Generate a non-mutating artifact remediation plan.")
    fix.add_argument("path")
    fix.add_argument("--plan", action="store_true", required=True)
    fix.add_argument("--format", choices=["markdown", "json"], default="markdown")
    fix.add_argument("--output")

    project = commands.add_parser("project", help="Plan safe project-level changes.")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    retarget = project_commands.add_parser(
        "retarget", help="Plan migration to another immutable profile revision."
    )
    retarget.add_argument("--config", default="researchplot.toml")
    retarget.add_argument("--profile", required=True)
    retarget.add_argument("--plan", action="store_true", required=True)
    retarget.add_argument("--apply", action="store_true")
    retarget.add_argument("--output")

    serve = commands.add_parser("serve", help="Run the loopback-only local workspace.")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--no-browser", action="store_true")

    explain = commands.add_parser("explain", help="Explain one rule and its provenance.")
    explain.add_argument("rule")
    explain.add_argument("--profile", required=True)
    _add_json_flag(explain)

    doctor = commands.add_parser("doctor", help="Inspect local capabilities for a profile.")
    doctor.add_argument("--profile", required=True)
    _add_json_flag(doctor)
    return parser


class RuleChange(TypedDict):
    rule_id: str
    left: dict[str, object]
    right: dict[str, object]


class ProfileDiff(TypedDict):
    left: str
    right: str
    added: list[str]
    removed: list[str]
    changed: list[RuleChange]


class DoctorResult(TypedDict):
    profile: str
    profile_digest: str
    matplotlib: str
    backend: str
    widths_mm: dict[str, float]
    requested_fonts: list[str]
    installed_fonts: list[str]
    latex_available: bool
    sources: list[dict[str, object]]
    caveats: list[str]


class _Serializable(Protocol):
    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class _PayloadReport:
    payload: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _emit_text(text: str, output: str | Path | None = None) -> None:
    if output is None:
        print(text)
    else:
        atomic_write_text(output, text.rstrip() + "\n")


def _emit_json(payload: object, output: str | Path | None = None) -> None:
    if output is None:
        print(strict_json_dumps(payload))
    else:
        atomic_write_json(output, payload)


def _profile_rows(profiles: list[VenueProfile], as_json: bool) -> None:
    if as_json:
        _emit_json([profile.to_dict() for profile in profiles])
        return
    if not profiles:
        print("No matching profiles.")
        return
    for profile in profiles:
        print(
            f"{profile.coordinate:<34} {profile.name} | widths: {', '.join(profile.width_options)}"
        )


def _profile_diff(left_name: str, right_name: str) -> ProfileDiff:
    left_path = Path(left_name)
    right_path = Path(right_name)
    left = load_profile(left_path) if left_path.is_file() else resolve_profile(left_name)
    right = load_profile(right_path) if right_path.is_file() else resolve_profile(right_name)
    left_rules = {rule.id: rule.to_dict() for rule in left.rules}
    right_rules = {rule.id: rule.to_dict() for rule in right.rules}
    added = sorted(right_rules.keys() - left_rules.keys())
    removed = sorted(left_rules.keys() - right_rules.keys())
    changed = sorted(
        key for key in left_rules.keys() & right_rules.keys() if left_rules[key] != right_rules[key]
    )
    return {
        "left": left.coordinate,
        "right": right.coordinate,
        "added": added,
        "removed": removed,
        "changed": [
            {"rule_id": key, "left": left_rules[key], "right": right_rules[key]} for key in changed
        ],
    }


def _doctor(profile_name: str) -> DoctorResult:
    profile = resolve_profile(profile_name)
    installed = {font.name.casefold(): font.name for font in font_manager.fontManager.ttflist}
    family_rule = profile.get_rule("font.family")
    requested = (
        [str(item) for item in family_rule.value]
        if family_rule and isinstance(family_rule.value, tuple)
        else []
    )
    available = [installed[name.casefold()] for name in requested if name.casefold() in installed]
    return {
        "profile": profile.coordinate,
        "profile_digest": profile.digest,
        "matplotlib": matplotlib.__version__,
        "backend": str(matplotlib.get_backend()),
        "widths_mm": {name: profile.width_mm(name) for name in profile.width_options},
        "requested_fonts": requested,
        "installed_fonts": available,
        "latex_available": shutil.which("latex") is not None,
        "sources": [source.to_dict() for source in profile.sources],
        "caveats": list(profile.caveats),
    }


def _expand_paths(values: Sequence[str]) -> list[Path]:
    supported = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".eps"}
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in sorted(path.rglob("*"))
                if candidate.is_file() and candidate.suffix.casefold() in supported
            )
        else:
            paths.append(path)
    return paths


def _direct_checks(args: argparse.Namespace) -> list[tuple[Path, Report]]:
    if not args.profile:
        raise ValueError("--profile is required when --config is not used.")
    selected = make_target(
        args.profile,
        role=args.role,
        width=args.width,
        content=args.content,
    )
    if args.frozen:
        verify_profile_lock(load_profile_lock(args.lock), selected.profile)
    paths = _expand_paths(args.paths)
    if not paths:
        raise ValueError("Provide at least one figure path or directory.")
    return [(path, selected.audit(path)) for path in paths]


def _report_selection(args: argparse.Namespace) -> tuple[str, str | None]:
    shortcuts = int(bool(args.json)) + int(bool(args.sarif)) + int(args.html is not None)
    if shortcuts > 1:
        raise ValueError("Choose only one of --json, --sarif, or --html.")
    if args.report_format != "text" and shortcuts:
        raise ValueError("Do not combine --format with a report shortcut.")
    selected = args.report_format
    output = args.output
    if args.json:
        selected = "json"
    elif args.sarif:
        selected = "sarif"
    elif args.html is not None:
        selected = "html"
        if output is not None and Path(output) != Path(args.html):
            raise ValueError("--html FILE and --output must not name different files.")
        output = args.html
    if selected == "html" and output is None:
        raise ValueError("HTML reports require --output FILE or --html [FILE].")
    return selected, output


def _emit_reports(
    reports: list[tuple[Path, Report]], report_format: str, output: str | None
) -> None:
    if report_format == "sarif":
        _emit_json(reports_to_sarif(reports), output)
    elif report_format == "json":
        _emit_json(
            [{"path": path.as_posix(), "report": report.to_dict()} for path, report in reports],
            output,
        )
    elif report_format == "html":
        assert output is not None
        write_html_report([(path, report) for path, report in reports], output)
    else:
        _emit_text("\n\n".join(f"{path}\n{report}" for path, report in reports), output)


def _assessment_payload(assessment: PlanAssessment, profile: VenueProfile) -> dict[str, object]:
    payload = assessment.to_dict()
    findings = list(assessment.findings)
    payload.update(
        {
            "findings": [item.to_dict() for item in findings],
            "summary": {
                "findings": len(findings),
                "failures": len(assessment.failures),
                "warnings": sum(
                    1
                    for item in findings
                    if item.outcome.value == "fail" and item.level.value == "recommended"
                ),
                "unresolved": len(assessment.unresolved) + len(assessment.capability_gaps),
            },
            "sources": [source.to_dict() for source in profile.sources],
            "caveats": list(profile.caveats),
        }
    )
    return payload


def _emit_assessment(
    assessment: PlanAssessment,
    profile: VenueProfile,
    report_format: str,
    output: str | None,
) -> None:
    payload = _assessment_payload(assessment, profile)
    if report_format == "json":
        _emit_json(payload, output)
    elif report_format == "sarif":
        reports = [
            (
                Path(
                    f"{item.figure_id}/"
                    f"{item.deliverable_id or (item.report.findings[0].phase if item.report.findings else 'evidence')}"
                ),
                item.report,
            )
            for item in assessment.evidence
        ]
        _emit_json(reports_to_sarif(reports), output)
    elif report_format == "html":
        assert output is not None
        write_html_report(_PayloadReport(payload), output)
    else:
        lines = [
            f"Profile: {assessment.profile}",
            f"Verdict: {assessment.verdict.value}",
            f"Required coverage: {len(assessment.coverage)} checks",
        ]
        if assessment.capability_gaps:
            lines.append("Capability gaps:")
            lines.extend(f"- {item}" for item in assessment.capability_gaps)
        for result in assessment.unresolved:
            lines.append(
                f"- unresolved: {result.requirement.figure_id} / "
                f"{result.requirement.rule_id} ({result.status.value})"
            )
        _emit_text("\n".join(lines), output)


def _exit_for_reports(reports: Sequence[tuple[Path, Report]]) -> int:
    verdicts = {report.verdict for _, report in reports}
    if Verdict.NON_COMPLIANT in verdicts:
        return 1
    if Verdict.INDETERMINATE in verdicts:
        return 3
    return 0


def _exit_for_verdict(verdict: Verdict) -> int:
    return {
        Verdict.COMPLIANT: 0,
        Verdict.NON_COMPLIANT: 1,
        Verdict.INDETERMINATE: 3,
    }[verdict]


def _project_settings(path: str | Path) -> Mapping[str, object]:
    config_path = Path(path)
    try:
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not parse {config_path}: {exc}") from exc
    tool = payload.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("researchplot"), dict):
        return cast(dict[str, object], tool["researchplot"])
    return payload


def _load_project(path: str | Path) -> Project:
    settings = _project_settings(path)
    if "schema_version" in settings:
        return Project.load(path)
    return project_from_v1(ProjectConfig.load(path), warn=False)


def _safe_id(value: str, index: int) -> str:
    selected = re.sub(r"[^a-z0-9_-]+", "-", value.casefold()).strip("-_")
    return selected or f"figure-{index}"


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _display_path(path: Path, root: Path) -> str:
    selected = path.resolve()
    try:
        selected = selected.relative_to(root.resolve())
    except ValueError:
        pass
    return selected.as_posix()


def _toml_array(values: Sequence[str]) -> str:
    return "[" + ", ".join(_toml_string(value) for value in values) + "]"


def _toml_inline(value: object) -> str:
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_toml_inline(item) for item in value) + "]"
    if isinstance(value, ManualAttestation):
        value = {
            "reviewer": value.reviewer,
            "date": value.date,
            "rationale": value.rationale,
            "evidence": value.evidence,
        }
    elif isinstance(value, Waiver):
        value = {
            "profile_digest": value.profile_digest,
            "reviewer": value.reviewer,
            "reason": value.reason,
            "expires_on": value.expires_on,
        }
    if isinstance(value, Mapping):
        return (
            "{ "
            + ", ".join(
                f"{_toml_string(str(key))} = {_toml_inline(item)}"
                for key, item in sorted(value.items())
            )
            + " }"
        )
    raise ValueError(f"Cannot serialize {type(value).__name__} as project TOML.")


def _toml_statements(values: Mapping[str, object]) -> str:
    return (
        "{ "
        + ", ".join(
            f"{_toml_string(key)} = {_toml_inline(value)}" for key, value in sorted(values.items())
        )
        + " }"
    )


def _project_toml(spec: ProjectSpec, *, root: Path) -> str:
    lines = [
        f"schema_version = {spec.schema_version}",
        f"profile = {_toml_string(spec.profile)}",
        f"policy = {_toml_string(str(spec.policy))}",
    ]
    if spec.lock_path is not None:
        lines.append(f"lock = {_toml_string(_display_path(spec.lock_path, root))}")
    if spec.metadata:
        raise ValueError("TOML migration cannot safely rewrite non-empty project metadata.")
    if spec.manuscript is not None:
        manuscript = spec.manuscript
        lines.extend(
            [
                "",
                "[manuscript]",
                f"path = {_toml_string(_display_path(manuscript.path, root))}",
                f"format = {_toml_string(str(manuscript.format))}",
                f"required = {'true' if manuscript.required else 'false'}",
            ]
        )
        if manuscript.metadata:
            raise ValueError("TOML migration cannot safely rewrite manuscript metadata.")
        for hint in manuscript.matching_hints:
            lines.extend(["", "[[manuscript.matching_hints]]"])
            lines.append(f"figure = {_toml_string(hint.figure_id)}")
            if hint.pages:
                lines.append("pages = [" + ", ".join(str(page) for page in hint.pages) + "]")
            if hint.number is not None:
                number = (
                    str(hint.number) if isinstance(hint.number, int) else _toml_string(hint.number)
                )
                lines.append(f"number = {number}")
            if hint.caption is not None:
                lines.append(f"caption = {_toml_string(hint.caption)}")

    for figure in spec.figures:
        if figure.metadata or any(panel.metadata for panel in figure.panels):
            raise ValueError("TOML migration cannot safely rewrite non-empty figure metadata.")
        lines.extend(["", "[[figures]]"])
        lines.append(f"id = {_toml_string(figure.id)}")
        lines.append(f"role = {_toml_string(str(figure.role))}")
        if figure.width is not None:
            lines.append(f"width = {_toml_string(figure.width)}")
        lines.append(f"content = {_toml_string(str(figure.content))}")
        if figure.number is not None:
            number = (
                str(figure.number)
                if isinstance(figure.number, int)
                else _toml_string(figure.number)
            )
            lines.append(f"number = {number}")
        for name, value in (
            ("caption", figure.caption),
            ("alt_text", figure.alt_text),
            ("long_description", figure.long_description),
        ):
            if value is not None:
                lines.append(f"{name} = {_toml_string(value)}")
        if figure.key_trends:
            lines.append(f"key_trends = {_toml_array(figure.key_trends)}")
        if figure.source_data:
            lines.append(
                "source_data = "
                + _toml_array([_display_path(path, root) for path in figure.source_data])
            )
        if figure.data_table is not None:
            lines.append(f"data_table = {_toml_string(_display_path(figure.data_table, root))}")
        if figure.attachments:
            lines.append(
                "attachments = "
                + _toml_array([_display_path(path, root) for path in figure.attachments])
            )
        if figure.attestations:
            lines.append(f"attestations = {_toml_statements(figure.attestations)}")
        if figure.waivers:
            lines.append(f"waivers = {_toml_statements(figure.waivers)}")
        for panel in figure.panels:
            lines.extend(["", "[[figures.panels]]", f"id = {_toml_string(panel.id)}"])
            if panel.order is not None:
                lines.append(f"order = {panel.order}")
            for name, value in (
                ("label", panel.label),
                ("description", panel.description),
                ("alt_text", panel.alt_text),
                ("long_description", panel.long_description),
            ):
                if value is not None:
                    lines.append(f"{name} = {_toml_string(value)}")
            if panel.source_data:
                lines.append(
                    "source_data = "
                    + _toml_array([_display_path(path, root) for path in panel.source_data])
                )
        for deliverable in figure.deliverables:
            output_format = cast(OutputFormat, deliverable.format)
            lines.extend(
                [
                    "",
                    "[[figures.deliverables]]",
                    f"id = {_toml_string(deliverable.id)}",
                    f"format = {_toml_string(output_format.value)}",
                ]
            )
            if deliverable.path is not None:
                lines.append(f"path = {_toml_string(_display_path(deliverable.path, root))}")
            lines.append(f"required = {'true' if deliverable.required else 'false'}")
            lines.append(f"preferred = {'true' if deliverable.preferred else 'false'}")
            if deliverable.metadata:
                raise ValueError("TOML migration cannot safely rewrite deliverable metadata.")
    return "\n".join(lines) + "\n"


def _write_lock(profile: VenueProfile, path: str | Path, *, force: bool) -> Path:
    destination = validate_output_path(path, allow_existing=force)
    return atomic_write_json(
        destination,
        ProfileLock.from_profile(profile).to_dict(),
        allow_existing=force,
    )


def _run_init(args: argparse.Namespace) -> int:
    profile = resolve_profile(args.profile)
    output = Path(args.output).expanduser().resolve()
    lock_path = Path(args.lock).expanduser().resolve()
    if output == lock_path:
        raise ValueError("Project configuration and profile lock must use different paths.")
    width = args.width if args.width is not None else profile.default_width
    figures: list[FigureSpec] = []
    used_ids: set[str] = set()
    for index, raw_path in enumerate(args.figure, start=1):
        path = Path(raw_path).expanduser().resolve()
        if not path.suffix:
            raise ValueError(f"Figure path needs a supported extension: {path}")
        output_format = coerce_format(path.suffix)
        base = _safe_id(path.stem, index)
        figure_id = base
        duplicate = 2
        while figure_id in used_ids:
            figure_id = f"{base}-{duplicate}"
            duplicate += 1
        used_ids.add(figure_id)
        figures.append(
            FigureSpec(
                figure_id,
                (
                    DeliverableSpec(
                        "main",
                        output_format,
                        path=path,
                        required=True,
                        preferred=True,
                    ),
                ),
                role=args.role,
                width=width,
                content=args.content,
            )
        )
    spec = ProjectSpec(
        profile=profile.coordinate,
        figures=tuple(figures),
        lock_path=lock_path,
        config_path=output,
    )
    # Resolve every target before creating either output.
    Project(spec)
    validate_output_path(output, allow_existing=args.force)
    validate_output_path(lock_path, allow_existing=args.force)
    atomic_write_text(
        output,
        _project_toml(spec, root=output.parent),
        allow_existing=args.force,
    )
    _write_lock(profile, lock_path, force=args.force)
    payload = {
        "project": output.as_posix(),
        "lock": lock_path.as_posix(),
        "profile": profile.coordinate,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"Created {output}")
        print(f"Locked {profile.coordinate} in {lock_path}")
    return 0


def _run_migrate(args: argparse.Namespace) -> int:
    legacy = ProjectConfig.load(args.config)
    output = Path(args.output).expanduser().resolve()
    lock_path = Path(args.lock).expanduser().resolve()
    if output == lock_path:
        raise ValueError("Migrated configuration and profile lock must use different paths.")
    spec = replace(
        project_spec_from_v1(legacy, warn=False),
        lock_path=lock_path,
        config_path=output,
    )
    Project(spec)
    validate_output_path(output, allow_existing=args.force)
    validate_output_path(lock_path, allow_existing=args.force)
    atomic_write_text(
        output,
        _project_toml(spec, root=output.parent),
        allow_existing=args.force,
    )
    _write_lock(legacy.profile, lock_path, force=args.force)
    payload = {
        "source": Path(args.config).resolve().as_posix(),
        "project": output.as_posix(),
        "lock": lock_path.as_posix(),
        "profile": legacy.profile.coordinate,
    }
    if args.json:
        _emit_json(payload)
    else:
        print(f"Migrated {args.config} -> {output}")
        print(f"Profile lock: {lock_path}")
    return 0


def _run_profile(args: argparse.Namespace) -> int:
    command = args.profile_command
    if command == "list":
        _profile_rows(list_profiles(kind=args.kind, year=args.year), args.json)
        return 0
    if command == "search":
        _profile_rows(search_profiles(args.query), args.json)
        return 0
    if command in {"show", "info"}:
        profile = resolve_profile(args.profile)
        if args.json:
            _emit_json(profile.to_dict())
        else:
            print(f"{profile.name} [{profile.coordinate}]")
            print(profile.scope)
            for rule in profile.rules:
                print(
                    f"- {rule.level.value}: {rule.id} "
                    f"({rule.probe} {rule.constraint.operator.value} {rule.value!r})"
                )
            for source in profile.sources:
                print(f"Source: {source.title} | {source.locator} | {source.url}")
        return 0
    if command == "diff":
        result = _profile_diff(args.left, args.right)
        if args.json:
            _emit_json(result)
        else:
            print(f"{result['left']} -> {result['right']}")
            print(f"Added: {', '.join(result['added']) or 'none'}")
            print(f"Removed: {', '.join(result['removed']) or 'none'}")
            print(
                "Changed: " + (", ".join(item["rule_id"] for item in result["changed"]) or "none")
            )
        return 0
    if command == "validate":
        profile = load_profile(args.file)
        payload = {"valid": True, "profile": profile.coordinate, "digest": profile.digest}
        if args.json:
            _emit_json(payload)
        else:
            print(f"Valid: {profile.coordinate}")
        return 0
    if command == "status":
        profiles = [resolve_profile(args.profile)] if args.profile else list_profiles()
        rows: list[dict[str, object]] = []
        has_errors = False
        for profile in profiles:
            report = validate_profile_governance(profile)
            has_errors = has_errors or not report.passed
            rows.append(
                {
                    "profile": profile.coordinate,
                    "status": profile.status.value,
                    "passed": report.passed,
                    "issues": [
                        {"code": issue.code, "message": issue.message, "error": issue.error}
                        for issue in report.issues
                    ],
                }
            )
        if args.json:
            _emit_json(rows)
        else:
            for row in rows:
                print(
                    f"{row['profile']}: {'ready' if row['passed'] else 'invalid'} ({row['status']})"
                )
                for issue in cast(list[dict[str, object]], row["issues"]):
                    print(f"- {issue['code']}: {issue['message']}")
        return 2 if has_errors else 0
    if command == "lock":
        profile = resolve_profile(args.profile)
        output = _write_lock(profile, args.output, force=args.force)
        if args.json:
            _emit_json(
                {"path": output.as_posix(), "profile": profile.coordinate, "digest": profile.digest}
            )
        else:
            print(output)
        return 0
    if command == "verify":
        lock = load_profile_lock(args.lock)
        profile = resolve_profile(args.profile or lock.coordinate)
        verify_profile_lock(lock, profile)
        payload = {
            "valid": True,
            "lock": str(Path(args.lock)),
            "profile": profile.coordinate,
            "digest": profile.digest,
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Verified {profile.coordinate} against {args.lock}")
        return 0
    if command == "sync":
        client = RegistryClient(
            args.base_url,
            trusted_root=args.trusted_root,
            cache_dir=args.cache_dir,
        )
        diagnostic = client.diagnostic()
        if not diagnostic.available:
            raise RuntimeError(diagnostic.message)
        client.refresh()
        profile_payload: dict[str, object] | None = None
        if args.coordinate:
            profile_payload = client.fetch_profile(args.coordinate).to_dict()
        payload = {
            "refreshed": True,
            "registry": args.base_url,
            "diagnostic": {"code": diagnostic.code, "message": diagnostic.message},
            "profile": profile_payload,
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Verified registry metadata refreshed from {args.base_url}")
            if args.coordinate:
                print(f"Verified target: {args.coordinate}")
        return 0
    raise ValueError(f"Unsupported profile command: {command}")


def _run_check(args: argparse.Namespace) -> int:
    report_format, output = _report_selection(args)
    if args.config:
        if args.paths or args.profile:
            raise ValueError("--config cannot be combined with paths or --profile.")
        project = _load_project(args.config)
        assessment = project.plan(frozen=args.frozen).check()
        _emit_assessment(assessment, project.profile, report_format, output)
        return _exit_for_verdict(assessment.verdict)
    reports = _direct_checks(args)
    _emit_reports(reports, report_format, output)
    return _exit_for_reports(reports)


def _run_bundle(args: argparse.Namespace) -> int:
    command = args.bundle_command
    if command == "build":
        if Path(args.output).suffix.casefold() in {".zip", ".tar"}:
            raise ValueError(
                "bundle build creates a directory; use bundle archive for ZIP or TAR output."
            )
        project = _load_project(args.config)
        if args.policy is not None:
            project = Project(replace(project.spec, policy=args.policy))
        if args.frozen:
            project.plan(frozen=True).verify_lock()
        result = project.bundle(args.output)
        payload = {
            "path": result.path.as_posix(),
            "manifest": result.manifest_path.as_posix(),
            "passed": result.passed,
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Built {result.path}")
            print(f"Manifest: {result.manifest_path}")
        return _exit_for_reports([(Path(item.name), item.report) for item in result.items])
    if command == "verify":
        path = Path(args.path)
        verification = (
            verify_deterministic_archive(
                path,
                manifest_name=args.manifest,
                require_deterministic_metadata=not args.no_strict,
            )
            if path.suffix.casefold() in {".zip", ".tar"}
            else verify_manifest(path, manifest_name=args.manifest, strict=not args.no_strict)
        )
        if args.json:
            _emit_json(verification.to_dict())
        else:
            print("Bundle integrity: " + ("valid" if verification.valid else "invalid"))
            for issue in verification.issues:
                print(f"- {issue.code}: {issue.path or '-'}: {issue.message}")
        return 0 if verification.valid else 1
    if command == "archive":
        archive_result = create_deterministic_archive(
            args.source,
            args.output,
            manifest_name=args.manifest,
            strict=not args.no_strict,
        )
        payload = {
            "path": archive_result.path.as_posix(),
            "sha256": archive_result.sha256,
            "bytes": archive_result.bytes,
            "verification": archive_result.verification.to_dict(),
        }
        if args.json:
            _emit_json(payload)
        else:
            print(f"Archived {archive_result.path} ({archive_result.sha256})")
        return 0
    manifest = Path(args.manifest)
    if manifest.is_dir():
        manifest = manifest / "researchplot-manifest.json"
    if command in {"jats", "JATS"}:
        fragment = submission_manifest_to_jats(manifest, group_id=args.group_id)
        _emit_text(fragment, args.output)
        return 0
    if command in {"ro-crate", "rocrate", "RO-Crate"}:
        crate = submission_manifest_to_ro_crate(
            manifest,
            name=args.name,
            description=args.description,
            license=args.license,
            creators=tuple(args.creator),
        )
        atomic_write_json(args.output, crate, allow_existing=args.force)
        print(args.output)
        return 0
    raise ValueError(f"Unsupported bundle command: {command}")


def _manuscript_payload(audit: ManuscriptAudit) -> dict[str, object]:
    payload = audit.to_dict()
    placements = audit.placement_audit
    if placements is None:
        gap = (
            "Structural PDF facts were audited, but no configured figures were supplied "
            "for placement matching."
        )
    elif placements.coverage_complete:
        gap = (
            "All configured placements were measured, but this command did not evaluate "
            "venue-specific manuscript rules; it does not claim full compliance."
        )
    else:
        gap = (
            f"{len(placements.unresolved)} configured figure placement(s) remain missing, "
            "ambiguous, or unmeasured."
        )
    payload.update(
        {
            "verdict": Verdict.INDETERMINATE.value,
            "capability_gap": gap,
        }
    )
    return payload


def _run_manuscript(args: argparse.Namespace) -> int:
    if args.path and args.config:
        raise ValueError("Choose either a manuscript path or --config, not both.")
    if args.config:
        project = _load_project(args.config)
        if project.spec.manuscript is None:
            raise ValueError("The project does not configure a manuscript.")
        if str(project.spec.manuscript.format) != "pdf":
            raise ValueError("Compiled manuscript auditing currently supports PDF only.")
        audit = project.audit_manuscript(max_pages=args.max_pages)
    elif args.path:
        audit = audit_manuscript_pdf(args.path, max_pages=args.max_pages)
    else:
        raise ValueError("Provide a manuscript PDF path or --config.")
    payload = _manuscript_payload(audit)
    if args.json or args.output:
        _emit_json(payload, args.output)
    else:
        print(f"Manuscript: {audit.path}")
        print(f"Pages: {audit.page_count}; fonts: {audit.font_count}")
        print("Verdict: indeterminate")
        if audit.placement_audit is not None:
            placements = audit.placement_audit
            print(
                f"Placements: {len(placements.matches) - len(placements.unresolved)} matched; "
                f"{len(placements.unresolved)} unresolved"
            )
            for match in placements.unresolved:
                print(f"- {match.figure_id}: {match.status.value}: {match.detail}")
        for warning in audit.warnings:
            print(f"- {warning}")
    return 3


def _run_fix(args: argparse.Namespace) -> int:
    if not args.plan:  # argparse requires it; retained as a defensive API guard.
        raise ValueError("Automatic artifact mutation is disabled; use --plan.")
    plan = plan_remediation(inspect_artifact(args.path))
    if args.format == "json":
        _emit_json(plan.to_dict(), args.output)
    else:
        _emit_text(plan.to_markdown(), args.output)
    if plan.empty:
        return 0
    if any(item.severity in {"critical", "high"} for item in plan.remediations):
        return 1
    return 3


def _retarget_plan(project: Project, profile_name: str) -> dict[str, object]:
    destination = resolve_profile(profile_name)
    figures: list[dict[str, object]] = []
    changes_required = project.profile.coordinate != destination.coordinate
    for figure in project.spec.figures:
        width = figure.width
        width_supported = width is None or width in destination.width_options
        proposed_width = width if width_supported else destination.default_width
        target = make_target(
            destination,
            role=figure.role,
            width=proposed_width,
            content=figure.content,
        )
        allowed = target.plan_export().allowed_formats
        declared = tuple(cast(OutputFormat, item.format) for item in figure.deliverables)
        disallowed = tuple(item for item in declared if allowed and item not in allowed)
        manual = not width_supported or bool(disallowed)
        changes_required = changes_required or manual
        figures.append(
            {
                "id": figure.id,
                "current_width": width,
                "proposed_width": proposed_width,
                "declared_formats": [item.value for item in declared],
                "allowed_formats": [item.value for item in allowed],
                "disallowed_formats": [item.value for item in disallowed],
                "manual_review_required": manual,
            }
        )
    return {
        "from_profile": project.profile.coordinate,
        "to_profile": destination.coordinate,
        "changes_required": changes_required,
        "safe_automatic_apply": False,
        "reason": (
            "ResearchPlot will not rewrite TOML comments or choose replacement formats without "
            "author review. Apply the proposed coordinate and per-figure changes manually."
        ),
        "figures": figures,
    }


def _run_project(args: argparse.Namespace) -> int:
    if args.project_command != "retarget":
        raise ValueError(f"Unsupported project command: {args.project_command}")
    if args.apply:
        raise RuntimeError(
            "Safe comment-preserving TOML retargeting is unavailable; rerun with --plan only."
        )
    project = Project.load(args.config)
    payload = _retarget_plan(project, args.profile)
    _emit_json(payload, args.output)
    return 3 if payload["changes_required"] else 0


def _run_explain(args: argparse.Namespace) -> int:
    profile = resolve_profile(args.profile)
    rule = profile.get_rule(args.rule)
    if rule is None:
        raise ValueError(f"Profile {profile.coordinate} has no rule {args.rule!r}.")
    sources = [source.to_dict() for source in profile.sources if source.id in rule.source_ids]
    payload = {"profile": profile.coordinate, "rule": rule.to_dict(), "sources": sources}
    if args.json:
        _emit_json(payload)
    else:
        print(f"{rule.id}: {rule.description}")
        print(
            f"{rule.level.value}; {rule.verification.value}; "
            f"{rule.probe} {rule.constraint.operator.value} {rule.value!r}"
        )
        for source in sources:
            print(f"Source: {source['title']} | {source['locator']} | {source['url']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable compliance-aware exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command in {"profile", "profiles", "venues"}:
            return _run_profile(args)
        if args.command == "init":
            return _run_init(args)
        if args.command == "migrate":
            return _run_migrate(args)
        if args.command == "audit":
            report_format, output = _report_selection(args)
            reports = _direct_checks(args)
            _emit_reports(reports, report_format, output)
            return _exit_for_reports(reports)
        if args.command == "check":
            return _run_check(args)
        if args.command == "bundle":
            return _run_bundle(args)
        if args.command == "manuscript":
            return _run_manuscript(args)
        if args.command == "fix":
            return _run_fix(args)
        if args.command == "project":
            return _run_project(args)
        if args.command == "serve":
            from .webapp import serve as serve_webapp

            serve_webapp(port=args.port, open_browser=not args.no_browser)
            return 0
        if args.command == "explain":
            return _run_explain(args)
        if args.command == "doctor":
            result = _doctor(args.profile)
            if args.json:
                _emit_json(result)
            else:
                print(f"ResearchPlot doctor: {result['profile']}")
                print(f"Matplotlib {result['matplotlib']} ({result['backend']})")
                print("Fonts: " + (", ".join(result["installed_fonts"]) or "fallbacks required"))
                print(
                    "LaTeX: "
                    + ("available" if result["latex_available"] else "not installed (optional)")
                )
            return 0
        raise ValueError(f"Unsupported command: {args.command}")
    except KeyboardInterrupt:
        print("researchplot: interrupted", file=sys.stderr)
        return 2
    except CompliancePolicyError as exc:
        print(f"researchplot: {exc}", file=sys.stderr)
        return 1 if exc.report.verdict is Verdict.NON_COMPLIANT else 3
    except (
        ArtifactInspectionError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"researchplot: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
