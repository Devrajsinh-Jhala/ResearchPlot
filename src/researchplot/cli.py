"""ResearchPlot compliance-as-code command line interface."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import matplotlib
from matplotlib import font_manager

from .compliance import CompliancePolicyError, Report, Verdict
from .inspectors import ArtifactInspectionError
from .models import VenueProfile
from .project import ProjectConfig, write_profile_lock
from .registry import list_profiles, load_profile, resolve_profile, search_profiles
from .sarif import reports_to_sarif
from .submission import BundleResult, Submission
from .target import target as make_target


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

    profiles = commands.add_parser("profile", help="Browse, validate, diff, and lock profiles.")
    profile_commands = profiles.add_subparsers(dest="profile_command", required=True)
    profile_list = profile_commands.add_parser("list", help="List installed profile revisions.")
    profile_list.add_argument("--kind", choices=["journal", "conference", "publisher"])
    profile_list.add_argument("--year", type=int)
    profile_list.add_argument("--json", action="store_true")
    profile_search = profile_commands.add_parser("search", help="Search profile IDs and aliases.")
    profile_search.add_argument("query")
    profile_search.add_argument("--json", action="store_true")
    profile_show = profile_commands.add_parser("show", help="Show rules and official sources.")
    profile_show.add_argument("profile")
    profile_show.add_argument("--json", action="store_true")
    profile_diff = profile_commands.add_parser("diff", help="Compare two immutable revisions.")
    profile_diff.add_argument("left")
    profile_diff.add_argument("right")
    profile_diff.add_argument("--json", action="store_true")
    profile_validate = profile_commands.add_parser(
        "validate", help="Validate an external schema-v2 profile JSON file."
    )
    profile_validate.add_argument("file")
    profile_validate.add_argument("--json", action="store_true")
    profile_lock = profile_commands.add_parser("lock", help="Write a deterministic profile lock.")
    profile_lock.add_argument("profile")
    profile_lock.add_argument("--output", default="researchplot.lock.json")

    check = commands.add_parser("check", help="Audit figure files or a configured project.")
    check.add_argument("paths", nargs="*")
    check.add_argument("--config")
    check.add_argument("--profile")
    check.add_argument("--width")
    check.add_argument("--role", default="main")
    check.add_argument("--content", default="data-visualization")
    check.add_argument(
        "--format", dest="report_format", choices=["text", "json", "sarif"], default="text"
    )
    check.add_argument("--output", help="Write JSON or SARIF to this file instead of stdout.")

    bundle = commands.add_parser("bundle", help="Build audited submission directories.")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    bundle_build = bundle_commands.add_parser("build", help="Build from researchplot.toml.")
    bundle_build.add_argument("--config", default="researchplot.toml")
    bundle_build.add_argument("--output", default="submission")
    bundle_build.add_argument("--json", action="store_true")

    explain = commands.add_parser("explain", help="Explain one rule and its provenance.")
    explain.add_argument("rule")
    explain.add_argument("--profile", required=True)
    explain.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor", help="Inspect local capabilities for a profile.")
    doctor.add_argument("--profile", required=True)
    doctor.add_argument("--json", action="store_true")
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
    sources: list[dict[str, str]]
    caveats: list[str]


def _profile_rows(profiles: list[VenueProfile], as_json: bool) -> None:
    if as_json:
        print(json.dumps([profile.to_dict() for profile in profiles], indent=2))
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


def _configured_checks(config: ProjectConfig) -> list[tuple[Path, Report]]:
    reports: list[tuple[Path, Report]] = []
    for figure in config.figures:
        target = make_target(
            config.profile,
            role=figure.role,
            width=figure.width,
            content=figure.content,
        )
        reports.append((figure.path, target.audit(figure.path)))
    return reports


def _direct_checks(args: argparse.Namespace) -> list[tuple[Path, Report]]:
    if not args.profile:
        raise ValueError("--profile is required when --config is not used.")
    paths = _expand_paths(args.paths)
    if not paths:
        raise ValueError("Provide at least one figure path or directory.")
    target = make_target(
        args.profile,
        role=args.role,
        width=args.width,
        content=args.content,
    )
    return [(path, target.audit(path)) for path in paths]


def _emit_checks(
    reports: list[tuple[Path, Report]], report_format: str, output: str | None
) -> None:
    if report_format == "sarif":
        payload: object = reports_to_sarif(reports)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    elif report_format == "json":
        payload = [
            {"path": path.as_posix(), "report": report.to_dict()} for path, report in reports
        ]
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = "\n\n".join(f"{path}\n{report}" for path, report in reports)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _exit_for_reports(reports: Sequence[tuple[Path, Report]]) -> int:
    verdicts = {report.verdict for _, report in reports}
    if Verdict.NON_COMPLIANT in verdicts:
        return 1
    if Verdict.INDETERMINATE in verdicts:
        return 3
    return 0


def _build_bundle(config_path: str, output: str) -> BundleResult:
    config = ProjectConfig.load(config_path)
    submission = Submission(config.profile, output_dir=output, policy=config.policy)
    for figure in config.figures:
        submission.add(
            figure.path.name,
            figure.path,
            role=figure.role,
            width=figure.width,
            content=figure.content,
            alt_text=figure.alt_text,
            caption=figure.caption,
            source_data=figure.source_data,
        )
    return submission.build()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a stable compliance-aware exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "profile":
            if args.profile_command == "list":
                _profile_rows(list_profiles(kind=args.kind, year=args.year), args.json)
            elif args.profile_command == "search":
                _profile_rows(search_profiles(args.query), args.json)
            elif args.profile_command == "show":
                profile = resolve_profile(args.profile)
                if args.json:
                    print(json.dumps(profile.to_dict(), indent=2))
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
            elif args.profile_command == "diff":
                diff_result = _profile_diff(args.left, args.right)
                if args.json:
                    print(json.dumps(diff_result, indent=2))
                else:
                    print(f"{diff_result['left']} -> {diff_result['right']}")
                    print(f"Added: {', '.join(diff_result['added']) or 'none'}")
                    print(f"Removed: {', '.join(diff_result['removed']) or 'none'}")
                    print(
                        "Changed: "
                        + (", ".join(item["rule_id"] for item in diff_result["changed"]) or "none")
                    )
            elif args.profile_command == "validate":
                profile = load_profile(args.file)
                payload = {"valid": True, "profile": profile.coordinate, "digest": profile.digest}
                print(
                    json.dumps(payload, indent=2) if args.json else f"Valid: {profile.coordinate}"
                )
            else:
                output = write_profile_lock(resolve_profile(args.profile), args.output)
                print(output)
            return 0

        if args.command == "check":
            config_path = args.config
            if config_path is None and not args.profile and Path("researchplot.toml").is_file():
                config_path = "researchplot.toml"
            reports = (
                _configured_checks(ProjectConfig.load(config_path))
                if config_path
                else _direct_checks(args)
            )
            _emit_checks(reports, args.report_format, args.output)
            return _exit_for_reports(reports)

        if args.command == "bundle":
            bundle_result = _build_bundle(args.config, args.output)
            if args.json:
                print(
                    json.dumps(
                        {
                            "path": bundle_result.path.as_posix(),
                            "manifest": bundle_result.manifest_path.as_posix(),
                            "passed": bundle_result.passed,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"Built {bundle_result.path}")
                print(f"Manifest: {bundle_result.manifest_path}")
            return _exit_for_reports(
                [(Path(item.name), item.report) for item in bundle_result.items]
            )

        if args.command == "explain":
            profile = resolve_profile(args.profile)
            selected_rule = profile.get_rule(args.rule)
            if selected_rule is None:
                raise ValueError(f"Profile {profile.coordinate} has no rule {args.rule!r}.")
            selected_sources = [
                source.to_dict()
                for source in profile.sources
                if source.id in selected_rule.source_ids
            ]
            payload = {
                "profile": profile.coordinate,
                "rule": selected_rule.to_dict(),
                "sources": selected_sources,
            }
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                print(f"{selected_rule.id}: {selected_rule.description}")
                print(
                    f"{selected_rule.level.value}; {selected_rule.verification.value}; "
                    f"{selected_rule.probe} {selected_rule.constraint.operator.value} "
                    f"{selected_rule.value!r}"
                )
                for source_payload in selected_sources:
                    print(
                        f"Source: {source_payload['title']} | {source_payload['locator']} | "
                        f"{source_payload['url']}"
                    )
            return 0

        doctor_result = _doctor(args.profile)
        if args.json:
            print(json.dumps(doctor_result, indent=2))
        else:
            print(f"ResearchPlot doctor: {doctor_result['profile']}")
            print(f"Matplotlib {doctor_result['matplotlib']} ({doctor_result['backend']})")
            print(
                "Fonts: "
                + (", ".join(doctor_result["installed_fonts"]) or "requested fonts unavailable")
            )
            print(
                "LaTeX: "
                + ("available" if doctor_result["latex_available"] else "not installed (optional)")
            )
        return 0
    except CompliancePolicyError as exc:
        print(f"researchplot: {exc}", file=sys.stderr)
        return 1 if exc.report.verdict is Verdict.NON_COMPLIANT else 3
    except (
        ArtifactInspectionError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"researchplot: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
