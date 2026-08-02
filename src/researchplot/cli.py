"""Standard-library command line interface for ResearchPlot."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from datetime import date
from typing import Any

import matplotlib
from matplotlib import font_manager

from .audit import audit_file
from .registry import list_venues, resolve_venue, search_venues


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchplot", description="Venue-aware research figure compliance tools."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"researchplot {__import__('researchplot').__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    venues = commands.add_parser("venues", help="Browse bundled venue profiles.")
    venue_commands = venues.add_subparsers(dest="venue_command", required=True)
    venue_list = venue_commands.add_parser("list", help="List verified venue profiles.")
    venue_list.add_argument("--kind", choices=["journal", "conference", "publisher"])
    venue_list.add_argument("--year", type=int)
    venue_list.add_argument("--json", action="store_true")
    venue_search = venue_commands.add_parser("search", help="Search profile names and aliases.")
    venue_search.add_argument("query")
    venue_search.add_argument("--json", action="store_true")
    venue_info = venue_commands.add_parser("info", help="Show profile rules and official sources.")
    venue_info.add_argument("venue")
    venue_info.add_argument("--json", action="store_true")

    doctor = commands.add_parser("doctor", help="Inspect local capabilities for a venue.")
    doctor.add_argument("--venue", required=True)
    doctor.add_argument("--json", action="store_true")

    audit = commands.add_parser("audit", help="Audit an exported figure file.")
    audit.add_argument("file")
    audit.add_argument("--venue", required=True)
    audit.add_argument("--width", required=True)
    audit.add_argument(
        "--artwork", required=True, choices=["vector", "halftone", "combination", "line_art"]
    )
    audit.add_argument("--json", action="store_true")
    return parser


def _print_profiles(profiles: list[Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps([profile.to_dict() for profile in profiles], indent=2))
        return
    if not profiles:
        print("No matching venue profiles.")
        return
    for profile in profiles:
        year = f" ({profile.year})" if profile.year else ""
        print(f"{profile.id:<20} {profile.name}{year} — widths: {', '.join(profile.width_options)}")


def _doctor(venue: str) -> dict[str, Any]:
    profile = resolve_venue(venue)
    installed = {font.name.casefold() for font in font_manager.fontManager.ttflist}
    family_rule = profile.get_rule("font.family")
    requested: list[str] = (
        [str(item) for item in family_rule.value]
        if family_rule is not None and isinstance(family_rule.value, tuple)
        else []
    )
    available = [
        name
        for name in requested
        if name.casefold() in installed or name in {"serif", "sans-serif", "monospace"}
    ]
    verified = date.fromisoformat(profile.verified_on)
    age_days = (date.today() - verified).days
    return {
        "profile_id": profile.id,
        "matplotlib": matplotlib.__version__,
        "backend": matplotlib.get_backend(),
        "widths": {name: profile.width_mm(name) for name in profile.width_options},
        "requested_fonts": requested,
        "available_font_fallbacks": available,
        "latex_available": shutil.which("latex") is not None,
        "verified_on": profile.verified_on,
        "freshness_warning": (
            "Publisher profile is older than twelve months; re-check the official sources."
            if profile.kind.value in {"journal", "publisher"} and age_days > 365
            else None
        ),
        "sources": [source.to_dict() for source in profile.sources],
        "caveats": list(profile.caveats),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ResearchPlot CLI and return a documented exit code."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "venues":
            if args.venue_command == "list":
                _print_profiles(list_venues(kind=args.kind, year=args.year), args.json)
            elif args.venue_command == "search":
                _print_profiles(search_venues(args.query), args.json)
            else:
                profile = resolve_venue(args.venue)
                if args.json:
                    print(json.dumps(profile.to_dict(), indent=2))
                else:
                    print(f"{profile.name} [{profile.id}]")
                    print(profile.scope)
                    print(f"Verified: {profile.verified_on}")
                    print(
                        "Widths: "
                        + ", ".join(
                            f"{name}={profile.width_mm(name):g} mm"
                            for name in profile.width_options
                        )
                    )
                    for rule in profile.rules:
                        print(
                            f"- {rule.level.value}: {rule.id} = {rule.value} {rule.unit or ''}".rstrip()
                        )
                    for source in profile.sources:
                        print(f"Source: {source.title} — {source.url}")
            return 0
        if args.command == "doctor":
            result = _doctor(args.venue)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"ResearchPlot doctor: {result['profile_id']}")
                print(f"Matplotlib {result['matplotlib']} ({result['backend']})")
                print(
                    "Fonts: "
                    + (
                        ", ".join(result["available_font_fallbacks"])
                        or "no requested family found; Matplotlib fallback will be used"
                    )
                )
                print(
                    f"LaTeX: {'available' if result['latex_available'] else 'not installed (optional)'}"
                )
                if result["freshness_warning"]:
                    print(f"Warning: {result['freshness_warning']}")
                for caveat in result["caveats"]:
                    print(f"Caveat: {caveat}")
            return 0
        report = audit_file(
            args.file,
            venue=args.venue,
            width=args.width,
            artwork=args.artwork,
        )
        print(json.dumps(report.to_dict(), indent=2) if args.json else report)
        return 0 if report.passed else 1
    except (FileNotFoundError, OSError, ValueError) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc), "exit_code": 2}), file=sys.stderr)
        else:
            print(f"researchplot: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
