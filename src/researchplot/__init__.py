"""Source-backed venue compliance for research figures and submissions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .compliance import (
    CompliancePolicyError,
    Finding,
    Outcome,
    Policy,
    Report,
    RuleEngine,
    TargetContext,
    Verdict,
)
from .contracts import export_manifest_schema, report_schema, submission_manifest_schema
from .inspectors import (
    ArtifactInspection,
    ArtifactInspectionError,
    ArtifactParseError,
    UnsupportedArtifactError,
    inspect_artifact,
)
from .manifest import ArtifactRecord, ExportManifest
from .models import (
    ConstraintOperator,
    ContentKind,
    FigureRole,
    OutputFormat,
    RuleApplicability,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    SourceRef,
    VenueKind,
    VenueProfile,
    VenueResolutionWarning,
    VenueRule,
    VerificationMode,
)
from .project import FigureConfig, ProjectConfig, write_profile_lock
from .registry import (
    list_profiles,
    load_profile,
    profile_schema,
    resolve_profile,
    search_profiles,
    validate_profile_data,
)
from .sarif import reports_to_sarif
from .style import StyleContext
from .submission import BundleResult, Submission, SubmissionItemResult
from .target import Target, target
from .transactional_export import ExportResult

try:
    __version__ = version("researchplot-venues")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

# The concise public spelling does not discard the explicit model name.
Profile = VenueProfile

__all__ = [
    "ArtifactInspection",
    "ArtifactInspectionError",
    "ArtifactParseError",
    "ArtifactRecord",
    "BundleResult",
    "CompliancePolicyError",
    "ConstraintOperator",
    "ContentKind",
    "ExportManifest",
    "ExportResult",
    "FigureConfig",
    "FigureRole",
    "Finding",
    "Outcome",
    "OutputFormat",
    "Policy",
    "Profile",
    "ProjectConfig",
    "Report",
    "RuleApplicability",
    "RuleConstraint",
    "RuleEngine",
    "RuleLevel",
    "RulePhase",
    "SourceRef",
    "StyleContext",
    "Submission",
    "SubmissionItemResult",
    "Target",
    "TargetContext",
    "UnsupportedArtifactError",
    "VenueKind",
    "VenueProfile",
    "VenueResolutionWarning",
    "VenueRule",
    "Verdict",
    "VerificationMode",
    "__version__",
    "export_manifest_schema",
    "inspect_artifact",
    "list_profiles",
    "load_profile",
    "profile_schema",
    "reports_to_sarif",
    "report_schema",
    "resolve_profile",
    "search_profiles",
    "submission_manifest_schema",
    "target",
    "validate_profile_data",
    "write_profile_lock",
]
