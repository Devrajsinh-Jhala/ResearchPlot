"""Source-backed venue compliance for research figures and submissions."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .api_types import ArtworkType, CheckStatus, EvidenceConfidence, EvidencePhase
from .artifact_security import (
    ArchiveResult,
    BatchInspectionBudget,
    InspectionBudget,
    InspectionExecution,
    InspectionResourceError,
    InspectionTimeoutError,
    ManifestIssue,
    ManifestVerification,
    ManifestVerificationError,
    create_deterministic_archive,
    inspect_artifact_isolated,
    inspect_artifacts_bounded,
    verify_deterministic_archive,
    verify_manifest,
)
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
from .contracts import (
    export_manifest_schema,
    report_schema,
    submission_manifest_schema,
    validation_report_schema,
)
from .html_report import render_html_report, write_html_report
from .inspector_protocol import (
    PROTOCOL_VERSION,
    ExternalInspectionResult,
    InspectorAllowlist,
    InspectorAllowlistError,
    InspectorCapabilities,
    InspectorCapability,
    InspectorDiagnostic,
    InspectorExecutable,
    InspectorExecutionError,
    InspectorFilePin,
    InspectorIntegrityError,
    InspectorLimits,
    InspectorProtocolClient,
    InspectorProtocolError,
    InspectorRemoteError,
    InspectorResourceError,
    InspectorResponseError,
    InspectorTimeoutError,
)
from .inspectors import (
    ArtifactInspection,
    ArtifactInspectionError,
    ArtifactParseError,
    UnsupportedArtifactError,
    inspect_artifact,
)
from .manifest import ArtifactRecord, ExportManifest
from .manuscript import ManuscriptAudit, ManuscriptPageAudit, audit_manuscript_pdf
from .manuscript_matching import (
    FigurePlacementMatch,
    ManuscriptPlacementAudit,
    PlacedObject,
    PlacementMatchMethod,
    PlacementMatchStatus,
    match_manuscript_figures,
)
from .migration import figure_spec_from_target, project_from_v1, project_spec_from_v1
from .models import (
    AggregateExpression,
    AllExpression,
    AnyExpression,
    ConstraintOperator,
    ContentKind,
    FigureRole,
    NotExpression,
    OutputFormat,
    ProbeExpression,
    ProfileCoordinate,
    ProfileGovernance,
    ProfileStatus,
    QuantifierExpression,
    RuleApplicability,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    SourceKind,
    SourceRef,
    VenueKind,
    VenueProfile,
    VenueResolutionWarning,
    VenueRule,
    VerificationMode,
)
from .observations import Observation, ObservationSet
from .planning import (
    CompliancePlan,
    CoverageRequirement,
    CoverageResult,
    CoverageStatus,
    ExportPlan,
    ExportSetting,
    FigurePlan,
    PlanAssessment,
    PlanEvidence,
    plan_export,
)
from .probes import ProbeDefinition, ProbeValueKind, get_probe, list_probes
from .profile_lock import (
    ProfileLock,
    ProfileLockError,
    load_profile_lock,
    resolve_locked_profile,
    verify_profile_lock,
)
from .project import FigureConfig, ProjectConfig, write_profile_lock
from .project_api import ExecutablePlan, FigureTarget, PlanPolicyError, Project, enforce_assessment
from .provenance import collect_environment_provenance
from .registry import (
    list_profiles,
    load_profile,
    profile_schema,
    resolve_profile,
    search_profiles,
    translate_v2_profile,
    validate_profile_data,
)
from .remediation import (
    Remediation,
    RemediationKind,
    RemediationPlan,
    plan_remediation,
)
from .remote_registry import RegistryCapabilityError, RegistryClient, RegistryDiagnostic
from .rule_expressions import evaluate_constraint, evaluate_expression, expression_probes
from .sarif import reports_to_sarif
from .specs import (
    PROJECT_SCHEMA_VERSION,
    DeliverableSpec,
    FigureSpec,
    ManualAttestation,
    ManuscriptFormat,
    ManuscriptMatchHint,
    ManuscriptSpec,
    PanelSpec,
    ProjectSpec,
    Waiver,
    project_schema,
)
from .standards import (
    JATS_VERSION,
    RO_CRATE_CONTEXT,
    RO_CRATE_PROFILE,
    RO_CRATE_VERSION,
    submission_manifest_to_jats,
    submission_manifest_to_ro_crate,
    write_ro_crate_metadata,
)
from .style import StyleContext
from .submission import BundleResult, Submission, SubmissionItemResult
from .target import Target, target
from .transactional_export import ExportResult
from .units import Quantity, UnitDefinition, UnitDimension, convert_value, list_units
from .visual_diagnostics import (
    AccessibilityPreview,
    PreviewImage,
    VisualDiagnostic,
    VisualDiagnostics,
    diagnose_visual,
    render_accessibility_previews,
    write_accessibility_previews,
)
from .webapp import LocalWebError, ServerInfo, create_server
from .webapp import serve as serve_local_workspace

try:
    __version__ = version("researchplot-venues")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

# The concise public spelling does not discard the explicit model name.
Profile = VenueProfile
ValidationReport = PlanAssessment
CheckResult = Finding
PhaseCoverage = CoverageResult
SourceReference = SourceRef

_LEGACY_NAMES = {
    "PlotStyle",
    "accuracy_vs_epoch",
    "bar",
    "boxplot",
    "confusion_matrix",
    "contour_plot",
    "dendrogram",
    "error_band",
    "heatmap",
    "hexbin",
    "histogram",
    "learning_curves",
    "line",
    "loss_vs_epoch",
    "pairplot",
    "pie",
    "precision_recall_curve",
    "quiver",
    "radar_chart",
    "roc_curve",
    "sankey",
    "scatter",
    "stacked_bar",
    "surface_3d",
    "time_series",
    "violinplot",
}


def __getattr__(name: str) -> Any:
    """Lazily expose deprecated plotting helpers without importing their extras."""

    if name in _LEGACY_NAMES:
        return getattr(import_module(".plots", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_NAMES)


__all__ = [
    "ArtifactInspection",
    "ArtifactInspectionError",
    "ArtifactParseError",
    "ArtifactRecord",
    "AccessibilityPreview",
    "AggregateExpression",
    "AllExpression",
    "AnyExpression",
    "ArchiveResult",
    "ArtworkType",
    "BatchInspectionBudget",
    "BundleResult",
    "CompliancePolicyError",
    "ConstraintOperator",
    "ContentKind",
    "CompliancePlan",
    "CheckResult",
    "CheckStatus",
    "CoverageRequirement",
    "CoverageResult",
    "CoverageStatus",
    "DeliverableSpec",
    "ExportManifest",
    "ExportPlan",
    "ExportResult",
    "ExportSetting",
    "ExternalInspectionResult",
    "EvidenceConfidence",
    "EvidencePhase",
    "ExecutablePlan",
    "FigureConfig",
    "FigurePlan",
    "FigureRole",
    "FigureSpec",
    "FigureTarget",
    "FigurePlacementMatch",
    "Finding",
    "InspectionBudget",
    "InspectionExecution",
    "InspectionResourceError",
    "InspectionTimeoutError",
    "InspectorAllowlist",
    "InspectorAllowlistError",
    "InspectorCapabilities",
    "InspectorCapability",
    "InspectorDiagnostic",
    "InspectorExecutable",
    "InspectorExecutionError",
    "InspectorFilePin",
    "InspectorIntegrityError",
    "InspectorLimits",
    "InspectorProtocolClient",
    "InspectorProtocolError",
    "InspectorRemoteError",
    "InspectorResourceError",
    "InspectorResponseError",
    "InspectorTimeoutError",
    "JATS_VERSION",
    "LocalWebError",
    "ManifestIssue",
    "ManifestVerification",
    "ManifestVerificationError",
    "ManuscriptAudit",
    "ManuscriptPageAudit",
    "ManuscriptPlacementAudit",
    "Outcome",
    "Observation",
    "ObservationSet",
    "NotExpression",
    "OutputFormat",
    "PROJECT_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "PlanAssessment",
    "PlanEvidence",
    "PlanPolicyError",
    "PhaseCoverage",
    "Policy",
    "Profile",
    "ProfileCoordinate",
    "ProfileGovernance",
    "ProfileLock",
    "ProfileLockError",
    "ProfileStatus",
    "PreviewImage",
    "ProbeDefinition",
    "ProbeExpression",
    "ProbeValueKind",
    "Project",
    "ProjectConfig",
    "ProjectSpec",
    "Report",
    "Remediation",
    "RemediationKind",
    "RemediationPlan",
    "RegistryCapabilityError",
    "RegistryClient",
    "RegistryDiagnostic",
    "RuleApplicability",
    "RuleConstraint",
    "RuleEngine",
    "RuleLevel",
    "RulePhase",
    "RO_CRATE_CONTEXT",
    "RO_CRATE_PROFILE",
    "RO_CRATE_VERSION",
    "SourceRef",
    "SourceReference",
    "SourceKind",
    "StyleContext",
    "ManuscriptFormat",
    "ManuscriptMatchHint",
    "ManuscriptSpec",
    "ManualAttestation",
    "PanelSpec",
    "PlacedObject",
    "PlacementMatchMethod",
    "PlacementMatchStatus",
    "Submission",
    "SubmissionItemResult",
    "ServerInfo",
    "Target",
    "TargetContext",
    "UnsupportedArtifactError",
    "UnitDefinition",
    "UnitDimension",
    "VenueKind",
    "VenueProfile",
    "VenueResolutionWarning",
    "VenueRule",
    "ValidationReport",
    "Verdict",
    "VisualDiagnostic",
    "VisualDiagnostics",
    "Waiver",
    "VerificationMode",
    "Quantity",
    "QuantifierExpression",
    "__version__",
    "audit_manuscript_pdf",
    "export_manifest_schema",
    "enforce_assessment",
    "figure_spec_from_target",
    "evaluate_constraint",
    "evaluate_expression",
    "expression_probes",
    "convert_value",
    "collect_environment_provenance",
    "create_deterministic_archive",
    "create_server",
    "diagnose_visual",
    "get_probe",
    "inspect_artifact",
    "inspect_artifact_isolated",
    "inspect_artifacts_bounded",
    "list_profiles",
    "list_probes",
    "list_units",
    "match_manuscript_figures",
    "load_profile",
    "load_profile_lock",
    "profile_schema",
    "project_schema",
    "plan_export",
    "plan_remediation",
    "project_from_v1",
    "project_spec_from_v1",
    "reports_to_sarif",
    "report_schema",
    "resolve_profile",
    "resolve_locked_profile",
    "render_accessibility_previews",
    "render_html_report",
    "search_profiles",
    "submission_manifest_schema",
    "validation_report_schema",
    "submission_manifest_to_jats",
    "submission_manifest_to_ro_crate",
    "serve_local_workspace",
    "target",
    "translate_v2_profile",
    "validate_profile_data",
    "write_profile_lock",
    "verify_profile_lock",
    "verify_deterministic_archive",
    "verify_manifest",
    "write_accessibility_previews",
    "write_html_report",
    "write_ro_crate_metadata",
    *sorted(_LEGACY_NAMES),
]
