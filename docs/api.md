# Python API

The 2.0 API is project-oriented. The v1 `Target` API remains supported throughout 2.x
for phase-local checks and migration.

## Primary project workflow

```python
import researchplot as rp

project = rp.Project.load("researchplot.toml")
plan = project.plan(frozen=True)
report = plan.check()

if report.verdict is rp.Verdict.COMPLIANT:
    bundle = project.bundle("dist/submission")
```

`frozen=True` requires `ProjectSpec.lock_path` and verifies the exact profile before
checking artifacts.

## Figure workflow

```python
figure = project.figure("figure-1")

with figure.style(deliverable="main", latex=False) as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(x, y)
    report = figure.check(fig=fig)
    export = figure.export(fig, policy="violations")
```

`figure.check()` combines configured bundle metadata, existing deliverables, and an
optional live Matplotlib figure. Use `figure.validate(fig)` or `figure.audit(path)` only
when a raw phase-local compatibility `Report` is intentional.

## Project specifications

::: researchplot.specs
    options:
      members:
        - ProjectSpec
        - FigureSpec
        - PanelSpec
        - DeliverableSpec
        - ManuscriptSpec
        - ManuscriptMatchHint
        - ManuscriptFormat
        - ManualAttestation
        - Waiver
        - project_schema
      members_order: source
      show_root_heading: false

## Planning and coverage

::: researchplot.planning
    options:
      members:
        - CompliancePlan
        - FigurePlan
        - ExportPlan
        - ExportSetting
        - CoverageRequirement
        - CoverageResult
        - CoverageStatus
        - PlanEvidence
        - PlanAssessment
        - plan_export
      members_order: source
      show_root_heading: false

## Project execution

::: researchplot.project_api
    options:
      members:
        - Project
        - ExecutablePlan
        - FigureTarget
        - PlanPolicyError
        - enforce_assessment
      members_order: source
      show_root_heading: false

## Profiles and locks

```python
profile = rp.resolve_profile("nature@2026.08.0")
profiles = rp.list_profiles()
matches = rp.search_profiles("vision")
lock = rp.ProfileLock.from_profile(profile)
report_v2_schema = rp.validation_report_schema()
```

::: researchplot.models
    options:
      members:
        - ProfileCoordinate
        - VenueProfile
        - VenueRule
        - SourceRef
        - ProfileGovernance
        - ProfileStatus
        - RuleApplicability
        - RuleConstraint
        - ProbeExpression
        - AllExpression
        - AnyExpression
        - NotExpression
        - QuantifierExpression
        - AggregateExpression
        - ConstraintOperator
        - RuleLevel
        - RulePhase
        - VerificationMode
        - VenueKind
        - FigureRole
        - ContentKind
        - OutputFormat
      members_order: source
      show_root_heading: false

::: researchplot.profile_lock
    options:
      members:
        - ProfileLock
        - ProfileLockError
        - load_profile_lock
        - verify_profile_lock
        - resolve_locked_profile
      members_order: source
      show_root_heading: false

## Artifact inspection and remediation

```python
execution = rp.inspect_artifact_isolated("figure.pdf")
inspection = execution.inspection
remediation = rp.plan_remediation(inspection)
```

::: researchplot.artifact_security
    options:
      members:
        - InspectionBudget
        - BatchInspectionBudget
        - InspectionExecution
        - inspect_artifact_isolated
        - inspect_artifacts_bounded
        - verify_manifest
        - create_deterministic_archive
        - verify_deterministic_archive
        - ManifestIssue
        - ManifestVerification
        - ArchiveResult
      members_order: source
      show_root_heading: false

::: researchplot.remediation
    options:
      members:
        - Remediation
        - RemediationPlan
        - RemediationKind
        - plan_remediation
      members_order: source
      show_root_heading: false

## Visual and manuscript diagnostics

::: researchplot.visual_diagnostics
    options:
      members:
        - AccessibilityPreview
        - PreviewImage
        - VisualDiagnostic
        - VisualDiagnostics
        - render_accessibility_previews
        - diagnose_visual
        - write_accessibility_previews
      members_order: source
      show_root_heading: false

::: researchplot.manuscript
    options:
      members:
        - ManuscriptAudit
        - ManuscriptPageAudit
        - audit_manuscript_pdf
      members_order: source
      show_root_heading: false

The structural result optionally carries a conservative placement audit when configured
figures are supplied.

::: researchplot.manuscript_matching
    options:
      members:
        - ManuscriptPlacementAudit
        - FigurePlacementMatch
        - PlacedObject
        - PlacementMatchMethod
        - PlacementMatchStatus
        - match_manuscript_figures
      members_order: source
      show_root_heading: false

Placement matching is conservative; a complete placement set is not yet integrated
with venue-specific manuscript rules in `CompliancePlan`.

## Allowlisted external inspectors

The optional extension boundary is a versioned, JSON-only subprocess protocol. Exact
executables and support files must be explicitly allowlisted; capability negotiation,
hash pins, timeouts, output limits, and typed observations fail closed.

::: researchplot.inspector_protocol
    options:
      members:
        - InspectorProtocolClient
        - InspectorAllowlist
        - InspectorExecutable
        - InspectorFilePin
        - InspectorLimits
        - InspectorCapabilities
        - InspectorCapability
        - ExternalInspectionResult
      members_order: source
      show_root_heading: false

## JATS and RO-Crate

::: researchplot.standards
    options:
      members:
        - submission_manifest_to_jats
        - submission_manifest_to_ro_crate
        - write_ro_crate_metadata
      members_order: source
      show_root_heading: false

## HTML and local workspace

```python
rp.write_html_report(report, "build/report.html")
server, info = rp.create_server(port=0)
print(info.url)
```

::: researchplot.html_report
    options:
      members_order: source
      show_root_heading: false

::: researchplot.webapp
    options:
      members:
        - ServerInfo
        - LocalWebError
        - create_server
        - serve
      members_order: source
      show_root_heading: false

## Environment provenance

```python
provenance = dict(rp.collect_environment_provenance())
```

The tuple contains stable library/runtime/backend/font-setting/LaTeX keys and an
rcParams digest. It intentionally omits usernames, hostnames, and filesystem paths.

## Compatibility API

```python
target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)
live = target.validate(fig)
saved = target.audit("figure.pdf")
result = target.export(fig, "figure.pdf", policy="complete")
```

::: researchplot.target
    options:
      members:
        - Target
        - target
      members_order: source
      show_root_heading: false

`researchplot.plots` is loaded lazily only when a deprecated plotting name is used.
Install `[plots]` for its optional NumPy/pandas/Seaborn/SciPy/scikit-learn integrations.

## Exceptions and serialization

Invalid project/profile/configuration input, unsafe artifact input, parser failures,
resource limits, and unavailable operational capabilities are exceptions. Policy
exceptions retain their report or assessment. Do not catch a broad exception merely to
force publication.

`to_dict()` returns JSON-compatible values and stable enum strings. Human terminal
phrasing may improve without a schema change. See [report and manifest formats](formats.md).
