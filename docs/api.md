# Python API

The 1.0 API is target-oriented. Create one `Target`, then style, validate, audit, and
export through it so the profile and figure intent cannot drift between operations.

## Primary workflow

```python
import researchplot as rp

target = rp.target(
    "nature@2026.08.0",
    role=rp.FigureRole.MAIN,
    width="single",
    content=rp.ContentKind.LINE_ART,
)

with target.style() as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(x, y)

result = target.export(fig, "figure1.pdf", policy=rp.Policy.COMPLETE)
```

String enum values are accepted at user-facing boundaries; typed code can use enums.

## Package API

::: researchplot
    options:
      members:
        - target
        - resolve_profile
        - list_profiles
        - search_profiles
        - load_profile
        - profile_schema
        - report_schema
        - export_manifest_schema
        - submission_manifest_schema
        - validate_profile_data
        - Submission
        - Target
        - StyleContext
        - ExportResult
        - BundleResult
        - Report
        - Finding
        - VenueProfile
        - VenueRule
        - SourceRef
        - Verdict
        - Outcome
        - RuleLevel
        - VerificationMode
        - FigureRole
        - ContentKind
        - OutputFormat
        - Policy
      members_order: source
      show_root_heading: false

## Target creation

```python
target = rp.target(
    profile,
    *,
    role,
    width,
    content,
)
```

`profile` accepts a `VenueProfile` or query string. Production code should use an exact
coordinate.

## Target operations

```python
with target.style(latex=False, overrides={"axes.grid": True}) as style:
    fig, ax = style.subplots(aspect=0.62)

live_report = target.validate(fig)
file_report = target.audit("figure1.pdf")
result = target.export(fig, "figure1.pdf", policy="complete")
```

- `style()` uses a reversible `matplotlib.rc_context`.
- `validate()` inspects live artists and dimensions.
- `audit()` inspects an existing supported artifact.
- `export()` stages, post-audits, and commits according to policy.

Style `overrides` are validated Matplotlib `rcParams`; they layer after profile and
target settings but remain subject to validation. LaTeX is opt-in through `style()`.

## Reports

```python
report.verdict
report.failures
report.warnings
report.unresolved
report.findings
report.to_dict()
```

Use `Verdict.COMPLIANT`, `Verdict.NON_COMPLIANT`, and `Verdict.INDETERMINATE` rather
than coercing a report to Boolean. Rule levels and outcomes are separate enums.

## Profiles

```python
profile = rp.resolve_profile("nature@2026.08.0")
profiles = rp.list_profiles()
matches = rp.search_profiles("computer vision")
candidate = rp.load_profile("profiles/candidate.json")
schema = rp.profile_schema()
report_contract = rp.report_schema()
export_contract = rp.export_manifest_schema()
manifest_contract = rp.submission_manifest_schema()
```

The exact profile attributes and immutable typed rule structures are documented below.

::: researchplot.models
    options:
      members_order: source
      show_root_heading: false

## Exceptions

ResearchPlot uses intentional exception classes for invalid profiles/configuration,
unknown or ambiguous resolution, unavailable required capabilities, compliance-policy
blocking, and export transactions. A policy-blocking exception retains the report so a
caller can render or persist the evidence.

Do not catch `Exception` merely to force publication. Handle invalid input separately
from `NON_COMPLIANT` and `INDETERMINATE` evidence.

## Serialization

`to_dict()` returns JSON-compatible values and stable enum strings. Human `str()` output
is for terminal display and may improve without a major release. See
[Manifest and report formats](formats.md).
