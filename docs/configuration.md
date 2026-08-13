# Project configuration

`researchplot.toml` is the reproducible boundary between a paper, its figure files, and
the venue evidence used to review them. ResearchPlot 2.0 writes and reads strict schema
version 3 for the primary project API.

## Complete example

```toml
schema_version = 3
profile = "nature@2026.08.0"
policy = "complete"
lock = "researchplot.lock.json"

[metadata]
paper_id = "example-2026"

[manuscript]
path = "manuscript/paper.pdf"
format = "pdf"
required = false

[[manuscript.matching_hints]]
figure = "figure-1"
pages = [3]
number = 1
caption = "Response increases across measured inputs."

[[figures]]
id = "figure-1"
number = 1
role = "main"
width = "single"
content = "line-art"
caption = "Response increases across measured inputs."
alt_text = "Line chart rising monotonically from input zero to three."
long_description = "The response is zero, one, four, and nine at inputs zero through three."
source_data = ["data/figure1.csv"]
data_table = "data/figure1-accessible.csv"
attachments = ["code/figure1.py"]

[[figures.panels]]
id = "panel-a"
label = "a"
alt_text = "Observed response."
source_data = ["data/figure1-a.csv"]

[[figures.deliverables]]
id = "main"
format = "pdf"
path = "figures/figure1.pdf"
required = true
preferred = true

[[figures.deliverables]]
id = "preview"
format = "png"
path = "figures/figure1.png"
required = false
preferred = false

[figures.waivers."figure.title.prohibited"]
profile_digest = "c1a79e3c48483773284ecde6024f7b65ec0fa657b88f29fd8b7e23e421110ffa"
reviewer = "A. Researcher"
reason = "Recommendation reviewed for this draft; title retained during author review."
expires_on = "2026-09-30"
```

## Root fields

| Field | Required | Meaning |
| --- | :---: | --- |
| `schema_version` | Yes | Must be integer `3`. |
| `profile` | Yes | Exact installed coordinate such as `nature@2026.08.0`. |
| `policy` | No | `complete` (default), `violations`, or `off`. |
| `lock` | No | Profile lock path; required by `Project.plan(frozen=True)`. |
| `figures` | Yes | Non-empty array of logical figure tables. |
| `manuscript` | No | Compiled manuscript metadata; only PDF structural audit is implemented. |
| `metadata` | No | JSON-compatible project metadata. |

Unknown fields are errors. A schema-v3 project never silently accepts a bare profile
alias.

## Figure and panel fields

Every figure needs a stable lowercase ID and at least one required deliverable. IDs may
contain lowercase letters, numbers, underscores, and hyphens. Deliverable IDs and paths
must be unique within the project.

`role` and `content` select conditional profile rules. Common content values include
`line-art`, `data-visualization`, `combination`, `halftone`, and `photograph`; use the
enum values exposed by the installed release for authoritative choices.

Descriptions and data references are author-provided evidence. ResearchPlot does not
generate or semantically grade them. A `PanelSpec` can carry its own label,
description, source data, and metadata.

Each manual attestation is keyed by rule ID and records reviewer, ISO date, rationale,
and referenced evidence. It can satisfy only a profile rule classified as manual.

For example, an `acm-acmart` project can attest to its manual distinct-description
rule:

```toml
[figures.attestations."metadata.alt_text.distinct_from_caption"]
reviewer = "A. Researcher"
date = "2026-08-03"
rationale = "The description states the key trend not repeated in the caption."
evidence = ["reviews/figure1-accessibility.md"]
```

Each waiver records the rule ID, exact profile digest, reviewer, reason, and ISO expiry.
A digest mismatch is a project error; an expired waiver remains serialized but is not
active in planning. No waiver changes the venue verdict. String-only
attestation/waiver values are deprecated v1 compatibility forms.

## Deliverables and export planning

Each `DeliverableSpec` defines:

- stable `id`;
- one supported output `format`;
- optional `path`;
- whether it is `required`;
- whether it is the single `preferred` representation;
- optional metadata.

Allowed formats and required companion formats come from profile rules. The planner
does not treat every allowed alternative as a mandatory output. If no profile format
rule exists, the export plan marks the format as unspecified and requires an explicit
choice.

```python
project = rp.Project.load("researchplot.toml")
figure = project.figure("figure-1")

print(figure.export_plan.allowed_formats)
print(figure.export_plan.selected_formats)
print(figure.export_plan.required_companions)
print(figure.export_plan.settings)
```

## Path behavior

Paths are relative to the directory containing `researchplot.toml`. The schema-v3
loader rejects absolute paths, drive/root prefixes, `..` components, and resolved
symlink/junction targets outside that project root. Bundle and archive verifiers
independently reject traversal, absolute member paths, symlinks, and case-colliding
entries.

Referenced deliverables do not need to exist while planning or authoring. Checks skip
missing optional artifacts; required coverage then remains indeterminate.

## Load, plan, and check

```python
import researchplot as rp

project = rp.Project.load("researchplot.toml")
plan = project.plan(frozen=True)
report = plan.check()
```

Pass live Matplotlib figures by stable ID when available:

```python
report = plan.check(live_figures={"figure-1": fig})
```

Unknown live IDs are errors. Existing deliverable paths are audited and configured
captions, descriptions, source data, and attestations contribute bundle-phase evidence.

From the CLI:

```bash
researchplot check --config researchplot.toml --frozen
researchplot check --config researchplot.toml --format json --output build/report.json
researchplot check --config researchplot.toml --format sarif --output build/report.sarif
```

## Locks and frozen CI

```bash
researchplot profile lock nature@2026.08.0 --output researchplot.lock.json
```

Commit both `researchplot.toml` and `researchplot.lock.json`. Frozen planning checks the
coordinate, canonical profile digest, document digest, and source-content digest values
that are available in the profile. A missing or mismatched lock fails before artifact
inspection.

```mermaid
sequenceDiagram
    participant CI
    participant L as Lock verifier
    participant P as Compliance plan
    participant A as Artifact inspector
    CI->>L: check --frozen
    alt lock missing or digest mismatch
        L-->>CI: input error; inspect nothing
    else exact lock verified
        L->>P: compile requirements
        P->>A: inspect configured evidence
        A-->>CI: verdict + coverage
    end
```

## Migration

The v1 reader remains available throughout 2.x. `researchplot migrate` creates a
separate schema-v3 file by default, leaving the source untouched. Review its output;
fields without a safe one-to-one translation remain explicit migration findings.

See [Migration](migration.md) for API and plotting-wrapper compatibility.

## GitHub Action

The repository ships a composite action that installs an exact release, emits SARIF,
and preserves ResearchPlot's exit code:

```yaml
permissions:
  contents: read
  security-events: write

steps:
  - uses: actions/checkout@v4
  - uses: Devrajsinh-Jhala/ResearchPlot@v2.0.0
    with:
      version: "2.0.0"
      config: researchplot.toml
      frozen: "true"
      upload-sarif: "true"
```

Frozen mode is the action default. Commit the lock named by the project before enabling
it. For a direct audit instead of a project, provide `profile`, `paths`, `lock`, and
target metadata explicitly.
