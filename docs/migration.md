# Migrate from 0.2 to 1.0

ResearchPlot 1.0 is intentionally breaking. It replaces loosely coupled venue/style
arguments and binary pass semantics with one target, typed intent, tri-state reports,
and post-audited transactional exports.

## Before upgrading

1. Pin the last `0.2` release in a clean branch.
2. Make existing figure generation deterministic and commit representative outputs.
3. Replace top-level plotting wrappers with direct Matplotlib calls.
4. Upgrade Python to 3.11 or newer.
5. Read the profile sources and select exact 1.0 coordinates.

Install the new release:

```bash
python -m pip install --upgrade "researchplot-venues>=1,<2"
```

The distribution, import, and CLI names do not change.

## API map

| 0.2 | 1.0 | Why it changed |
| --- | --- | --- |
| `rp.use(venue, width=...)` | `rp.target(...).style()` | Target binds role, width, and content once. |
| `validate_figure(fig, venue=..., width=...)` | `target.validate(fig)` | Prevent validation/export target mismatch. |
| `audit_file(path, venue=..., artwork=...)` | `target.audit(path)` | Uses typed content and output format separately. |
| `export_figure(...) -> list[Path]` | `target.export(...) -> ExportResult` | Returns report and manifest; verifies written files. |
| `ArtworkType` | `ContentKind` + `OutputFormat` | Artwork semantics and file representation are different dimensions. |
| `report.passed: bool` | `report.verdict: Verdict` | A skipped required check is indeterminate, not passed. |
| Combined check status/severity | `Outcome` and `RuleLevel` | Recommendation failures are distinct from required failures. |
| Mutable publisher profile name | `<id>@<revision>` coordinate | Published rule evidence is immutable and reproducible. |
| `researchplot venues ...` | `researchplot profile ...` | Profiles are versioned evidence, not just venue names. |
| Top-level plotting helpers | Direct Matplotlib | The core focuses on compliance, audit, and bundles. |

## Style context

Before:

```python
with rp.use("nature", width="single") as style:
    fig, ax = style.subplots()
    ax.plot(x, y)
```

After:

```python
target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)

with target.style() as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(x, y)
```

The new context rejects an explicit height or `figsize` that contradicts the target;
`0.2` could silently clamp or override dimensions.

## Validation decisions

Before:

```python
report = rp.validate_figure(fig, venue="nature", width="single")
if report.passed:
    publish()
```

After:

```python
report = target.validate(fig)

if report.verdict is rp.Verdict.COMPLIANT:
    publish()
elif report.verdict is rp.Verdict.INDETERMINATE:
    request_manual_review(report.unresolved)
else:
    fix_required_failures(report.failures)
```

Do not replace `report.passed` with `report.verdict != NON_COMPLIANT`; that recreates
the `0.2` bug by treating unresolved required evidence as success.

## Artwork versus output format

`0.2` used values such as `vector`, `halftone`, and `line_art` in one enum even though
they represented two concepts.

In 1.0:

```python
target = rp.target(..., content=rp.ContentKind.LINE_ART)
result = target.export(fig, "figure1.pdf")  # PDF is inferred from the path
```

Content selects artwork-specific rules. The output suffix or an explicit
`OutputFormat` selects serialization rules.

## Export behavior

Before:

```python
paths = style.export(fig, "figure1", artwork="vector")
```

After:

```python
result = target.export(fig, "figure1.pdf", policy="complete")
paths = result.paths
report = result.report
manifest = result.manifest
```

The new pipeline checks the live figure, writes candidates in staging, audits the
actual files, applies policy, and commits with rollback for handled failures. Multi-file
destinations are replaced sequentially, so abrupt process termination and
non-cooperating concurrent writers remain outside that guarantee. Existing code must
choose an explicit suffix rather than relying on an artwork label to imply file
representations.

## Profile resolution

Before:

```python
profile = rp.resolve_venue("cvpr")
```

After:

```python
profile = rp.resolve_profile("cvpr-2026@2026.08.0")
```

Unpinned queries still help with discovery but warn. Commit an exact coordinate and
profile lock for reproducible work.

## CLI migration

```text
# 0.2
researchplot venues list
researchplot venues info nature
researchplot audit figure.pdf --venue nature --width single --artwork vector

# 1.0
researchplot profile list
researchplot profile show nature@2026.08.0
researchplot check figure.pdf --profile nature@2026.08.0 \
  --role main --width single --content line-art
```

Exit code `3` is new and means required checks are unresolved. Update scripts that
previously assumed every nonzero result was a known violation.

## Legacy plotting wrappers

Functions such as `bar`, `line`, `pairplot`, and `roc_curve` are no longer exported as
the primary package API. During migration, install the optional compatibility surface:

```bash
python -m pip install "researchplot-venues[plots]==0.2.1"
```

Then import explicitly from `researchplot.plots` where the installed release provides
the frozen wrappers. They retain deprecation warnings and do not receive new compliance
features. Replace them with direct Matplotlib/Seaborn calls before relying on 1.x long
term.

```python
# Legacy
fig, ax = researchplot.line(x, y, "Time", "Accuracy", show=False)

# Native Matplotlib
with target.style() as style:
    fig, ax = style.subplots()
    ax.plot(x, y)
    ax.set(xlabel="Time", ylabel="Accuracy")
```

## Create a project contract

After individual figures work, add `researchplot.toml`, run
`researchplot profile lock nature@2026.08.0`, and commit both files. The configuration
makes CI and local checks use the same target metadata; the lock is an evidence snapshot
and the package version should also be pinned because 1.0 does not enforce the lock
automatically.

See [Project configuration](configuration.md) and [Compliance reports](compliance.md).
