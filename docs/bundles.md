# Export and submission bundles

ResearchPlot separates figure export, submission-directory construction, archive
verification, and metadata interoperability. Each boundary has different guarantees.

## Figure export

```python
figure = project.figure("figure-1")

with figure.style(deliverable="main") as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(x, y)
    result = figure.export(fig, policy="violations")

print(result.paths)
print(result.report.verdict)
print(result.manifest_path)
```

Export stages the candidate artifact, inspects the actual serialized file, evaluates
policy, writes a single-export manifest, and commits accepted output. A handled commit
failure rolls back. Multiple paths are replaced sequentially, so abrupt process
termination and non-cooperating concurrent writers remain outside the transaction
guarantee.

The planner distinguishes allowed formats, selected formats, a preferred format, and
required companions. It never interprets every allowed alternative as mandatory.

## Build a project submission directory

```python
import researchplot as rp

project = rp.Project.load("researchplot.toml")
result = project.bundle("dist/submission")

print(result.path)
print(result.manifest_path)
print(result.passed)
```

Or use the CLI:

```bash
researchplot bundle build \
  --config researchplot.toml \
  --output dist/submission
```

The builder stages a new directory and refuses to replace an existing destination. It
normalizes portable relative names, copies source data where representable, records
reports and profile provenance, and hashes every artifact.

!!! warning "Current schema-v3 bridge"

    `Project.bundle()` currently delegates to the v1 submission manifest. An
    existing-file figure must have one existing preferred required deliverable. The
    bridge accepts one source-data file per figure and rejects generic attachments or
    multiple source files instead of losing their semantics. The destination argument
    is a directory; `policy=` and `ro_crate=` are not `Project.bundle()` parameters.

The lower-level v1-compatible `Submission` class remains available throughout 2.x:

```python
submission = rp.Submission(
    "nature@2026.08.0",
    output_dir="dist/submission",
    policy="complete",
)
submission.add(
    "figure1",
    fig,
    role="main",
    width="single",
    content="line-art",
    formats=("pdf",),
    alt_text="Line chart with a monotonic increase.",
    caption="Response increases across measured inputs.",
    source_data="data/figure1.csv",
)
result = submission.build()
```

## Verify the directory manifest

```python
verification = rp.verify_manifest("dist/submission")

if not verification.valid:
    for issue in verification.issues:
        print(issue.code, issue.path, issue.message)
```

Strict verification checks manifest shape, member paths, case collisions, file type,
size, digest, missing members, unexpected members, and symlink/junction escapes without
trusting archive names.

## Create and verify a deterministic archive

```python
archive = rp.create_deterministic_archive(
    "dist/submission",
    "dist/submission.zip",
)
print(archive.sha256)

verification = rp.verify_deterministic_archive("dist/submission.zip")
assert verification.valid
```

The writer supports `.zip` and uncompressed `.tar`. Both include only the manifest and
its declared artifacts, in stable order, with normalized timestamps, ownership, and
permissions. `SOURCE_DATE_EPOCH` controls the canonical archive timestamp when set;
otherwise ResearchPlot uses a fixed reproducible epoch. The writer refuses an existing
destination and places temporary data beside the final path.

Transactional PDF and SVG export also removes volatile creation dates, fixes the
ResearchPlot creator metadata, and uses a profile-derived SVG hash salt. Repeated exports
of the same figure and target therefore have stable vector bytes under the tested
Matplotlib backend and dependency set.

## JATS 1.4 metadata

Convert an existing submission manifest into a JATS `<fig-group>` fragment:

```python
jats = rp.submission_manifest_to_jats(result.manifest_path)
Path("dist/figures.xml").write_text(jats, encoding="utf-8")
```

The converter includes available labels, captions, alt text, graphics, and MIME hints.
It does not inspect the referenced files or invent missing prose. Long descriptions,
panel grouping, and accessible data alternatives are limited by what the current v1
submission manifest can represent.

## RO-Crate 1.3 metadata

```python
crate = rp.submission_manifest_to_ro_crate(
    result.manifest_path,
    name="Example paper figure evidence",
    creators=["A. Researcher"],
    license="https://creativecommons.org/licenses/by/4.0/",
)
rp.write_ro_crate_metadata(crate, "dist/ro-crate-metadata.json")
```

The converter links declared artifacts, hashes, profile identity, sources, and optional
people from the manifest. It is a metadata projection, not a complete archive builder.

## What a bundle proves

The manifest records the profile coordinate/digest, source references and caveats,
target metadata, emitted artifact paths, formats, byte counts, SHA-256 hashes, reports,
and the author metadata supported by the bridge. It does not prove scientific
correctness, authorship, venue acceptance, or that every official requirement has been
encoded.

Run a complete coverage-aware project check before building. A passing v1 bundle item
report is phase-local; it must not override project-level missing live or manuscript
evidence.
