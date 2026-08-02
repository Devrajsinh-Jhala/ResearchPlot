# ResearchPlot

**Source-backed venue compliance, artifact auditing, and submission bundles for
Matplotlib figures.**

[![PyPI](https://img.shields.io/pypi/v/researchplot-venues.svg)](https://pypi.org/project/researchplot-venues/)
[![Python](https://img.shields.io/pypi/pyversions/researchplot-venues.svg)](https://pypi.org/project/researchplot-venues/)
[![CI](https://github.com/Devrajsinh-Jhala/ResearchPlot/actions/workflows/ci.yml/badge.svg)](https://github.com/Devrajsinh-Jhala/ResearchPlot/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://devrajsinh-jhala.github.io/ResearchPlot/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/LICENSE)

ResearchPlot turns a venue name into a reproducible figure target. It styles a
Matplotlib figure at the requested physical width, evaluates source-backed rules,
exports transactionally, audits the files that were actually written, and can assemble a
submission bundle with hashes and provenance.

It is a compliance assistant, not an acceptance guarantee. Every result distinguishes
verified facts from recommendations, inferences, manual requirements, and checks the
software could not establish.

![ResearchPlot compliance architecture](https://raw.githubusercontent.com/Devrajsinh-Jhala/ResearchPlot/main/docs/assets/architecture.svg)

## Install

```bash
python -m pip install researchplot-venues
```

The distribution is called `researchplot-venues`; the import and command remain
`researchplot`:

```python
import researchplot as rp
```

ResearchPlot 1.x requires Python 3.11 or newer. It works offline at runtime, does not
download fonts or templates, and does not require LaTeX. The old plotting wrappers are
not shipped in 1.x. If an older project still needs them, pin the final 0.2 release in
a separate environment:

```bash
python -m pip install "researchplot-venues[plots]==0.2.1"
```

## Create and verify one figure

```python
from pathlib import Path

import matplotlib.pyplot as plt
import researchplot as rp

target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)

with target.style() as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot([0, 1, 2, 3], [0, 1, 4, 9], marker="o")
    ax.set(xlabel="Input", ylabel="Response")

    result = target.export(fig, Path("submission") / "figure1.pdf", policy="complete")

print(result.report.verdict)
print(result.paths)
print(result.manifest_path)
plt.close(fig)
```

`policy="complete"` rejects both known required violations and unresolved required
checks. It stages output in a temporary location, audits the resulting artifact, and
starts the final commit only after the selected policy succeeds. Handled commit failures
trigger rollback. Multi-file destinations are replaced sequentially, so abrupt process
termination and non-cooperating concurrent writers remain outside that guarantee.

## Build a submission bundle

```python
submission = rp.Submission(
    "nature@2026.08.0",
    output_dir="submission",
    policy="complete",
)
submission.add(
    "figure1",
    fig,
    role="main",
    width="single",
    content="line-art",
    formats=("pdf",),
    alt_text="A line chart whose response rises quadratically with input.",
    source_data="data/figure1.csv",
)
bundle = submission.build()
print(bundle.manifest_path)
print(bundle.passed)
```

The manifest records the ResearchPlot version, immutable profile coordinate and digest,
full source metadata and caveats, target metadata, file SHA-256 hashes, automated
findings, manual attestations, captions, alt text, and copied source-data files when
provided.

## Compliance has three verdicts

| Verdict | Meaning |
| --- | --- |
| `COMPLIANT` | Every applicable required rule was checked and passed. |
| `NON_COMPLIANT` | At least one applicable required rule failed. |
| `INDETERMINATE` | No required rule failed, but at least one required rule could not be established. |

Rule level and check outcome are separate. A recommendation can warn without blocking,
while inferred guidance is informational. A required `SKIP` is never presented as a
pass.

Export policies let CI choose the boundary:

- `violations`: block known required failures;
- `complete`: also block unresolved required checks;
- `off`: always return evidence without blocking.

## Profiles are versioned evidence

Coordinates use `<profile-id>@<revision>`, for example
`nature@2026.08.0` or `cvpr-2026@2026.08.0`. Pinning the coordinate makes a paper
reproducible. An unpinned ID or friendly alias resolves to a bundled revision and
warns, so a changing default cannot go unnoticed.

The initial 1.0 catalog covers IEEE journals, Nature, generic Elsevier artwork,
NeurIPS 2026, ICML 2026, CVPR 2026, ACL 2026, PLOS Biology, and ACM `acmart`.
Missing official guidance remains unspecified rather than being invented. Run:

```bash
researchplot profile list
researchplot profile show nature@2026.08.0
researchplot explain figure.width.single --profile nature@2026.08.0
```

| Profile | Final width options |
| --- | --- |
| `ieee-journal@2026.08.0` | 88.9 / 181.864 mm |
| `nature@2026.08.0` | 89 / 183 mm |
| `elsevier-generic@2026.08.0` | 30 / 90 / 140 / 190 mm |
| `neurips-2026@2026.08.0` | 139.7 mm |
| `icml-2026@2026.08.0` | 82.55 / 171.45 mm |
| `cvpr-2026@2026.08.0` | 83.34375 / 174.625 mm |
| `acl-2026@2026.08.0` | 77 / 160 mm |
| `plos-biology@2026.08.0` | 132 / 190.5 mm; 66.8–190.5 mm allowed range |
| `acm-acmart@2026.08.0` | Bundle metadata only; no generic width |

Profiles are bundled JSON validated against a published schema. Every rule states its
applicability, strength, verification mode, official source, section or page locator,
and verification date. Runtime resolution remains offline.

## Check files and projects in CI

```toml
# researchplot.toml
profile = "nature@2026.08.0"
policy = "complete"

[[figures]]
path = "figures/figure1.pdf"
role = "main"
width = "single"
content = "line-art"
alt_text = "A line chart comparing the measured response across four inputs."
source_data = "data/figure1.csv"
```

```bash
researchplot check --config researchplot.toml
researchplot check --config researchplot.toml --format json
researchplot check --config researchplot.toml --format sarif > researchplot.sarif
researchplot bundle build --config researchplot.toml
```

Exit codes are stable: `0` compliant, `1` non-compliant, `2` invalid input or a
required capability failure, and `3` indeterminate. SARIF output can be uploaded to
GitHub code scanning for inline annotations.

## What ResearchPlot inspects

- Exact physical width and maximum height, with explicit tolerance.
- Font family and size, line and marker sizes, and prohibited in-figure titles.
- Color-only series distinction when a profile includes the applicable accessibility
  rule, plus bundle alt-text presence for ACM.
- PDF page boxes across all pages, recursive font resources, embedding, and Type 3
  fonts.
- SVG physical dimensions, `viewBox`, text, font declarations, external references,
  and embedded assets.
- PNG, JPEG, and TIFF dimensions, effective DPI, color mode, bit depth, ICC metadata,
  compression, and file size.
- EPS format, `BoundingBox`, and `HiResBoundingBox` dimensions.

Unobservable properties are reported as skipped. They are not guessed and do not
silently pass.

## Documentation

- [Getting started](https://devrajsinh-jhala.github.io/ResearchPlot/getting-started/)
- [Architecture and compliance model](https://devrajsinh-jhala.github.io/ResearchPlot/architecture/)
- [Profile provenance](https://devrajsinh-jhala.github.io/ResearchPlot/profiles/)
- [Project configuration and CLI](https://devrajsinh-jhala.github.io/ResearchPlot/configuration/)
- [Submission bundles](https://devrajsinh-jhala.github.io/ResearchPlot/bundles/)
- [Migration from 0.2](https://devrajsinh-jhala.github.io/ResearchPlot/migration/)
- [Python API](https://devrajsinh-jhala.github.io/ResearchPlot/api/)

## Scope and safety

ResearchPlot does not interpret scientific correctness, detect manipulation, replace a
publisher's instructions, generate figures or alt text with AI, scrape templates at
runtime, or guarantee acceptance. A journal-specific instruction always overrides a
generic publisher profile. Review each finding's linked source and use
`researchplot profile show` to inspect all profile caveats.

ResearchPlot is MIT licensed. See
[CONTRIBUTING.md](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/CONTRIBUTING.md)
before proposing a profile or behavioral change, and cite the project using
[CITATION.cff](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/CITATION.cff).
