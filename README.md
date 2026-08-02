# ResearchPlot

ResearchPlot creates Matplotlib figures at verified publication widths, explains
where each venue rule came from, validates live figures, and audits exported files.
It is a compliance assistant—not a guarantee that a publisher will accept a figure.

[![PyPI](https://img.shields.io/pypi/v/researchplot-venues.svg)](https://pypi.org/project/researchplot-venues/)
[![CI](https://github.com/Devrajsinh-Jhala/ResearchPlot/actions/workflows/ci.yml/badge.svg)](https://github.com/Devrajsinh-Jhala/ResearchPlot/actions/workflows/ci.yml)

## Installation

```bash
python -m pip install researchplot-venues
```

The PyPI distribution is named `researchplot-venues`; the Python package and CLI
remain `researchplot`, so existing imports and commands do not change.

The core requires Python 3.10+ and does not require LaTeX or network access. Install
the deprecated high-level plotting helpers separately:

```bash
python -m pip install "researchplot-venues[plots]"
```

## Venue-native workflow

```python
import matplotlib.pyplot as plt
import researchplot as rp

x = [0, 1, 2, 3]
y = [0, 1, 4, 9]

with rp.use("CVPR 2026", width="single") as style:
    fig, ax = style.subplots()
    ax.plot(x, y, marker="o")
    ax.set(xlabel="Input", ylabel="Output")

    report = style.validate(fig)
    print(report)
    paths = style.export(fig, "figure1", artwork="vector")

plt.close(fig)
```

Exact IDs are reproducible. A bare conference name such as `cvpr` resolves to the
newest bundled verified year and warns with the resolved ID. Unknown and ambiguous
names raise an error with suggestions; they never silently become IEEE.

```python
profile = rp.resolve_venue("Nature")
print(profile.width_options)  # ('single', 'double')
print(profile.width_mm("single"))  # 89.0
print(profile.sources)  # official guidance and verification dates
```

## Verified catalog

| Profile | Final widths | Scope |
| --- | --- | --- |
| `ieee-journal` | 88.9 / 181.864 mm | General IEEE journal artwork |
| `nature` | 89 / 183 mm | Flagship Nature journal |
| `elsevier-generic` | 30 / 90 / 140 / 190 mm | Generic Elsevier artwork |
| `neurips-2026` | 139.7 mm | NeurIPS 2026 template |
| `icml-2026` | 82.55 / 171.45 mm | ICML 2026 instructions |
| `cvpr-2026` | 83.34375 / 174.625 mm | CVPR 2026 author kit |
| `acl-2026` | 77 / 160 mm | ACL 2026 formatting rules |

Every bundled rule is `required`, `recommended`, or `inferred`. Missing official
guidance stays unspecified. Profiles include official URLs, verification dates,
scope, and caveats; all runtime resolution is offline.

## Validation and file auditing

```python
report = rp.validate_figure(fig, venue="nature", width="single")
if not report.passed:
    for failure in report.failures:
        print(failure.message, failure.source_urls)

report = rp.audit_file(
    "figure1.pdf",
    venue="nature",
    width="single",
    artwork="vector",
)
print(report.to_dict())
```

Required violations fail; recommendations warn; inferred guidance is informational.
Anything the validator cannot establish is marked `skip`, not passed. Strict export
blocks only required failures.

The auditor supports PDF, SVG, PNG, JPEG, TIFF, and EPS. It checks applicable file
format, physical dimensions, raster DPI and color mode metadata, PDF font embedding
and Type 3 fonts, SVG dimensions/text presence, and EPS bounding boxes.

## Command line

```bash
researchplot venues list
researchplot venues search cvpr
researchplot venues info cvpr-2026
researchplot doctor --venue nature
researchplot audit figure.pdf --venue nature --width single --artwork vector
researchplot audit figure.pdf --venue nature --width single --artwork vector --json
```

Exit code `0` means no required failures, `1` means compliance failures, and `2`
means invalid input, an unreadable file, or a missing required capability.

## Legacy plotting helpers

The original positional interfaces remain available in `0.2.x` and display by
default, but emit a deprecation warning. They now return `(Figure, Axes)` (or a
Seaborn `PairGrid`) and accept keyword-only `ax=None` and `show=True`.

```python
from researchplot import bar, pairplot, stacked_bar

fig, ax = bar([3, 5, 2], ["A", "B", "C"], show=False)
```

Supported helpers: `bar`, `stacked_bar`, `scatter`, `line`, `histogram`, `boxplot`,
`heatmap`, `confusion_matrix`, `accuracy_vs_epoch`, `loss_vs_epoch`, `roc_curve`,
`precision_recall_curve`, `violinplot`, `contour_plot`, `pie`, `hexbin`, `pairplot`,
`learning_curves`, `time_series`, `radar_chart`, `dendrogram`, `quiver`, `surface_3d`,
`sankey`, and `error_band`.

`science`, `cell`, `springer`, and `pnas` remain unverified legacy styles only. The
generic `elsevier` alias also warns that a target journal can override its rules.
See the [migration guide](docs/migration.md) for native replacements.

## Limitations

- Official venue instructions can change and journal-specific rules can override a
  publisher profile. Always follow the linked official sources and profile caveats.
- Source date and file metadata cannot prove every visual property; unresolved checks
  remain visible as skipped checks.
- LaTeX is external and opt-in (`rp.use(..., latex=True)`). ResearchPlot never
  downloads fonts or templates.
- Automatic template ingestion, online profile updates, other plotting backends, and
  broader venue coverage are post-0.2 roadmap work.

Documentation lives in [`docs/`](docs/index.md). Contributions that add a profile
must include traceable official sources and tests. ResearchPlot is MIT licensed.
