# Gallery

ResearchPlot deliberately stays close to Matplotlib. These examples demonstrate
compliance workflows rather than introducing another plotting grammar.

## A single-column line figure

```python
import matplotlib.pyplot as plt
import researchplot as rp

target = rp.target(
    "ieee-journal@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)

with target.style() as style:
    fig, ax = style.subplots(aspect=0.68)
    x = [0, 1, 2, 3, 4]
    ax.plot(x, [v**2 for v in x], marker="o", label="Quadratic")
    ax.plot(x, [2 * v + 1 for v in x], marker="s", linestyle="--", label="Linear")
    ax.set(xlabel="Input", ylabel="Response")
    ax.legend(frameon=False)
    result = target.export(fig, "submission/figure1.pdf")

print(result.report.verdict)
plt.close(fig)
```

Markers and line styles provide redundant encoding in addition to color.

## Existing Seaborn code

ResearchPlot can style a normal Seaborn call because Seaborn draws into Matplotlib:

```python
import seaborn as sns

target = rp.target(
    "acl-2026@2026.08.0",
    role="main",
    width="double",
    content="combination",
)

with target.style() as style:
    fig, ax = style.subplots(aspect=0.48)
    sns.boxplot(data=frame, x="system", y="score", hue="split", ax=ax)
    ax.set(xlabel="System", ylabel="Score")
    report = target.validate(fig)
```

Seaborn is not a core dependency; install it in your own analysis environment. Legacy
plot wrappers are available only in a separate environment pinned to
`researchplot-venues[plots]==0.2.1`.

## Audit a designer-produced SVG

```python
target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="combination",
)

report = target.audit("designer-output/figure3.svg")
for check in report.findings:
    if check.outcome != rp.Outcome.PASS:
        print(check.rule_id, check.message)
```

This does not require the figure to originate in Matplotlib. Only live-artist checks
are unavailable; the SVG inspector still evaluates supported artifact properties.

## Batch configuration

The complete configuration example is available as
[`examples/researchplot.toml`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/researchplot.toml).

```bash
researchplot check --config examples/researchplot.toml
```

## Executable sources

- [`examples/quickstart.py`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/quickstart.py)
  is the minimal native workflow.
- [`examples/submission_bundle.py`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/submission_bundle.py)
  demonstrates two figures and manifest metadata.
- [`examples/gallery.ipynb`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/gallery.ipynb)
  is kept without output in version control.
- [`examples/regression_gallery.py`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/regression_gallery.py)
  generates a small Linux CI gallery for visual drift.

Correctness tests assert artist structure, rule observations, and physical artifact
metadata. Cross-platform pixels are not treated as compliance truth; the small
Linux-only regression gallery is a review aid.
