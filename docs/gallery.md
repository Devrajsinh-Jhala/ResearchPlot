# Workflow gallery

ResearchPlot stays close to Matplotlib. These examples demonstrate evidence workflows,
not another plotting grammar.

## Venue-sized line figure

```python
import matplotlib.pyplot as plt
import researchplot as rp

project = rp.Project.load("researchplot.toml")
figure = project.figure("figure-1")

with figure.style(deliverable="main") as style:
    fig, ax = style.subplots(aspect=0.68)
    x = [0, 1, 2, 3, 4]
    ax.plot(x, [v**2 for v in x], marker="o", label="Quadratic")
    ax.plot(x, [2 * v + 1 for v in x], marker="s", linestyle="--", label="Linear")
    ax.set(xlabel="Input", ylabel="Response")
    ax.legend(frameon=False)
    report = figure.check(fig=fig)
    result = figure.export(fig, policy="violations")

print(report.verdict, result.paths)
plt.close(fig)
```

Markers and line styles reinforce color distinctions.

## Existing Seaborn analysis

```python
import seaborn as sns

figure = project.figure("figure-2")
with figure.style(deliverable="main") as style:
    fig, ax = style.subplots(aspect=0.48)
    sns.boxplot(data=frame, x="system", y="score", hue="split", ax=ax)
    ax.set(xlabel="System", ylabel="Score")
    report = figure.check(fig=fig)
```

Seaborn is not a base dependency. ResearchPlot's `[plots]` extra is needed only for the
deprecated wrapper layer, not for normal Seaborn usage.

## Designer-produced SVG

```bash
researchplot audit designer-output/figure3.svg \
  --profile nature@2026.08.0 \
  --role main \
  --width single \
  --content combination \
  --format html \
  --output build/figure3.html
```

Only file-phase rules can be established. The project report remains indeterminate
when required live or bundle evidence is missing.

## Accessibility review images

```python
preview = rp.render_accessibility_previews("figures/figure1.png")
paths = rp.write_accessibility_previews(
    preview,
    "build/previews",
    stem="figure1",
)
diagnostics = rp.diagnose_visual("figures/figure1.png")
```

The preview variants and confidence-labeled visual diagnostics are review aids, not
automatic venue passes. Live Matplotlib inputs also enable rendered clipping, overlap,
whitespace, point-size, and colormap-luminance prompts.

## Deterministic bundle archive

```python
bundle = project.bundle("dist/submission")
verification = rp.verify_manifest(bundle.path)
assert verification.valid

archive = rp.create_deterministic_archive(
    bundle.path,
    "dist/submission.zip",
)
assert rp.verify_deterministic_archive(archive.path).valid
```

## Existing executable examples

The repository's older `examples/` files demonstrate the v1-compatible `Target` and
`Submission` APIs. They remain valid during 2.x but are not schema-v3 project examples
unless explicitly labeled. Documentation snippets are tested separately from image
pixels; physical metadata and artist structure are the compliance truth.
