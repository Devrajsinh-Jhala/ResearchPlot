# Interoperability

ResearchPlot owns compliance evidence and artifact preflight, not plotting syntax. Any
tool that writes a supported PDF, SVG, EPS, PNG, JPEG, or TIFF can participate in the
file and bundle phases.

## Plain Matplotlib

```python
figure = project.figure("figure-1")

with figure.style(deliverable="main") as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(x, y, marker="o")
    report = figure.check(fig=fig)
```

This preserves live artist evidence and sets exact physical width through a reversible
`matplotlib.rc_context`.

## Seaborn and pandas

Seaborn and pandas draw into Matplotlib figures, so pass the project axes explicitly:

```python
import seaborn as sns

with project.figure("figure-1").style(deliverable="main") as style:
    fig, ax = style.subplots()
    sns.lineplot(data=data, x="time", y="response", ax=ax)
    report = project.figure("figure-1").check(fig=fig)
```

Seaborn and pandas are not base dependencies. Install them directly for native use or
install `[plots]` only for the deprecated ResearchPlot wrappers.

## SciencePlots, TUEPlots, and PlotStyle

Apply third-party rcParams inside or before the ResearchPlot style context, then
validate the resulting live figure. ResearchPlot's explicit user overrides take
precedence over profile defaults but remain subject to venue checks.

```python
with project.figure("figure-1").style(
    overrides={"axes.grid": True},
) as style:
    fig, ax = style.subplots()
```

A style package can improve appearance without proving venue compliance. Keep the
report as the evidence boundary.

## R, Julia, Plotly, browser tools, and design applications

Export the final artifact from the authoring tool, then audit it:

```bash
researchplot audit figure.pdf \
  --profile nature@2026.08.0 \
  --width single \
  --role main \
  --content line-art
```

File-only checks may be indeterminate when a requirement needs live or author evidence.
Add the artifact to schema-v3 configuration with caption, alt text, long description,
source data, panels, and attestations, then run a project check.

ResearchPlot does not rewrite arbitrary saved artifacts. Its remediation plan separates
measured problems that need re-export or author review; it never claims an existing
raster can be losslessly resized, recolored, or restyled.

## CI and external tooling

Use JSON for full evidence and SARIF for code-scanning annotations:

```bash
researchplot check --config researchplot.toml --frozen \
  --format json --output build/researchplot.json

researchplot check --config researchplot.toml --frozen \
  --format sarif --output build/researchplot.sarif
```

Preserve the native JSON report because SARIF severity alone cannot distinguish all
coverage and verdict semantics.

## Publication metadata

A verified submission manifest can be projected into JATS 1.4 figure markup and
RO-Crate 1.3 JSON-LD:

```bash
researchplot bundle jats dist/submission --output dist/figures.xml
researchplot bundle ro-crate dist/submission \
  --output dist/ro-crate-metadata.json
```

These converters use only supplied manifest metadata. They do not invent descriptions,
scientific relationships, or author identities.

## Capability boundary

| Evidence | Matplotlib | Seaborn/pandas | R/Julia/Plotly | Design tool |
| --- | :---: | :---: | :---: | :---: |
| Exact ResearchPlot style context | Yes | Yes, via Matplotlib axes | No | No |
| Live artist inspection | Yes | Yes | No native adapter | No native adapter |
| Saved-file audit | Yes | Yes | Yes | Yes |
| Project metadata/coverage | Yes | Yes | Yes | Yes |
| Bundle/hash verification | Yes | Yes | Yes | Yes |

This matrix describes technical capability, not a comparative quality claim about any
plotting library.
