# Accessibility review

ResearchPlot supports deterministic accessibility evidence and author-controlled
descriptions. It does not use generative AI to write alt text or infer the scientific
meaning of a figure.

## Structured descriptions

Schema-v3 figures can record:

- a concise `alt_text`;
- a `long_description` for complex content;
- a caption;
- panel-level labels, short text, long descriptions, and source data;
- an optional accessible `data_table`;
- source data and code/data attachments.

```toml
[[figures]]
id = "figure-1"
caption = "Response increases across measured inputs."
alt_text = "Line chart rising monotonically from input zero to three."
long_description = "The plotted response values are zero, one, four, and nine."
data_table = "data/figure1-accessible.csv"

[[figures.panels]]
id = "panel-a"
label = "a"
alt_text = "Observed response by input."
source_data = ["data/figure1-a.csv"]
```

Presence can be checked only when a profile encodes the applicable metadata rule.
Quality and scientific completeness remain human judgments.

## Color-vision and grayscale previews

```python
import researchplot as rp

preview = rp.render_accessibility_previews("figures/figure1.png")
paths = rp.write_accessibility_previews(
    preview,
    "build/previews",
    stem="figure1",
)
```

The output includes original, grayscale, protanopia, deuteranopia, and tritanopia
screening images. Existing files are never overwritten. The simulations are review
approximations, not a diagnosis of any person's vision and not proof that every trace
is distinguishable.

Matplotlib figures and Pillow images can also be passed directly:

```python
preview = rp.render_accessibility_previews(fig, dpi=144)
```

## Advisory visual diagnostics

```python
diagnostics = rp.diagnose_visual("figures/figure1.png")

for finding in diagnostics.diagnostics:
    print(finding.code, finding.severity, finding.confidence, finding.message)
    print(*finding.limitations, sep="\n- ")
```

Every source receives global luminance-range, luminance-entropy,
near-black/near-white-fraction, and transparency diagnostics. These signals may reveal
a nearly empty render or prompt closer contrast review.

When the source is a live Matplotlib `Figure`, the rendered canvas additionally checks:

- visible text and legend bounds outside the figure canvas;
- substantial visible-text bounding-box overlaps;
- an advisory axes/canvas whitespace fraction;
- visible text below a generic 5 pt legibility prompt;
- sampled colormap luminance monotonicity and step-uniformity.

Every diagnostic carries numeric confidence and concrete limitations. Bounding boxes
cannot prove painted glyph collisions, a later tight-bbox save can change clipping,
intentional annotation can overlap, and diverging/cyclic/categorical colormaps need not
be monotonic.

The diagnostics do **not** segment meaningful graphical objects, perform OCR,
establish WCAG contrast for every label/trace, establish full perceptual uniformity,
determine whether color is the only scientific encoding, or align panel labels.

All visual diagnostics are advisory unless a verified profile explicitly maps a
compatible observation to a venue rule.

## Author review checklist

- View the figure at its final physical dimensions, not only enlarged on screen.
- Confirm that markers, line styles, shapes, labels, or position reinforce important
  color distinctions.
- Read the short text without seeing the figure; it should identify the visual and its
  essential result without duplicating a long caption verbatim.
- Provide a long description or accessible data table when the short alternative cannot
  convey relationships, panels, or trends.
- Check panel reading order, labels, units, legends, and abbreviations.
- Review the original and all preview variants; do not accept a simulation score as a
  replacement for human review.

## Standards and venue rules

ResearchPlot can use WCAG 2.2 and W3C complex-image guidance as general recommendations.
A check becomes a required venue rule only when a cited official venue source makes it
required. Generic accessibility advice never masquerades as publisher policy.

JATS export includes available captions and alt text from the current submission
manifest. The richer project model preserves long descriptions and panel metadata, but
the v1 bundle-manifest bridge cannot yet serialize every field into JATS.
