# Compiled manuscript PDF audit

ResearchPlot treats final PDF placement as separate evidence from a standalone figure.
It audits manuscript structure and conservatively matches configured figures to unique
measurable PDF image/Form XObject placements.

## Configure figures and hints

```toml
[manuscript]
path = "manuscript/paper.pdf"
format = "pdf"
required = true

[[manuscript.matching_hints]]
figure = "figure-1"
pages = [3]
number = 1
caption = "Response increases across measured inputs."
```

Hints are manual association evidence. A page/number/caption hint is accepted only when
it identifies exactly one top-level measurable graphical object on the matching page.
Text hints cannot distinguish a caption from an in-text reference, so that limitation
travels with the match.

## Run the audit

```bash
researchplot manuscript check --config researchplot.toml
researchplot manuscript check manuscript/paper.pdf --max-pages 500 --json
```

`--output FILE` writes JSON. A direct PDF path performs structural inspection only;
`--config` additionally supplies logical figures, deliverables, and matching hints.

```python
project = rp.Project.load("researchplot.toml")
audit = project.audit_manuscript(max_pages=500)

print(audit.page_count)
if audit.placement_audit is not None:
    for match in audit.placement_audit.matches:
        print(match.figure_id, match.status, match.method, match.detail)
```

## Evidence priority

ResearchPlot tries evidence in this order:

1. an embedded ResearchPlot figure/provenance ID on a PDF XObject or marked-content
   property;
2. an exact SHA-256 fingerprint of decoded raster pixels against an existing configured
   PNG/JPEG/TIFF deliverable;
3. configured page, figure-number, and caption hints plus a unique top-level image/Form
   XObject.

Strong ambiguous evidence is never replaced by a weaker guess. One placed object cannot
satisfy two configured figures. Status is `matched`, `ambiguous`, `missing`, or
`unmeasured`, and every result states its method, confidence, candidates, and
limitations.

The current implementation does **not** compute a canonical vector-object fingerprint
for a standalone PDF/SVG figure. Vector/Form placement can be associated by embedded
provenance or a unique configured hint.

## Measured placement data

For each image or Form XObject invocation, the audit can report:

- page and nested object name/type;
- bounding box in PDF points and millimetres;
- intrinsic placed width and height;
- object and page rotation;
- source raster pixel dimensions and effective X/Y DPI;
- page crop box and whether measured bounds extend outside it;
- source artifact dimensions and X/Y scale when a source can be inspected.

Nested Form XObjects, PDF graphics-state transforms, page object limits, and content
operation limits are handled without executing actions. Inline images are disclosed but
not fingerprinted.

Structural audit also records page geometry, resource/font counts, transparency,
annotations, and passive active-content indicators.

## Coverage and exit behavior

`ManuscriptPlacementAudit.coverage_complete` is true only when every configured figure
has one resolved placement. Missing, repeated, colliding, or unmeasured candidates stay
unresolved. The audit warns when resolved placements do not follow configured figure
order or configured figure numbers are duplicated.

`researchplot manuscript check` currently returns exit code `3` even when placement
coverage is complete. Placement measurement is implemented, but venue-specific
manuscript rules and integration of manuscript observations into the aggregate
`CompliancePlan` are not. A required manuscript therefore remains an explicit project
capability gap; the tool never presents placement coverage alone as venue compliance.

The `[manuscript]` extra is currently dependency-free; rendered fallback matching is
not implemented. Native LaTeX and DOCX source parsing remain outside the 2.0 scope.
