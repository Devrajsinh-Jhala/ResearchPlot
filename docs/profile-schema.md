# Profile schema

Profile JSON is validated against the JSON Schema 2020-12 document bundled as
`researchplot/profile.schema.json`. The installed schema—not a copied example in these
docs—is authoritative for field names and constraints.

## Validate a file

```bash
researchplot profile validate path/to/profile.json
```

The validator rejects unknown schema versions, malformed coordinates, incompatible
operator/value/unit combinations, missing sources, references to unknown source IDs,
invalid applicability values, and other structural errors.

Python callers can load a local file or validate in-memory data without changing the
built-in registry:

```python
import researchplot as rp

profile = rp.load_profile("path/to/profile.json")
profile = rp.validate_profile_data(payload, filename="candidate.json")
schema = rp.profile_schema()  # defensive copy of the bundled schema
```

## Conceptual structure

```json
{
  "schema_version": 2,
  "id": "example-journal",
  "revision": "2026.08.0",
  "effective_date": "2026-08-01",
  "name": "Example Journal",
  "kind": "journal",
  "year": null,
  "aliases": ["example"],
  "scope": "Main-article figures for Example Journal.",
  "default_width": "single",
  "verified_on": "2026-08-01",
  "caveats": ["Special collections may provide additional instructions."],
  "sources": [
    {
      "id": "artwork-guide",
      "title": "Example Journal artwork guide",
      "url": "https://example.org/artwork-guide",
      "locator": "Figures > Size",
      "retrieved_on": "2026-08-01",
      "verified_on": "2026-08-01"
    }
  ],
  "rules": [
    {
      "id": "figure.width.single",
      "level": "required",
      "probe": "artifact.width_mm",
      "constraint": {
        "operator": "approx",
        "value": 89,
        "unit": "mm",
        "tolerance": 0.5
      },
      "applies_to": {"widths": ["single"]},
      "verification": "automated",
      "phases": ["live", "file"],
      "source_ids": ["artwork-guide"],
      "description": "Single-column figures must be 89 mm wide."
    }
  ]
}
```

A profile digest is calculated from canonical profile content. Authors do not choose a
digest that disagrees with the content.

## Rule anatomy

```json
{
  "id": "raster.minimum_resolution",
  "level": "required",
  "applies_to": {
    "roles": ["main"],
    "content_kinds": ["line_art"],
    "output_formats": ["tiff"]
  },
  "probe": "artifact.dpi",
  "constraint": {
    "operator": "gte",
    "value": 600,
    "unit": "dpi",
    "tolerance": null
  },
  "verification": "automated",
  "phases": ["file"],
  "source_ids": ["artwork-guide"],
  "description": "Line art must be exported at 600 DPI or higher."
}
```

### Applicability

Applicability selects rules by figure role, content kind, and output format. An omitted
dimension means that the rule is not narrowed by that dimension; it does not mean an
arbitrary default should be invented.

Public enum vocabularies include `FigureRole`, `ContentKind`, and `OutputFormat`.
Profile validation fails on unsupported values.

### Probe and constraint

The probe names a typed observation produced by an inspector. The constraint declares
an operator, expected value, and optional unit. `ConstraintOperator` supports only
combinations meaningful for that observation type; for example, a numeric `gte`
constraint cannot compare a list of file formats.

Rules should express venue evidence, not contain executable code.

### Rule level

- `required` comes from explicit official requirements;
- `recommended` comes from explicit preferences or recommendations;
- `inferred` is maintainer-derived and must be labeled as such.

Do not convert absent guidance into a required or recommended rule.

### Verification mode

- `automated`: an available inspector can establish the probe;
- `manual`: human evidence or an attestation is required;
- `unsupported`: the rule is recorded but no reliable check exists.

The mode controls reporting honesty. It is not a promise that every artifact format
exposes the same observation.

## Sources

A source record should identify:

- a stable source ID used by rules;
- the official title and canonical URL;
- the relevant section heading, page, anchor, or template file;
- the date maintainers verified it;
- optional notes needed to interpret scope.

Keep quoted publisher text short. Rules should encode a precise constraint and use the
locator to make the interpretation reviewable.

## Revisions and immutability

Use a new revision whenever a released rule, source interpretation, or behaviorally
meaningful caveat changes. Never replace a published coordinate in place. Typographical
documentation changes that do not affect packaged profile evidence can ship with the
next package release without creating a false venue revision.

Before proposing a new revision:

```bash
researchplot profile validate profile.json
researchplot profile diff current.json profile.json
```

Tests must include resolution, coordinate/digest stability, applicability, passing and
failing examples, unsupported observations, source references, and package inclusion.

## External profile packs

ResearchPlot discovers installed packs from the `researchplot.profiles` entry-point
group without network access. A pack can expose one profile mapping, a validated
`VenueProfile`, a JSON file or directory path, an iterable of those values, or a
zero-argument callable returning one of them:

```toml
[project.entry-points."researchplot.profiles"]
my_lab = "my_researchplot_profiles:profiles"
```

Coordinates must be unique, and normalized IDs, names, and aliases cannot collide with
another installed or bundled profile. A broken or conflicting installed pack is skipped
with a runtime warning so bundled profiles remain available. Profile-pack entry points
execute trusted local Python code; install only packs you trust. Run `researchplot
profile validate` on every JSON profile and see [CONTRIBUTING](contributing.md) before
proposing a built-in profile.
