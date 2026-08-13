# Profile schema v3

Venue profiles are declarative JSON validated against the bundled JSON Schema 2020-12
contract. They contain evidence and rules, never Python or templates.

```bash
researchplot profile validate path/to/profile.json
```

In Python:

```python
schema = rp.profile_schema()
profile = rp.load_profile("profiles/example.json")
```

`profile_schema()` returns an independent copy so callers cannot mutate the installed
contract.

## Identity and governance

A schema-v3 profile can define:

| Field | Purpose |
| --- | --- |
| `id`, `namespace`, `revision` | Immutable coordinate. Revision uses `YYYY.MM.PATCH`. |
| `name`, `aliases`, `kind`, `year`, `scope` | Human resolution and applicability context. |
| `status` | `draft`, `verified`, `deprecated`, or `withdrawn`. |
| `maintainers`, `license`, `governance` | Ownership and review metadata. |
| `extends` | Optional base profile composition. |
| `effective_date`, `verified_on` | Evidence timing. |
| `default_width` | Named default when a venue defines physical widths. |
| `caveats` | Limits that must travel with reports. |
| `sources`, `rules` | Official evidence and declarative assertions. |

Some migrated profiles legitimately omit optional governance fields. Status and review
metadata describe the encoded document, not a guarantee that every venue requirement
has been discovered.

## Sources

A source record includes a stable ID, title, official URL, locator, retrieval and
verification dates, source kind/publisher, and optional archive URL or SHA-256 content
fingerprint.

Use the narrowest official page or template section that supports a rule. Do not cite a
generic publisher page as evidence for a journal-specific claim. Leave an optional
fingerprint `null` when it was not captured; never invent one.

## Rules

A traditional one-probe rule contains:

```json
{
  "id": "figure.width.single",
  "level": "required",
  "applies_to": {"widths": ["single"]},
  "probe": "artifact.width_mm",
  "constraint": {
    "operator": "approx",
    "value": 89,
    "unit": "mm",
    "tolerance": 0.5
  },
  "verification": "automated",
  "phases": ["live", "file"],
  "source_ids": ["nature-panels"],
  "description": "Nature single-column width."
}
```

Rule level is `required`, `recommended`, or `inferred`. Verification mode is
`automated`, `manual`, or `unsupported`. Phases are `live`, `file`, `bundle`, and
`manuscript`.

Applicability may constrain width, role, content, or output format. A rule that does not
apply is not emitted as a failure.

## Operators

The closed operator set is:

```text
eq, ne, gt, gte, lt, lte,
in, not_in, between, subset,
contains, not_contains,
approx, exists, pattern, required, prohibited
```

`approx` accepts a numeric tolerance. `pattern` performs a bounded full match (patterns
are limited to 256 characters and input to 4,096 characters). Profile validation
verifies value shape and unit dimension against the probe.

## Composition

Expression rules can compose typed comparisons with `all`, `any`, and `not`:

```json
{
  "kind": "all",
  "expressions": [
    {
      "kind": "comparison",
      "probe": "artifact.width_mm",
      "constraint": {"operator": "gte", "value": 80, "unit": "mm"}
    },
    {
      "kind": "comparison",
      "probe": "artifact.width_mm",
      "constraint": {"operator": "lte", "value": 90, "unit": "mm"}
    }
  ]
}
```

Expressions cannot invoke arbitrary code. Every comparison still uses a known probe,
compatible phase, closed operator, finite value, and compatible unit.

Collection observations additionally support bounded quantifier and aggregate nodes:

- `quantifier`: apply one constraint to `all` or `any` members;
- `aggregate`: compare the `count`, `minimum`, or `maximum` of members.

Empty quantifiers fail; empty minimum/maximum aggregates are unresolved. Minimum and
maximum require numeric members. These operators are useful only with a registered
collection-valued probe; profiles cannot create arbitrary observations.

## Probe catalog

```python
for probe in rp.list_probes():
    print(probe.id, probe.value_kind, probe.dimension, probe.phases)
```

The current catalog includes physical artifact dimensions, DPI/file size/page count,
format, live titles/fonts/lines/markers/color signals, PDF font resources, raster
mode/compression, and bundle caption/alt-text/source-data/filename metadata.

Deep inspectors collect additional passive facts—for example PDF active content and
raster frame count—but those facts are not automatically available to a profile rule
until a compatible public probe is registered. This prevents an inspector field from
silently becoming required compliance evidence.

## Unit catalog

```python
rp.convert_value(3.5, "in", "mm")
```

Supported units include `mm`, `cm`, `in`, `pt`, `px`, `dpi`, byte-size units, ratio,
percent, count, degrees, and bit depth. Conversion is allowed only within the same unit
dimension and rejects non-finite numbers.

## Composition and conflict handling

Profile composition resolves an optional base and overlay deterministically. An overlay
must explicitly supersede a conflicting base rule; accidental duplicate rule IDs or
incompatible identity cause validation failure. Profiles are content-digested after
normalization.

## Migration from schema v2

```python
translated = rp.translate_v2_profile(v2_payload)
```

The translator exists throughout 2.x so immutable v2 evidence remains readable. New
profile contributions should author v3 directly and include conformance fixtures for
every automated rule.

## Review checklist

- Cite only official, directly supporting sources.
- Use a precise section, page, or heading locator.
- Classify required versus recommended language conservatively.
- Leave missing guidance unspecified.
- Confirm every probe/operator/unit/phase combination validates.
- Add passing and failing conformance artifacts for automated rules.
- Record caveats and generic-publisher scope prominently.
- Obtain the repository's required profile review before marking `verified`.
