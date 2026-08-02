# Profiles and provenance

A profile is a versioned, immutable package of venue evidence. It contains aliases,
venue kind and year, applicable figure roles, physical widths, typography and export
constraints, official sources, verification dates, caveats, and a digest.

## Coordinates and resolution

The reproducible form is:

```text
<profile-id>@<revision>
```

Examples:

```text
nature@2026.08.0
ieee-journal@2026.08.0
cvpr-2026@2026.08.0
```

The revision identifies the packaged evidence, not the venue's publication year. A
year-pinned conference ID and a profile revision serve different purposes.

Resolution follows these rules:

1. An exact coordinate wins and is reproducible.
2. An unpinned exact ID or alias resolves to the current installed revision and warns.
3. A bare conference alias resolves to the newest bundled verified year and warns with
   its exact coordinate.
4. Ambiguous and unknown queries fail with close matches; they never become IEEE.

```python
import researchplot as rp

profile = rp.resolve_profile("nature@2026.08.0")
print(profile.coordinate)
print(profile.digest)
print(profile.sources)

for profile in rp.list_profiles():
    print(profile.coordinate)
```

The 0.2 `resolve_venue()` and `list_venues()` top-level names are not part of the 1.0
public API; use profile terminology.

## Initial catalog

All initial built-in profiles use revision `2026.08.0`.

| Profile ID | Scope | Key official source |
| --- | --- | --- |
| `ieee-journal` | General IEEE journal graphics | [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/) |
| `nature` | Flagship Nature main and related figure roles | [Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) |
| `elsevier-generic` | Generic Elsevier artwork; a journal may override it | [Elsevier artwork sizing](https://www.elsevier.com/en-au/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing) |
| `neurips-2026` | NeurIPS 2026 proceedings template | [Official formatting package](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip) |
| `icml-2026` | ICML 2026 proceedings | [ICML author instructions](https://icml.cc/Conferences/2026/AuthorInstructions) |
| `cvpr-2026` | CVPR 2026 proceedings | [CVPR author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| `acl-2026` | ACL 2026 proceedings | [ACL formatting requirements](https://github.com/acl-org/acl-style-files/blob/master/formatting.md) |
| `plos-biology` | PLOS Biology figure files and submission metadata | [PLOS figure requirements](https://journals.plos.org/plosbiology/s/figures) |
| `acm-acmart` | Bundle-only ACM `acmart` figure descriptions; physical widths are intentionally unspecified | [ACM submission template](https://www.acm.org/binaries/content/assets/publications/taps/acm_layout_submission_template.pdf) |

This table is a discovery aid, not the source of truth. Run `profile show` to see the
exact URLs, section or page locators, rule levels, and caveats installed with your
package:

```bash
researchplot profile show nature@2026.08.0
researchplot profile show nature@2026.08.0 --json
```

### Named physical widths

| Profile | Width names and millimetres |
| --- | --- |
| `ieee-journal` | `single` 88.9; `double` 181.864 |
| `nature` | `single` 89; `double` 183 |
| `elsevier-generic` | `minimal` 30; `single` 90; `one-and-half` 140; `double` 190 |
| `neurips-2026` | `full` 139.7 |
| `icml-2026` | `single` 82.55; `double` 171.45 |
| `cvpr-2026` | `single` 83.34375; `double` 174.625 |
| `acl-2026` | `single` 77; `double` 160 |
| `plos-biology` | `text-column` 132; `full` 190.5; allowed main-figure range 66.8–190.5 |
| `acm-acmart` | No generic physical width; bundle metadata only |

Values are final physical widths. Rule level can differ: for example, PLOS Biology's
132 mm text-column width is recommended while its allowed range and full width carry
their own constraints. Inspect the profile rather than treating this table as a complete
rule set.

`acm-acmart` is intentionally bundle-only: ACM publication variants have different
page geometries, so the generic profile validates description metadata but cannot
create a venue-sized style. Combine its requirements with the specific conference or
journal instructions when preparing the figure itself.

```python
submission = rp.Submission("acm-acmart@2026.08.0", output_dir="acm-bundle")
submission.add(
    "figure1.pdf",
    "figures/figure1.pdf",
    alt_text="A line chart showing error decreasing as the sample size grows.",
    attestations={
        "metadata.alt_text.distinct_from_caption": (
            "The description conveys the trend that is not stated in the caption."
        )
    },
)
bundle = submission.build()
```

## Profile contents

Profiles conform to the bundled
[`profile.schema.json`](profile-schema.md) using JSON Schema 2020-12. The v2 model
includes:

- immutable `id`, `revision`, and computed SHA-256 digest;
- effective and optional retirement dates;
- venue type, year, aliases, scope, and caveats;
- named figure widths and default target metadata;
- rules with applicability, probe, constraint, level, and verification mode;
- source records with URL, title, locator, and verification date.

Rules are data, not venue-specific Python branches. This lets the same inspector
evaluate a PDF rule from a bundled profile or an independently installed pack.

## Provenance and freshness

ResearchPlot distinguishes four dates:

- the venue year, where relevant;
- the official source's effective date, when stated;
- the date maintainers verified the source;
- the immutable ResearchPlot profile revision.

Publisher profiles older than the configured freshness threshold warn. A stale warning
does not modify the profile, access the network, or assert that the guidance changed.
Year-pinned conference profiles remain immutable even after a later conference appears.

Source URLs are exposed in findings; `profile show` exposes the complete source records
and locators. If a publisher page disappears, the profile remains resolvable but
maintainers can release a new revision with repaired provenance. Published revisions
are never edited in place.

## Profile locks

A project lock captures the coordinates and digests used by a configuration:

```bash
researchplot profile lock nature@2026.08.0
```

Commit the generated lock with the paper. It is a deterministic evidence snapshot for
review and external verification. The 1.0 `check` command does not consume it
automatically, so CI should also pin the package version.

Compare revisions before updating:

```bash
researchplot profile diff nature@2026.08.0 ieee-journal@2026.08.0
```

## Validate proposed profile data

Maintainers and profile authors can validate a local schema-v2 JSON file before it is
bundled:

```bash
researchplot profile validate path/to/my-profile.json
```

The 1.0 registry also discovers installed, offline profile packs through the
`researchplot.profiles` Python entry-point group. See [Profile schema](profile-schema.md)
for the pack contract and [Contributing](contributing.md) before proposing a built-in
profile.

## Legacy styles are not profiles

The old `science`, `cell`, `springer`, and `pnas` styling labels are intentionally not
promoted to verified profiles. They remain available only by pinning the frozen 0.2.1
package in a separate environment; they do not claim venue compliance.
