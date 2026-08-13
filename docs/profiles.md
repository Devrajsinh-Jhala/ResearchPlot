# Profiles and provenance

A profile is an immutable set of venue evidence. It combines identity, scope, width
options, declarative rules, source records, caveats, and a digest; it does not contain
venue-specific executable code.

## Coordinates and digests

The reproducible coordinate is:

```text
<profile-id>@<revision>
```

Examples bundled in 2.0 include:

```text
nature@2026.08.0
cvpr-2026@2026.08.0
elsevier-generic@2026.08.0
```

The profile revision identifies the packaged evidence. It is different from a
conference year in the profile ID. A SHA-256 digest additionally covers the canonical
profile document.

Resolution rules are deterministic:

1. An exact coordinate wins.
2. An exact installed ID or alias resolves to its installed revision and warns.
3. A bare conference alias resolves only among installed profiles and warns with the
   selected coordinate.
4. Ambiguous or unknown input fails with close matches; it never falls back to IEEE.

```python
import researchplot as rp

profile = rp.resolve_profile("nature@2026.08.0")
print(profile.coordinate)
print(profile.digest)
print(profile.status)
print(profile.sources)
```

## Discover the installed catalog

ResearchPlot 2.0 ships 22 bundled profiles. The release documentation generator turns
the bundled data into evidence pages; the installed CLI remains the source of truth for
the exact profiles, rules, and revisions in your environment:

```bash
researchplot profile list
researchplot profile list --kind conference
researchplot profile search vision
researchplot profile show nature@2026.08.0
researchplot profile show nature@2026.08.0 --json
```

[Browse the generated profile evidence catalog](generated/profiles/index.md){ .md-button .md-button--primary }

The launch catalog includes journal, conference, and generic publisher evidence. A
generic publisher profile says exactly that: an individual journal can override it,
and ResearchPlot does not present generic guidance as journal-specific compliance.
Profiles awaiting sufficient official evidence should be `draft`, and missing rules
stay absent rather than inferred.

| Kind | Bundled profile IDs |
| --- | --- |
| Journals | `ieee-journal`, `jacs`, `jmlr`, `nature`, `physical-review-letters`, `plos-biology`, `tmlr` |
| Conferences | `aaai-2026`, `acl-2026`, `aistats-2026`, `colm-2026`, `cvpr-2026`, `eccv-2026`, `iclr-2026`, `icml-2026`, `neurips-2026`, `usenix-osdi-2026` |
| Publisher/generic or narrow publisher guidance | `aas-journals`, `acm-acmart`, `acs-generic`, `cell-graphical-abstract`, `elsevier-generic` |

Several new profiles are intentionally narrow—for example, a font-technology rule or
graphical-abstract-only scope. A `verified` status verifies the encoded evidence; it
does not imply that a one-rule profile covers the whole venue.

## Schema v3

Profile schema v3 supports:

- `id`, immutable revision, aliases, venue kind/year, status, scope, maintainers, and
  optional review/governance records;
- named physical widths and a default width;
- required, recommended, and inferred rules with applicability and evidence phases;
- typed probes, unit-safe constraints, declarative expressions, and verification mode;
- official sources with title, URL, section/page locator, retrieval and verification
  dates, optional archived URL, and optional content fingerprint;
- caveats, inheritance/composition metadata, and revision history.

Not every official page exposes every provenance field. Optional values such as a
source content fingerprint remain `null` when they were not established.

Expressions are data. They can perform supported comparisons and composition but
cannot import Python, interpolate templates, access the network, or execute a publisher
file. Profile validation checks probe/operator/unit/phase compatibility before use.

See [Profile schema](profile-schema.md) for authoring details.

## Rule provenance

A report can connect a finding to:

- exact profile coordinate and digest;
- rule ID, level, applicability, phase, probe, constraint, and verification mode;
- official source ID, URL, locator, dates, and any interpretation/rationale;
- profile scope and caveats.

Inspect those records before treating a check as authoritative. A verified profile
means its encoded evidence passed project governance; it does not guarantee that the
venue has no other requirements.

## Profile locks

Write and commit a lock beside the project configuration:

```bash
researchplot profile lock nature@2026.08.0 --output researchplot.lock.json
```

The lock records schema version, coordinate, profile digest, document digest, and
available source-content digests. A frozen plan verifies it before artifact parsing:

```python
project = rp.Project.load("researchplot.toml")
plan = project.plan(frozen=True)
report = plan.check()
```

A lock does not copy an entire official web page and does not prove that a URL remains
online. It establishes the exact installed evidence used for a run.

## Compare and validate

```bash
researchplot profile diff nature@2026.08.0 ieee-journal@2026.08.0
researchplot profile validate path/to/profile.json
```

`diff` compares immutable rule documents. Review required-rule changes separately from
recommendation or evidence-only updates when deciding whether to adopt a new revision.

## Signed registry

The optional registry client uses The Update Framework (TUF). It requires an explicit
repository URL, trusted root, and cache location; it fails closed if the dependency or
trust material is missing.

Normal profile resolution, checking, styling, export, bundle operations, and the local
workspace never sync. Only an explicit sync operation may contact a configured
registry. Downloaded profiles remain data-only and locks prevent silent changes.

ResearchPlot ships the client machinery, but the public registry's production URL,
root keys, rotation ceremony, publication approvals, and availability are operational
deployment concerns. See [Signed profile registry](profile-registry.md).

## Third-party and legacy profiles

Unsigned local JSON is useful for authoring, but must be labeled unverified and should
not enter frozen CI without an explicit trust decision. Legacy executable
`researchplot.profiles` entry points are disabled by default because importing a Python
plugin crosses a different security boundary and could shadow trusted data.

The schema-v2 translator remains available throughout 2.x for migration. New profiles
should target v3 directly.
