# Command line

The `researchplot` CLI supports profile discovery, artifact checking, bundle builds,
and rule explanation without importing Python in a manuscript project.

```bash
researchplot --help
```

## Profile commands

### List and search

```bash
researchplot profile list
researchplot profile list --kind conference
researchplot profile search nature
```

Use `--json` where offered for machine-readable discovery.

### Inspect provenance

```bash
researchplot profile show nature@2026.08.0
researchplot profile show nature@2026.08.0 --json
```

The output includes widths, supported figure roles, rules, rule levels, verification
modes, official sources, locators, verification dates, caveats, and the profile digest.

### Compare revisions

```bash
researchplot profile diff nature@2026.08.0 ieee-journal@2026.08.0
```

Diffs show rule changes between any two installed coordinates. Comparing two revisions
of the same profile is the normal upgrade workflow once both are installed. A diff does
not claim that one profile is suitable for a manuscript governed by another.

### Lock a project

```bash
researchplot profile lock nature@2026.08.0
researchplot profile lock nature@2026.08.0 --output researchplot.lock.json
```

The generated lock records the selected profile's exact coordinate, digest, and
sources. `check` does not enforce the lock automatically in 1.0.

### Validate profile JSON

```bash
researchplot profile validate path/to/profile.json
```

This validates the schema, vocabulary, unit and constraint compatibility, source
references, alias conflicts visible to the loader, and digest calculation. It does not
certify that the publisher source was interpreted correctly.

## Check artifacts

With a project configuration:

```bash
researchplot check --config researchplot.toml
```

Configured mode checks the `[[figures]]` entries in the file. Positional paths are for
direct mode and do not override configured entries.

For a standalone artifact:

```bash
researchplot check figure1.pdf \
  --profile nature@2026.08.0 \
  --role main \
  --width single \
  --content line-art
```

Output formats:

```bash
researchplot check --config researchplot.toml --format text
researchplot check --config researchplot.toml --format json
researchplot check --config researchplot.toml --format sarif
```

- `text` is concise plain text intended for people.
- `json` is the stable ResearchPlot report representation.
- `sarif` is SARIF 2.1.0 for compatible code-scanning systems.

Structured documents are written to standard output. Diagnostics use standard error.

## Build a bundle

```bash
researchplot bundle build --config researchplot.toml
```

The selected policy determines whether non-compliant or indeterminate output can be
committed. The bundle is fully staged before a final directory rename, and handled
errors clean the staging directory. A non-cooperating external writer racing for the
same destination remains outside that guarantee.

## Explain a rule

```bash
researchplot explain raster.resolution.range --profile plos-biology@2026.08.0
researchplot explain font.pdf.type3.prohibited --profile nature@2026.08.0
```

Explanation is grounded in installed profile data. It shows constraints, applicability,
source locators, and what ResearchPlot can or cannot inspect; it does not fetch or
summarize new web content at runtime.

## Inspect local capabilities

```bash
researchplot doctor --profile nature@2026.08.0
researchplot doctor --profile nature@2026.08.0 --json
```

`doctor` reports the Matplotlib version and backend, physical width options, relevant
installed fonts, and whether optional LaTeX is available. It does not install or
download a missing capability.

## Exit codes

| Code | Meaning |
| ---: | --- |
| `0` | The requested operation succeeded and all evaluated targets are compliant. |
| `1` | One or more required rules failed. |
| `2` | Invalid arguments/configuration/profile, unreadable input, or a missing required capability. |
| `3` | No required failure was observed, but at least one required check is unresolved. |

For discovery commands, successful output uses `0` and invalid input uses `2`.

## Reproducible automation

In CI:

- install a pinned `researchplot-venues` version;
- use pinned profile coordinates and commit the profile lock;
- preserve JSON/SARIF and the manifest as build artifacts;
- treat exit code `3` as a review requirement, not a pass;
- never scrape profile output to infer fields that are already available in JSON.
