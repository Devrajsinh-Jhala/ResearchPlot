# Command-line reference

The CLI mirrors the Python evidence model and preserves stable exit codes. Run
`researchplot COMMAND --help` for the parser bundled with the installed release.

## Initialize a project

Discover one or more existing artifacts and write schema-v3 configuration:

```bash
researchplot init \
  --profile nature@2026.08.0 \
  --figure figures/figure1.pdf \
  --figure figures/figure2.pdf \
  --width single \
  --role main \
  --content line-art \
  --output researchplot.toml \
  --lock researchplot.lock.json
```

`--force` is required to replace a generated output. Add `--json` for a
machine-readable result. Initialization does not invent captions, descriptions, source
data, or scientific metadata; review and complete the file.

## Migrate a v1 configuration

```bash
researchplot migrate \
  --config researchplot.toml \
  --output researchplot.v3.toml \
  --lock researchplot.lock.json
```

The default output is separate from the source. `--force` allows replacement of the
chosen output; retain a backup and review the translation. `--json` reports the files
written.

## Audit saved artifacts

```bash
researchplot audit figures/figure1.pdf figures/figure2.svg \
  --profile nature@2026.08.0 \
  --width single \
  --role main \
  --content line-art
```

`--artwork` is an alias for `--content`. Direct frozen audits accept
`--frozen --lock researchplot.lock.json`.

Render the same results as JSON, SARIF, or self-contained HTML:

```bash
researchplot audit figures/figure1.pdf --profile nature@2026.08.0 \
  --format json --output build/audit.json

researchplot audit figures/figure1.pdf --profile nature@2026.08.0 \
  --format sarif --output build/audit.sarif

researchplot audit figures/figure1.pdf --profile nature@2026.08.0 \
  --format html --output build/audit.html
```

`--json`, `--sarif`, and `--html [FILE]` are convenience forms. Do not combine
conflicting report selectors.

## Check a project

```bash
researchplot check --config researchplot.toml --frozen
researchplot check --config researchplot.toml \
  --format json --output build/report.json
```

Configuration mode is mutually exclusive with positional paths and direct `--profile`
target options. The CLI does not silently select a nearby configuration file. Direct
legacy-style checking remains an alias for artifact auditing:

```bash
researchplot check figure.pdf \
  --profile nature@2026.08.0 \
  --role main \
  --width single \
  --content line-art
```

## Browse profiles

The singular `profile` command is canonical; `profiles` and `venues` are compatibility
aliases.

```bash
researchplot profile list
researchplot profile list --kind journal --year 2026 --json
researchplot profile search vision
researchplot profile show cvpr-2026@2026.08.0
researchplot profile diff nature@2026.08.0 ieee-journal@2026.08.0
researchplot profile validate profiles/candidate.json
researchplot profile status
```

`show` also accepts the compatibility alias `info`.

### Lock and verify

```bash
researchplot profile lock nature@2026.08.0 \
  --output researchplot.lock.json

researchplot profile verify \
  --lock researchplot.lock.json \
  --profile nature@2026.08.0
```

Use `--force` only when deliberately replacing a lock. Both commands support `--json`.

### Explicit signed sync

```bash
researchplot profile sync \
  --base-url https://profiles.example.org/ \
  --trusted-root keys/root.json \
  --cache-dir .researchplot/profiles \
  --coordinate nature@2026.08.0
```

The optional `--coordinate` fetches one pinned target. Sync requires
`researchplot-venues[registry]`, valid TUF metadata, and explicit trust material. It
never falls back to unsigned content. No other normal command contacts the registry.

## Explain a rule

```bash
researchplot explain figure.width.single \
  --profile nature@2026.08.0
```

`--json` returns the complete rule and its selected source records.

## Build and verify bundles

```bash
researchplot bundle build \
  --config researchplot.toml \
  --output dist/submission \
  --policy complete \
  --frozen

researchplot bundle verify dist/submission
researchplot bundle archive dist/submission dist/submission.zip
researchplot bundle verify dist/submission.zip
```

`bundle build` writes a directory. `archive` creates deterministic ZIP. Verification is
strict by default; `--manifest NAME` selects a non-default manifest and `--no-strict`
permits extra files. `--json` is available for automation.

Generate interoperability metadata:

```bash
researchplot bundle jats dist/submission \
  --output dist/figures.xml \
  --group-id researchplot-figures

researchplot bundle ro-crate dist/submission \
  --output dist/ro-crate-metadata.json \
  --name "Paper figure evidence" \
  --creator "A. Researcher"
```

`jats` accepts `JATS`; `ro-crate` accepts `rocrate` and `RO-Crate`. The latter also
accepts `--description`, `--license`, repeated `--creator`, and `--force`.

## Inspect a compiled manuscript

```bash
researchplot manuscript check --config researchplot.toml
researchplot manuscript check manuscript/paper.pdf --max-pages 500 --json
```

`--output FILE` writes JSON. Configuration mode also performs conservative placement
matching from provenance IDs, exact raster fingerprints, or unique configured hints.
The command deliberately returns exit code `3` because venue-specific manuscript rules
are not yet evaluated even when every configured placement is measured.

## Plan remediation

```bash
researchplot fix figures/figure1.pdf --plan
researchplot fix figures/figure1.pdf --plan \
  --format json --output build/fix-plan.json
```

`fix` is read-only in 2.0. It classifies deterministic human actions from measured
artifact facts; there is no mutation flag in the implemented command.

## Plan retargeting

```bash
researchplot project retarget \
  --config researchplot.toml \
  --profile ieee-journal@2026.08.0 \
  --plan
```

`--output FILE` writes the plan. `--apply` currently returns exit code `2` with an
actionable message because a comment-preserving transactional TOML rewriter is not yet
available. The planner never edits scientific data or claims an existing raster can be
losslessly restyled.

## Local browser workspace

```bash
researchplot serve
researchplot serve --port 8765 --no-browser
```

The server always binds to `127.0.0.1`. A random port is selected when `--port` is
omitted. `--no-browser` prints the tokenized URL without opening it.

## Doctor

```bash
researchplot doctor --profile nature@2026.08.0
researchplot doctor --profile nature@2026.08.0 --json
```

Doctor reports the profile digest, Matplotlib/backend, physical widths, requested and
installed fonts, optional LaTeX availability, sources, and caveats. It does not modify
the environment.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Compliant or successful non-compliance operation. |
| `1` | One or more required rules fail. |
| `2` | Invalid/unsafe input, unreadable file, parser/resource failure, or missing operational capability. |
| `3` | Required evidence is unavailable; verdict is indeterminate. |

Profile discovery, init/migrate, metadata conversion, and planning commands use `0`
for success and `2` for invalid input. Never merge `1`, `2`, and `3` into one generic
failure in CI; they need different remediation.
