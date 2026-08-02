# Contributing

Thank you for improving ResearchPlot. Small fixes are welcome directly; open an issue
before a new profile, schema change, public API change, or large inspector.

## Development setup

ResearchPlot 1.x requires Python 3.11 or newer.

```bash
git clone https://github.com/Devrajsinh-Jhala/ResearchPlot.git
cd ResearchPlot
python -m venv .venv
```

Activate the environment, then install all development groups:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Run the same checks used for a release:

```bash
python -m ruff check .
python -m mypy src
python -m pytest
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

On Windows, quote the final `dist\*` argument or run the equivalent PowerShell file
expansion when required by your shell.

## Design principles

- Keep runtime profile resolution offline and deterministic.
- Separate declarative rules, typed observations, policy, and artifact writing.
- Never turn an unavailable check into a pass.
- Treat publisher statements as required or recommended only when the official source
  supports that classification; label maintainer inference explicitly.
- Keep Matplotlib global state reversible and LaTeX opt-in.
- Prefer direct Matplotlib composition over new plotting wrappers.
- Preserve reports, manifests, profile digests, and CLI exit-code contracts within a
  major version.

## Tests

Use Matplotlib's noninteractive Agg backend and close every figure. Assert artist
structure and physical artifact metadata instead of relying on cross-platform pixel
identity.

New core behavior needs passing, failing, skipped, malformed-input, and policy tests.
Export changes need transaction tests that prove no partial output remains after each
failure stage. File inspectors need deliberately malformed and adversarial fixtures as
well as valid PDF, SVG, PNG, JPEG, TIFF, or EPS samples.

The repository enforces at least 80% combined statement/branch coverage, with focused
adversarial tests expected for transaction, parser, verdict, and provenance paths.
Coverage is a signal, not a substitute for assertions about compliance semantics.

## Profile contributions

Profiles must use official, traceable sources. A proposal must include:

- an immutable `<profile-id>@<revision>` and valid schema-v2 JSON;
- scope, effective date where known, verification date, and caveats;
- a section, page, anchor, or template-file locator for every source;
- applicability, level, probe, constraint, verification mode, and source IDs for each
  rule;
- no invented value where official guidance is absent;
- resolver, alias-collision, applicability, digest, audit, and package-data tests;
- passing and deliberately failing examples for every automated required rule;
- documentation and changelog updates.

Run:

```bash
researchplot profile validate path/to/profile.json
researchplot profile diff old-profile.json path/to/profile.json
```

A released profile coordinate is immutable. A correction creates a new revision and
records why the evidence changed. A generic publisher profile must state that an
individual journal can override it.

Third-party profile packs use the `researchplot.profiles` entry-point group. Installed
packs execute as trusted Python code; local candidate JSON should still be checked with
`profile validate` before it is packaged or proposed as built-in evidence.

## Reports and file formats

Changes to `Report.to_dict()`, the manifest, or SARIF output require contract
tests and schema-version review. Do not infer the overall verdict from display text or
SARIF severity. Preserve the distinction between rule level, check outcome, and report
verdict.

Parsers must not execute embedded content or fetch external resources. Report
unavailable evidence as skipped and malformed input as an intentional error.

## Documentation

Examples must be executable against the documented public API and use exact profile
coordinates where reproducibility matters. Mermaid diagrams are supported through
MkDocs Material. Build documentation strictly before opening a pull request:

```bash
python -m mkdocs build --strict
```

Avoid promising acceptance or stating that metadata proves an unobservable property.

## Pull requests

Keep changes focused, update tests and changelog, and describe the official evidence or
behavioral contract affected. Do not commit build output, virtual environments, notebook
outputs, or generated submission artifacts.

## Releases

Maintainers build and test sdist and wheel in clean environments. Production publishing
runs only from a signed version tag through the protected `pypi` GitHub environment and
PyPI Trusted Publishing. Never upload a locally built artifact to production PyPI.

Release gates include lint, typing, branch coverage, strict docs, schema validation,
artifact fixtures, wheel/sdist installation without network or LaTeX, and a clean CLI
smoke test.
