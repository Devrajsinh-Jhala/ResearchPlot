# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-08-02

### Added

- Immutable schema-v2 profile coordinates, digests, applicability constraints, source
  locators, verification modes, profile locks, comparison, and local profile
  validation.
- Target-oriented styling, validation, auditing, and transactional post-audited export.
- Tri-state `COMPLIANT`, `NON_COMPLIANT`, and `INDETERMINATE` reports with independent
  rule levels and check outcomes.
- Typed figure roles, content kinds, output formats, rule constraints, observations,
  export policies, results, and manifest records.
- Submission bundles with SHA-256 hashes, profile provenance, automated findings,
  manual attestations, captions, alt text, and source-data references.
- Project configuration through `researchplot.toml`, batch checking, profile locks,
  JSON reports, SARIF 2.1.0 output, rule explanation, and stable CLI exit codes.
- Expanded PDF, SVG, raster, and EPS inspection plus accessibility checks when evidence
  can be established reliably.
- Source-backed PLOS Biology and ACM `acmart` profiles alongside migrated revisions of
  the seven existing profiles.
- Offline third-party profile-pack discovery through the `researchplot.profiles`
  entry-point group, with collision and schema validation.
- Public report, single-export manifest, and submission-bundle JSON Schemas, a
  composite GitHub Action, pre-commit hook, and scheduled official-source health
  checks.
- Complete documentation for architecture, compliance semantics, profiles, project
  configuration, bundles, accessibility, serialization, migration, API, CLI, and
  limitations, with Mermaid diagrams and executable examples.

### Changed

- Replace `use(venue, ...)` as the primary workflow with
  `target(profile, role=..., width=..., content=...).style()`.
- Return `ExportResult` with paths, report, and manifest instead of a bare path list.
- Split the former `ArtworkType` into content and output-format concepts.
- Make exact `<profile-id>@<revision>` coordinates the reproducible profile identity;
  unpinned IDs and aliases warn with their resolved coordinate.
- Make strict complete export block unresolved required checks as well as known
  failures, then use a staged, rollback-capable commit for multi-file output.
- Focus the package on Matplotlib-native compliance rather than high-level plotting
  wrappers; legacy helpers remain available only in the separately pinned 0.2.1 release.
- Require Python 3.11 or newer for the 1.x series.
- Rename CLI discovery from `venues` to `profile` and replace one-file `audit` with the
  project-capable `check` workflow.

### Removed

- Binary `ValidationReport.passed` semantics that treated skipped required checks as
  successful.
- Silent figure-height clamping and target-width overrides through conflicting
  `figsize` values.
- The mixed artwork/output-format model from the primary API.
- Top-level legacy plotting helpers from the supported 1.0 API surface.

### Security

- Stage output privately, audit it before publication, and roll back handled commit
  failures. Abrupt process termination and non-cooperating writers remain outside the
  multi-file transaction guarantee.
- Avoid fetching SVG external resources or executing embedded artifact content during
  inspection.
- Reject non-regular overwrite targets, claim no-overwrite destinations exclusively,
  preflight portable bundle paths, bound profile/raster/PDF recursion inputs, and keep
  broken third-party profile packs isolated from bundled profiles.

## [0.2.1] - 2026-08-02

### Fixed

- Resolve `researchplot.__version__` from the renamed `researchplot-venues`
  distribution metadata.

## [0.2.0] - 2026-08-02

### Added

- Immutable, bundled, source-backed profiles for seven publication venues.
- Human-friendly year-aware resolution, reversible style contexts, live validation,
  strict multi-format export, offline file auditing, typed reports, and a CLI.
- `src/` packaging, typed marker, Python 3.10–3.14 metadata, test/lint/type/build CI,
  documentation, contribution and security policies, and citation metadata.

### Changed

- Optional Seaborn, pandas, scikit-learn, SciPy, and NumPy integrations now live in
  `researchplot-venues[plots]`.
- Legacy plotting functions return figures/axes and emit deprecation warnings.

### Fixed

- Restored `pairplot` and exported `stacked_bar`.
- Removed recursive metric and dendrogram calls; added missing metric integrations.
- Added histogram, contour, and font-scale defaults; removed duplicate line rendering.
- Corrected NumPy truth checks, array-like validation, pie counts, error messages,
  function metadata, and composable return values.
- Removed generated build products, metadata, bytecode, and figures from source.

[1.0.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.2.1...v1.0.0
[0.2.1]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.1.0...v0.2.0
