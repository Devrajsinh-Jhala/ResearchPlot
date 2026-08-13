# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [2.0.0] - 2026-08-13

### Added

- Strict schema-v3 `ProjectSpec`, `FigureSpec`, `PanelSpec`, `DeliverableSpec`, and
  `ManuscriptSpec` models with stable IDs, explicit deliverables, descriptions, panels,
  source-data references, typed manual attestations, digest/expiry-bound waivers, and a
  bundled project JSON Schema.
- Coverage-aware `CompliancePlan` and `PlanAssessment` results. A project becomes
  `COMPLIANT` only when every applicable encoded required rule is covered and passes;
  missing required evidence produces `INDETERMINATE`.
- Explicit `ExportPlan` separation between allowed formats, selected/preferred formats,
  required companions, and format-specific settings.
- Profile schema v3 with typed probes, closed unit conversion, declarative comparison
  expressions, `all`/`any`/`not` composition, governance/status metadata, deterministic
  composition, source fingerprints, and schema-v2 translation.
- Deterministic profile locks plus an opt-in fail-closed TUF registry client requiring
  an explicit base URL, trusted root, and cache. Normal operations remain offline.
- Deeper passive PDF, SVG, raster, and EPS observations, including active-content
  indicators, PDF resources, SVG scripting/external references, raster EXIF/ICC/frame
  metadata, and format/content mismatches.
- Bounded subprocess artifact inspection, manifest/path/hash verification, and
  deterministic ZIP/TAR creation and verification with stable members, timestamps,
  ownership, permissions, and `SOURCE_DATE_EPOCH` support.
- Stable transactional PDF/SVG metadata and profile-derived SVG identifier salts for
  reproducible vector exports under a fixed dependency environment.
- Deterministic grayscale, protanopia, deuteranopia, and tritanopia previews plus
  confidence-labeled advisory luminance, entropy, transparency, rendered clipping,
  label-overlap, whitespace, point-size, and colormap-luminance diagnostics.
- Compiled-manuscript PDF structure and conservative placement auditing via embedded
  provenance IDs, exact raster fingerprints, or unique configured hints, with measured
  bounds, rotation, effective DPI/source scale, and crop-box clipping.
- Deterministic remediation plans, self-contained offline HTML reports, JATS 1.4 figure
  metadata, and RO-Crate 1.3 metadata projections.
- Loopback-only local browser workspace with per-launch tokens, origin checks, upload
  limits, a restrictive content-security policy, temporary-file cleanup, local raster
  previews, JSON download, and reproducible CLI commands.
- CLI commands for project initialization/migration, direct artifact audit, frozen
  checks, HTML output, profile status/verify/sync, bundle verify/archive/JATS/RO-Crate,
  structural manuscript checks, read-only remediation planning, retarget planning,
  local serving, and capability diagnostics.
- A 22-profile source-backed launch catalog across journals, conferences, and
  generic/narrow publisher guidance, with evidence metadata and page-generation
  tooling. Missing official guidance remains unspecified.
- Dedicated security workflow, dependency updates, profile evidence generation,
  release SBOM/provenance generation, issue/PR templates, and CODEOWNERS metadata.

### Changed

- Make the project graph and its explicit evidence coverage the primary API while
  retaining phase-local `Target` operations.
- Require exact profile coordinates in schema-v3 projects and verify locks before
  frozen inspection.
- Emit a published aggregate report schema version 2 from `PlanAssessment.to_dict()`,
  including a plan digest, sources, coverage, capability gaps, remediations, and
  privacy-safe environment provenance.
- Treat external file-only passes as phase-limited evidence instead of project-wide
  compliance.
- Build project bundles as verified directories; deterministic ZIP or TAR creation is a
  separate explicit step.
- Make the public website and documentation artifact-audit-first and document
  security, offline behavior, interoperability, migration, and current capability
  limits alongside every workflow.
- Require Python 3.11 or newer. LaTeX, web, registry, manuscript fallback, legacy plot
  integrations, and network access remain optional.

### Deprecated

- `Target`, `target()`, v1 project/report/manifest surfaces, legacy CLI aliases, and the
  high-level plotting wrappers remain available throughout 2.x but are compatibility
  bridges. No removal occurs before 3.0.
- Legacy plotting helpers load lazily and require `researchplot-venues[plots]`; new
  authoring should use native Matplotlib inside a `FigureTarget.style()` context.

### Security

- Keep profile sync as the only intended network operation and fail closed when trust
  material or TUF capability is unavailable.
- Reject unsafe bundle/archive member paths, symlinks, non-regular members, case
  collisions, digest mismatch, and unbounded manifest structures.
- Add bounded isolated parsing for untrusted artifacts and report active PDF/SVG content
  without executing it.
- Harden local web processing to loopback-only sessions with request tokens, origin
  checks, upload limits, redacted temporary paths, and no telemetry.
- Use exclusive or atomic same-directory writes for new v2 report/configuration paths
  where implemented.

### Known limitations

- Manuscript matching does not canonicalize standalone vector objects, reconcile all
  captions/references, or evaluate venue-specific manuscript rules; the command remains
  indeterminate (exit code `3`) even when every configured placement is measured.
- `Project.bundle()` uses the v1 submission-manifest bridge and rejects generic
  attachments or multiple source-data files rather than losing their semantics.
- Byte-for-byte archive and vector-export reproducibility still depends on a fixed
  ResearchPlot, Matplotlib, font, and backend environment.
- Semantic object contrast, color-only encoding, panel alignment, and cryptographic
  provenance signing for reviewer attestations are not yet implemented.

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

[2.0.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.2.1...v1.0.0
[0.2.1]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.1.0...v0.2.0
