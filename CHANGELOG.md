# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

## [0.2.0] - Unreleased

### Added

- Immutable, bundled, source-backed profiles for seven publication venues.
- Human-friendly year-aware resolution, reversible style contexts, live validation,
  strict multi-format export, offline file auditing, typed reports, and a CLI.
- `src/` packaging, typed marker, Python 3.10–3.14 metadata, test/lint/type/build CI,
  documentation, contribution and security policies, and citation metadata.

### Changed

- Optional Seaborn, pandas, scikit-learn, SciPy, and NumPy integrations now live in
  `researchplot[plots]`.
- Legacy plotting functions return figures/axes and emit deprecation warnings.

## [0.1.1] - Unreleased recovery release

### Fixed

- Restored `pairplot` and exported `stacked_bar`.
- Removed recursive metric and dendrogram calls; added missing metric integrations.
- Added histogram, contour, and font-scale defaults; removed duplicate line rendering.
- Corrected NumPy truth checks, array-like validation, pie counts, error messages,
  function metadata, and composable return values.
- Removed generated build products, metadata, bytecode, and figures from source.

[0.2.0]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/Devrajsinh-Jhala/ResearchPlot/compare/v0.1.0...v0.1.1
