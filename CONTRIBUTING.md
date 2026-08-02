# Contributing

Thank you for improving ResearchPlot. Open an issue before a large behavioral change.

## Development

```bash
git clone https://github.com/Devrajsinh-Jhala/ResearchPlot.git
cd ResearchPlot
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,plots,docs]"  # Windows
python -m ruff check .
python -m mypy src
python -m pytest
python -m build
python -m twine check dist/*
```

Use the Agg backend in tests, close every figure, and verify artists or physical file
metadata instead of relying on cross-platform pixel snapshots.

## Venue profiles

Profiles must use official, stable sources and include a verification date, scope,
caveats, and a level for every rule. Do not infer absent requirements. Add resolver,
width, validation, audit, and package-data tests with every profile. Conference
profiles are year-pinned and immutable after release; corrections require changelog
entries and a new package version.

## Releases

Maintainers build and test both sdist and wheel. Production publishing runs only for
signed version tags through the protected `pypi` GitHub environment and PyPI Trusted
Publishing. Never upload artifacts built on a workstation.
