# Final release and maintenance status

ResearchPlot maintenance ended on **2026-10-01**. **2.0.1** is the final release of
`researchplot-venues`. The published distributions, documentation, signed release tag,
and MIT-licensed source remain available for existing users and forks.

No further feature, venue-profile, dependency, or security updates are planned.
Upstream issues and pull requests may not receive a response. There is no support or
security-response commitment. See the
[security policy](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/SECURITY.md).

## What the final release changes

Version 2.0.1 closes two project-verdict gaps: missing required deliverables or
referenced evidence cannot produce a passing project result, and a `complete`
`Project.bundle()` checks the coverage-aware project result before committing the
submission directory. The bundle gate includes staged copied or generated artifacts
and supplied live figures. Required failures use CLI exit code `1`; missing required
evidence uses `3`. Input and capability errors continue to use `2`.

The release does not complete deferred 2.0 roadmap work. The
[limitations guide](https://devrajsinh-jhala.github.io/ResearchPlot/limitations/)
documents the narrow submission-manifest bridge, conservative manuscript matching,
advisory visual diagnostics, and optional registry setup. A report establishes only
the encoded rules for the selected historical profile, not editorial acceptance or
the completeness of current venue guidance.

## Venue profiles are historical evidence

The bundled profiles retain their exact coordinates, digests, source URLs, and
verification dates. They are not being revalidated after maintenance ends. Compare
those records with the venue's current official author instructions before a new
submission.

Bare conference aliases such as `cvpr` continue to select an installed **2026** profile.
They do not advance to the current or next conference year. Year-pinned conference
profiles remain immutable and do not receive publisher-style age warnings. Publisher
freshness warnings are a prompt for review, not proof that a source is still correct.

The optional registry client requires a user-configured trusted root and repository.
No maintained public ResearchPlot registry or future profile update service is
promised. Normal audit, check, export, bundle, and local-web operations remain offline.

## Install and preserve a working environment

Use a virtual environment with a tested Python version. The final release was designed
for Python 3.11–3.14; compatibility with future Python or dependency releases is not
maintained.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux), then install the exact release:

```bash
python -m pip install researchplot-venues==2.0.1
researchplot --version
researchplot profile list
researchplot doctor --profile nature@2026.08.0 --json
```

For the local workspace, use
`python -m pip install "researchplot-venues[web]==2.0.1"`. Select other extras explicitly
before recording the environment. Run your actual audit/export workflow and inspect
its verdict and unresolved checks before treating the environment as usable.

The package pin alone leaves dependency versions open. Capture the environment and
download its install artifacts while they are available:

```bash
python -m pip freeze > requirements-lock.txt
python -m pip download -r requirements-lock.txt -d wheelhouse
```

Store the lock, wheelhouse, Python version, operating system, project configuration,
profile lock, source data, and reports with the research project. To reinstall the
captured packages into another compatible virtual environment without PyPI:

```bash
python -m pip install --no-index --find-links wheelhouse -r requirements-lock.txt
```

A freeze file and wheelhouse apply to the Python/platform environment in which they
were prepared; they do not guarantee installation on a different operating system or
Python version. Source distributions may also need build tools or dependencies. Use
`python -m pip download --only-binary=:all: -r requirements-lock.txt -d wheelhouse`
when a fully wheel-based capture is needed and all selected packages provide compatible
wheels. Check the resulting offline install before relying on it.

Archive the Matplotlib backend, fonts, LaTeX state if used, and exported files as well.
Matching package versions alone cannot guarantee identical rendering. Frozen
dependencies can contain vulnerabilities: review upstream advisories and assess any
updates in an independently maintained environment. Upstream ResearchPlot will not
provide compatibility or security fixes.

## Continue development in a fork

Fork the MIT-licensed repository, preserve license and attribution notices, and give
your fork an explicit maintenance and security policy. Follow
[the fork development guide](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/CONTRIBUTING.md)
for setup, tests, profile governance, and packaging.

A released profile coordinate is immutable. Revalidated or corrected evidence must
have a new revision and digest with traceable official sources. Keep uncertain rules
unspecified and preserve unresolved checks in reports. Choose new package/publication
identities for a separately maintained distribution; upstream PyPI ownership and
signing identities do not transfer with a fork.

Before enabling publication or scheduled workflows in a fork, configure its own
GitHub environments, Trusted Publisher identities, signing keys, Pages settings, and
dependency-update policy. The archived upstream workflows and documentation are
reference material, not an ongoing service.
