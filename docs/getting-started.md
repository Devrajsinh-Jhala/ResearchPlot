# Getting started

This guide starts with an existing artifact, then connects it to a strict project and a
live Matplotlib figure.

## 1. Install

Create an isolated Python 3.11+ environment:

=== "Windows PowerShell"

    ```powershell
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install researchplot-venues
    ```

=== "macOS and Linux"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install researchplot-venues
    ```

Confirm the distribution, import, command, and bundled catalog:

```bash
python -c "import researchplot as rp; print(rp.__version__)"
researchplot --version
researchplot profile list
```

The base install works offline, does not download fonts, and does not require LaTeX.

## 2. Audit an existing file

```bash
researchplot audit figures/figure1.pdf \
  --profile nature@2026.08.0 \
  --role main \
  --width single \
  --content line-art
```

Use an exact `<profile-id>@<revision>` coordinate in a paper repository. Friendly
aliases are useful for discovery but warn with the installed revision they select.

The equivalent Python API is:

```python
import researchplot as rp

target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)
report = target.audit("figures/figure1.pdf")

print(report.verdict)
for finding in report.findings:
    print(finding.rule_id, finding.outcome, finding.message)
```

This is intentionally a phase-local report. A saved PDF may establish width and font
embedding but cannot necessarily establish every live-artistry or bundle requirement.
Use a project plan for a coverage-aware result.

## 3. Describe the project

Create `researchplot.toml`:

```toml
schema_version = 3
profile = "nature@2026.08.0"
policy = "complete"
lock = "researchplot.lock.json"

[[figures]]
id = "figure-1"
number = 1
role = "main"
width = "single"
content = "line-art"
caption = "Response increases across the four measured inputs."
alt_text = "Line chart with a monotonic increase from input zero to three."
source_data = ["data/figure1.csv"]

[[figures.deliverables]]
id = "main"
format = "pdf"
path = "figures/figure1.pdf"
required = true
preferred = true
```

Write an exact profile lock and inspect the project:

```bash
researchplot profile lock nature@2026.08.0 --output researchplot.lock.json
researchplot check --config researchplot.toml --frozen
```

Schema v3 rejects unknown keys and empty projects. Paths must be project-relative:
absolute paths, parent traversal, and resolved symlink/junction escapes are rejected.
Bundle/archive verification separately rejects traversal and symbolic-link entries.

## 4. Read coverage in Python

```python
project = rp.Project.load("researchplot.toml")
plan = project.plan(frozen=True)
assessment = plan.check()

print(assessment.verdict)
for gap in assessment.unresolved:
    print(gap.requirement.rule_id, gap.status)
for capability in assessment.capability_gaps:
    print("capability:", capability)
```

The three verdicts are not Boolean aliases:

- `COMPLIANT`: every applicable required rule is covered and passes;
- `NON_COMPLIANT`: at least one required rule fails;
- `INDETERMINATE`: required evidence or a required capability is missing.

## 5. Create the figure at its final width

```python
import matplotlib.pyplot as plt

figure = project.figure("figure-1")

with figure.style(deliverable="main") as style:
    fig, ax = style.subplots(aspect=0.62)
    ax.plot(
        [0, 1, 2, 3],
        [0, 1, 4, 9],
        marker="o",
        label="Measured response",
    )
    ax.set(xlabel="Input", ylabel="Response")
    ax.legend(frameon=False)

    assessment = figure.check(fig=fig)
    result = figure.export(fig, policy="violations")

print(assessment.verdict)
print(result.paths)
print(result.manifest_path)
plt.close(fig)
```

`style.subplots()` creates the exact final width. Its `aspect` is height divided by
width. Styling uses `matplotlib.rc_context`, so global state is restored on normal and
exceptional exits. The export is staged, inspects the generated file, and commits only
when the selected policy permits it.

## 6. Open the local workspace

```bash
python -m pip install "researchplot-venues[web]"
researchplot serve
```

The workspace runs on `127.0.0.1` with a per-launch token. Drop in a saved figure,
choose an installed profile, review evidence, and download JSON without sending the
artifact to a remote service.

## Next steps

- Learn [coverage and policy semantics](compliance.md).
- Add panels, multiple deliverables, a manuscript, and metadata in
  [project configuration](configuration.md).
- Inspect [profile provenance and locks](profiles.md).
- Build and verify [a submission bundle](bundles.md).
- Review [limitations and security boundaries](limitations.md).
