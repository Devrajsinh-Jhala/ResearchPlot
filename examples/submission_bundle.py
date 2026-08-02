"""Build a two-figure submission bundle with descriptive metadata."""

from pathlib import Path

import matplotlib.pyplot as plt

import researchplot as rp

PROFILE = "nature@2026.08.0"
OUTPUT = Path("build/examples/submission")

single = rp.target(PROFILE, role="main", width="single", content="line-art")
with single.style() as style:
    figure1, axes1 = style.subplots(aspect=0.62)
    x = [0, 1, 2, 3]
    axes1.plot(x, [value**2 for value in x], marker="o")
    axes1.set(xlabel="Input", ylabel="Response")

double = rp.target(PROFILE, role="extended-data", width="double", content="combination")
with double.style() as style:
    figure2, axes2 = style.subplots(aspect=0.48)
    x = [0, 1, 2, 3]
    axes2.plot(x, [value + 1 for value in x], marker="o", label="Control")
    axes2.plot(x, [2 * value + 1 for value in x], marker="s", linestyle="--", label="Treatment")
    axes2.set(xlabel="Time", ylabel="Response")
    axes2.legend(frameon=False)

submission = rp.Submission(PROFILE, output_dir=OUTPUT, policy="complete")
submission.add(
    "figure1",
    figure1,
    role="main",
    width="single",
    content="line-art",
    formats=("pdf",),
    alt_text="A rising curve whose slope increases at each input.",
    source_data=Path("examples/data/figure1.csv"),
)
submission.add(
    "figure2",
    figure2,
    role="extended-data",
    width="double",
    content="combination",
    formats=("pdf",),
    alt_text="Treatment rises twice as quickly as the control across four time points.",
    source_data=Path("examples/data/figure2.csv"),
)

result = submission.build()
print(f"Compliant: {result.passed}")
print(f"Manifest: {result.manifest_path}")

plt.close(figure1)
plt.close(figure2)
