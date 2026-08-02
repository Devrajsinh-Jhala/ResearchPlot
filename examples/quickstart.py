"""Create and post-audit one ResearchPlot 1.0 artifact."""

from pathlib import Path

import matplotlib.pyplot as plt

import researchplot as rp

OUTPUT = Path("build/examples/figure1.pdf")

target = rp.target(
    "nature@2026.08.0",
    role="main",
    width="single",
    content="line-art",
)

with target.style() as style:
    figure, axes = style.subplots(aspect=0.62)
    axes.plot([0, 1, 2, 3], [0, 1, 4, 9], marker="o")
    axes.set(xlabel="Input", ylabel="Response")
    result = target.export(figure, OUTPUT, policy="complete")

print(f"Verdict: {result.report.verdict.value}")
print(f"Artifact: {result.paths[0]}")
print(f"Manifest: {result.manifest_path}")
plt.close(figure)
