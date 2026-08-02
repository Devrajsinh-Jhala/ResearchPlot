"""Minimal venue-native ResearchPlot example."""

from pathlib import Path

import matplotlib.pyplot as plt

import researchplot as rp

with rp.use("CVPR 2026", width="single") as style:
    figure, axes = style.subplots()
    axes.plot([0, 1, 2, 3], [0, 1, 4, 9], marker="o")
    axes.set(xlabel="Input", ylabel="Output")
    print(style.validate(figure))
    style.export(figure, Path("build") / "quickstart", artwork="vector")

plt.close(figure)
