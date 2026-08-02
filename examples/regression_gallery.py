"""Generate a small Linux CI gallery for visual-drift review."""

from pathlib import Path

import matplotlib.pyplot as plt

import researchplot as rp

OUTPUT = Path("build/regression-gallery")
OUTPUT.mkdir(parents=True, exist_ok=True)

for venue, width in (("ieee-journal", "single"), ("nature", "double")):
    with rp.use(venue, width=width) as style:
        figure, axes = style.subplots()
        x = [0, 1, 2, 3, 4]
        axes.plot(x, [value**2 for value in x], marker="o", label="Quadratic")
        axes.plot(x, [2 * value + 1 for value in x], marker="s", linestyle="--", label="Linear")
        axes.set(xlabel="Input", ylabel="Response")
        axes.legend(frameon=False)
        figure.savefig(OUTPUT / f"{venue}-{width}.png", dpi=150, bbox_inches=None)
        plt.close(figure)
