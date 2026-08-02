"""Generate a small Linux CI gallery for visual-drift review."""

from pathlib import Path

import matplotlib.pyplot as plt

import researchplot as rp

OUTPUT = Path("build/regression-gallery")
OUTPUT.mkdir(parents=True, exist_ok=True)

for coordinate, width in (
    ("ieee-journal@2026.08.0", "single"),
    ("nature@2026.08.0", "double"),
):
    target = rp.target(
        coordinate,
        role="main",
        width=width,
        content="line-art",
    )
    with target.style() as style:
        figure, axes = style.subplots(aspect=0.62)
        x = [0, 1, 2, 3, 4]
        axes.plot(x, [value**2 for value in x], marker="o", label="Quadratic")
        axes.plot(
            x,
            [2 * value + 1 for value in x],
            marker="s",
            linestyle="--",
            label="Linear",
        )
        axes.set(xlabel="Input", ylabel="Response")
        axes.legend(frameon=False)
        profile_id = coordinate.partition("@")[0]
        figure.savefig(OUTPUT / f"{profile_id}-{width}.png", dpi=150, bbox_inches=None)
        plt.close(figure)
