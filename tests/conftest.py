from __future__ import annotations

import matplotlib
import pytest
from matplotlib import pyplot as plt

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def close_figures() -> None:
    yield
    plt.close("all")
