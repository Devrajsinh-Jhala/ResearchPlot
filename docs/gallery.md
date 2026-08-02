# Gallery

The executable gallery is in
[`examples/gallery.ipynb`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/gallery.ipynb),
with a small
[`examples/quickstart.py`](https://github.com/Devrajsinh-Jhala/ResearchPlot/blob/main/examples/quickstart.py)
script. Notebook
outputs are cleared in version control so documentation builds remain deterministic.

Gallery correctness tests inspect Matplotlib artist structure and output metadata.
They deliberately avoid cross-platform pixel snapshots; a Linux-only image regression
job generates and uploads a small artifact gallery on every CI run for visual-drift
review without treating platform pixels as compliance truth.
