# Migrating from 0.1

The `0.1` functions still work in `0.2.x`, show figures by default, and accept their
original positional arguments. They now return `(fig, ax)` and accept `ax=` and
`show=` keyword arguments. Install them with `researchplot[plots]`.

```python
# 0.1 compatibility
fig, ax = researchplot.line(x, y, "Time", "Accuracy", format="ieee", show=False)

# 0.2 native workflow
with researchplot.use("ieee-journal", width="single") as style:
    fig, ax = style.subplots()
    ax.plot(x, y)
    ax.set(xlabel="Time", ylabel="Accuracy")
```

Key changes:

- Venue resolution is strict; unknown names no longer fall back to IEEE.
- Conference names resolve to the newest verified bundled year and warn. Pin the ID.
- `science`, `cell`, `springer`, and `pnas` are unverified legacy styles, not profiles.
- Seaborn/pandas/scikit-learn/SciPy helpers moved to the `plots` extra.
- LaTeX is disabled by default and remains external.

Compatibility wrappers remain for at least two `0.2.x` releases and will not be
removed before `1.0`.
