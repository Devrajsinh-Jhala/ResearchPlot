# ResearchPlot

ResearchPlot is an offline, source-backed compliance layer for Matplotlib. It resolves
human venue names, creates figures at exact final dimensions, validates live artists,
exports allowed formats, and audits output metadata.

```python
import researchplot as rp

with rp.use("nature", width="single") as style:
    fig, ax = style.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    report = style.validate(fig)
    paths = style.export(fig, "result", artwork="vector")
```

Required rule violations block strict export. Recommendations warn, inferred guidance
is informational, and unprovable checks are skipped. Each result links back to its
official source.
