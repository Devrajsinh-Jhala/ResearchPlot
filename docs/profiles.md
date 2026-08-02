# Venue profiles and provenance

Profiles are immutable JSON resources bundled in the wheel. They include aliases,
venue kind and year, widths, typography, output rules, scope, caveats, official source
URLs, and verification dates.

| ID | Official source |
| --- | --- |
| `ieee-journal` | [IEEE resolution and size](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/) |
| `nature` | [Nature research figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/) |
| `elsevier-generic` | [Elsevier artwork sizing](https://www.elsevier.com/en-au/about/policies-and-standards/author/artwork-and-media-instructions/artwork-sizing) |
| `neurips-2026` | [NeurIPS 2026 template](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip) |
| `icml-2026` | [ICML 2026 instructions](https://icml.cc/Conferences/2026/AuthorInstructions) |
| `cvpr-2026` | [CVPR 2026 guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines) |
| `acl-2026` | [ACL formatting](https://github.com/acl-org/acl-style-files/blob/master/formatting.md) |

Run `researchplot venues info ID` for the complete bundled rule set. Publisher
profiles older than twelve months generate a freshness notice; year-pinned conference
profiles remain reproducible.
