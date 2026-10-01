# Security policy

ResearchPlot maintenance ended on **2026-10-01**. **No release receives security
support**, and no future security fixes, report acknowledgements, or coordinated
disclosure response are promised. Version 2.0.1 is the final upstream release. See
[MAINTENANCE.md](MAINTENANCE.md) for preservation and fork guidance.

Do not publish sensitive files or exploit details in a public issue. GitHub's private
vulnerability reporting may be used where available, but this project does not promise
to monitor or respond to it. Report vulnerabilities in Pillow, pypdf, Matplotlib, or
another dependency through that dependency's own security process. A maintained fork
must provide its own security contact and response policy.

ResearchPlot processes local figure files. Treat untrusted PDF, image, SVG, and EPS
inputs as potentially hostile. Run isolated inspection with least privilege, review
dependency advisories, and test any dependency updates in your own environment. The
final package cannot guarantee compatibility with future dependency releases, and
bounded inspection is not a security sandbox.
