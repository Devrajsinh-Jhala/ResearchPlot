"""Check source reachability and HTTPS redirects; this does not verify content drift."""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request

import researchplot as rp


def main() -> int:
    urls = sorted({source.url for profile in rp.list_profiles() for source in profile.sources})
    failures: list[str] = []
    for url in urls:
        request = urllib.request.Request(
            url,
            headers={
                "Range": "bytes=0-0",
                "User-Agent": "ResearchPlot/1.0 profile-source-health",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response.read(1)
                final_url = response.geturl()
                if urllib.parse.urlsplit(final_url).scheme.casefold() != "https":
                    failures.append(f"HTTPS downgrade to {final_url}: {url}")
                else:
                    suffix = f" -> {final_url}" if final_url != url else ""
                    print(f"OK   {response.status} {url}{suffix}")
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403, 405, 429}:
                print(f"WARN {exc.code} {url} (reachable but automated access is restricted)")
            else:
                failures.append(f"HTTP {exc.code}: {url}")
        except urllib.error.URLError as exc:
            failures.append(f"{exc.reason}: {url}")
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
