#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time

from urllib.request import urlopen
from urllib.error import URLError


def main() -> int:
    base_url = os.getenv("OPENRAG_API_URL", "http://localhost:8000")
    health = f"{base_url.rstrip('/')}/healthz"
    ready = f"{base_url.rstrip('/')}/healthz/ready"
    try:
        with urlopen(health, timeout=5) as r:
            print("/healthz:", r.read().decode("utf-8"))
        # Give the service a moment to finish boot
        time.sleep(0.5)
        with urlopen(ready, timeout=5) as r2:
            print("/healthz/ready:", r2.read().decode("utf-8"))
    except (URLError, Exception) as exc:
        print(f"Compose smoke failed: {exc}", file=sys.stderr)
        return 1
    print("Compose smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
