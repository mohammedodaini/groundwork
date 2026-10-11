"""End-to-end smoke test against a running instance.

    python scripts/smoke_test.py http://localhost:8000 [--api-key KEY]

Runs one research job through the real HTTP API and checks what a visitor
would: the job finishes, the report has claims, and its trace can be fetched.
CI runs it against the Docker image and the compose stack, the two deployment
paths the unit tests never touch. Standard library only, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

OBJECTIVE = "Find technical wholesalers in Gelderland that process orders manually."


def call(base: str, path: str, *, key: str | None, body: dict | None = None) -> tuple[int, object]:
    request = urllib.request.Request(
        base + path,
        method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"content-type": "application/json", **({"X-API-Key": key} if key else {})},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read() or b"null")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    base, key = args.base_url.rstrip("/"), args.api_key

    status, created = call(base, "/api/research", key=key, body={"objective": OBJECTIVE})
    if status != 202 or not isinstance(created, dict):
        print(f"FAIL: creating a job returned {status}")
        return 1
    job_id = created["job_id"]

    deadline = time.monotonic() + args.timeout
    while True:
        _, job = call(base, f"/api/research/{job_id}", key=key)
        assert isinstance(job, dict)
        if job["status"] not in ("PENDING", "RUNNING"):
            break
        if time.monotonic() > deadline:
            print(f"FAIL: job {job_id} still {job['status']} after {args.timeout:.0f}s")
            return 1
        time.sleep(0.25)

    result = job.get("result") or {}
    claims = [c for entity in result.get("entities", []) for c in entity.get("claims", [])]
    if job["status"] == "FAILED" or not claims:
        print(f"FAIL: status={job['status']} claims={len(claims)} error={job.get('error')}")
        return 1

    status, trace = call(base, f"/api/research/{job_id}/trace", key=key)
    if status != 200 or not trace:
        print(f"FAIL: trace returned {status}")
        return 1

    print(f"OK: job {job_id} finished as {job['status']} with {len(claims)} claims and a trace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
