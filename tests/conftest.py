"""
Shared fixtures for integration tests.

Requires a running backend (DATABASE_URL, API keys in .env).
Tests hit the real FastAPI app via TestClient — no mocks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRecord:
    """Single timing record for one test."""
    test_name: str
    endpoint: str
    method: str
    duration_s: float
    status_code: int
    response_size_bytes: int
    passed: bool
    error: str | None = None


@dataclass
class BenchmarkCollector:
    """Collects timing data across all tests in a session."""
    records: list[BenchmarkRecord] = field(default_factory=list)

    def add(self, record: BenchmarkRecord) -> None:
        self.records.append(record)

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = [r for r in self.records if r.passed]
        failed = [r for r in self.records if not r.passed]
        durations = [r.duration_s for r in self.records]

        return {
            "total": len(self.records),
            "passed": len(passed),
            "failed": len(failed),
            "total_duration_s": round(sum(durations), 3),
            "avg_duration_s": round(sum(durations) / len(durations), 3),
            "slowest_s": round(max(durations), 3),
            "fastest_s": round(min(durations), 3),
            "slowest_test": max(self.records, key=lambda r: r.duration_s).test_name,
            "endpoints_tested": len({r.endpoint for r in self.records}),
        }

    def to_table(self) -> str:
        if not self.records:
            return "No benchmarks recorded."

        lines = []
        lines.append("")
        lines.append("=" * 90)
        lines.append("INTEGRATION TEST BENCHMARK REPORT")
        lines.append("=" * 90)
        lines.append(
            f"{'Test':<45} {'Method':<7} {'Status':<7} {'Time':>8} {'Size':>10} {'Result':<6}"
        )
        lines.append("-" * 90)

        for r in sorted(self.records, key=lambda x: x.duration_s, reverse=True):
            size_str = _format_bytes(r.response_size_bytes)
            result = "PASS" if r.passed else "FAIL"
            lines.append(
                f"{r.test_name[:44]:<45} {r.method:<7} {r.status_code:<7} "
                f"{r.duration_s:>7.3f}s {size_str:>10} {result:<6}"
            )

        lines.append("-" * 90)
        s = self.summary()
        lines.append(
            f"Total: {s['total']} tests | "
            f"Passed: {s['passed']} | Failed: {s['failed']} | "
            f"Duration: {s['total_duration_s']}s | "
            f"Avg: {s['avg_duration_s']}s"
        )
        lines.append(
            f"Slowest: {s['slowest_test']} ({s['slowest_s']}s) | "
            f"Fastest: {s['fastest_s']}s | "
            f"Endpoints: {s['endpoints_tested']}"
        )
        lines.append("=" * 90)
        return "\n".join(lines)


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


# Module-level singleton so pytest hooks can access it without fixture lookup
_COLLECTOR = BenchmarkCollector()

# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def benchmark_collector() -> BenchmarkCollector:
    return _COLLECTOR


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """FastAPI TestClient — no real HTTP, but exercises the full app stack."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Per-test benchmark fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_benchmark(request, benchmark_collector):
    """Automatically records timing for every test.

    Tests can optionally set `request.node.benchmark_meta` to attach
    endpoint/method metadata. The `bench` fixture does this automatically.
    """
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start

    meta = getattr(request.node, "benchmark_meta", None)
    if meta:
        benchmark_collector.add(BenchmarkRecord(
            test_name=request.node.name,
            endpoint=meta.get("endpoint", "unknown"),
            method=meta.get("method", "?"),
            duration_s=round(duration, 4),
            status_code=meta.get("status_code", 0),
            response_size_bytes=meta.get("response_size_bytes", 0),
            passed=not request.node.rep_call.failed if hasattr(request.node, "rep_call") else True,
        ))


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test result on the node so _auto_benchmark can read it."""
    import pluggy
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call":
        item.rep_call = rep


# ---------------------------------------------------------------------------
# bench() helper — wraps API calls with timing + metadata
# ---------------------------------------------------------------------------

class BenchmarkHelper:
    """Wraps TestClient calls with automatic timing and metadata capture."""

    def __init__(self, client: TestClient, node):
        self._client = client
        self._node = node

    def get(self, url: str, **kwargs) -> Any:
        return self._call("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Any:
        return self._call("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> Any:
        return self._call("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> Any:
        return self._call("DELETE", url, **kwargs)

    def _call(self, method: str, url: str, **kwargs):
        start = time.perf_counter()
        resp = getattr(self._client, method.lower())(url, **kwargs)
        duration = time.perf_counter() - start

        self._node.benchmark_meta = {
            "endpoint": url,
            "method": method,
            "status_code": resp.status_code,
            "response_size_bytes": len(resp.content),
            "duration_s": round(duration, 4),
        }
        return resp


@pytest.fixture
def bench(client, request) -> BenchmarkHelper:
    """Use `bench.get(...)` / `bench.post(...)` instead of `client.get(...)`
    to automatically capture timing and endpoint metadata."""
    return BenchmarkHelper(client, request.node)


# ---------------------------------------------------------------------------
# Session teardown — print report + write JSON
# ---------------------------------------------------------------------------

def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001
    """Print benchmark table and write results to JSON after all tests."""
    collector = _COLLECTOR

    if not collector.records:
        return

    # Print table to terminal
    print(collector.to_table())

    # Write JSON report
    report_dir = Path("tests/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"benchmark_{int(time.time())}.json"

    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": collector.summary(),
        "tests": [
            {
                "name": r.test_name,
                "endpoint": r.endpoint,
                "method": r.method,
                "duration_s": r.duration_s,
                "status_code": r.status_code,
                "response_size_bytes": r.response_size_bytes,
                "passed": r.passed,
                "error": r.error,
            }
            for r in collector.records
        ],
    }
    report_file.write_text(json.dumps(report_data, indent=2))
    print(f"\nBenchmark report written to: {report_file}")
