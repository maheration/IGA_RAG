"""
Shared base classes for all pipeline quality gates.

Every gate in this project follows the same contract:
  1. Run a set of named Checks against a pipeline artifact.
  2. Each Check produces a CheckResult with status PASS / WARN / FAIL.
  3. A GateReport collects all results and prints a formatted summary.
  4. If any FAIL is present, raise_if_failed() raises GateFailure —
     the caller must never write its output file after that.

Why WARN exists:
  Some signals (e.g. a high short-line ratio) are informational in context —
  figure captions and table rows legitimately produce short lines.  Warnings
  surface potential issues without blocking the pipeline.

Pattern used in every gate:
    report = some_gate.run(artifact, doc_name, config)
    report.raise_if_failed()      # raises before writing the output file
    write_output(artifact, ...)   # only reached if all checks passed/warned
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"

    def symbol(self) -> str:
        return {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[self.name]


@dataclass
class CheckResult:
    """Result of a single named check."""

    name: str
    status: Status
    detail: str  # one-liner: measurement + threshold, always human-readable


@dataclass
class GateReport:
    """Aggregated report for one gate run against one document."""

    doc_name: str
    gate_name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if no check has status FAIL (WARNs are allowed)."""
        return all(r.status != Status.FAIL for r in self.results)

    def print_summary(self) -> None:
        """Print a formatted report table to stdout."""
        width = 72
        print(f"\n{'─' * width}")
        print(f"  Gate : {self.gate_name}")
        print(f"  File : {self.doc_name}")
        print(f"{'─' * width}")
        for r in self.results:
            tag = f"[{r.status.name}]"
            print(f"  {r.status.symbol()}  {tag:<8}  {r.name:<32}  {r.detail}")
        print(f"{'─' * width}")
        verdict = "PASSED ✓" if self.passed else "FAILED ✗  — output file was NOT written"
        print(f"  Verdict: {verdict}")
        print(f"{'─' * width}\n")

    def raise_if_failed(self) -> None:
        """Raise GateFailure if any check has FAIL status."""
        if not self.passed:
            failing = [r.name for r in self.results if r.status == Status.FAIL]
            raise GateFailure(
                f"Gate '{self.gate_name}' FAILED for '{self.doc_name}'. "
                f"Failing checks: {', '.join(failing)}."
            )


class GateFailure(RuntimeError):
    """
    Raised when a quality gate rejects a pipeline artifact.

    Callers should catch this at the top-level runner so that one bad
    document does not abort processing of subsequent documents.
    """
