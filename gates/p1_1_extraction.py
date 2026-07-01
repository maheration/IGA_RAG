"""
Phase 1.1 — Extraction quality gate.

Validates a cleaned text string produced by src.ingestion.extractor
*before* it is written to data/cleaned/.  If any hard-fail check fails,
GateReport.raise_if_failed() is called by the pipeline and the file is
never written.

Checks
------
FAIL  min_content_length      Text is suspiciously short (extraction silently failed).
FAIL  word_glue_ratio         Too many tokens are excessively long all-letter strings
                              with no internal spaces — a sign pdfplumber merged
                              adjacent words (e.g. "GlomerularDiseasesin2021").
FAIL  symbol_corruption       PDF font-encoding substitutions for ≥ / ≤ (rendered
                              as $ / #) survived the normalisation step.
WARN  short_line_ratio        Unusually many lines are very short (< 20 chars), which
                              often indicates residual column interleaving.
FAIL  excluded_sections_absent  A configured noise section (DISCLOSURE, REFERENCES …)
                              was not removed by the section filter.
FAIL  keyword_coverage        Fewer than the configured fraction of test-set keywords
                              appear in the cleaned text (signals content loss).

Thresholds are read from the ``quality_gate`` block in configs/extraction.yaml
so they can be tuned without touching code.

Run standalone against an already-written file:
    python -m gates.p1_1_extraction data/cleaned/my_document.txt
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

from gates.base import CheckResult, GateReport, Status

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
CONFIG_PATH   = PROJECT_ROOT / "configs" / "extraction.yaml"
TEST_SET_PATH = PROJECT_ROOT / "docs" / "test_set.jsonl"


# ── Public entry point ────────────────────────────────────────────────────────


def run(
    text: str,
    doc_name: str,
    config: dict,
    test_set_path: Path = TEST_SET_PATH,
) -> GateReport:
    """
    Execute all extraction quality checks and return a GateReport.

    Parameters
    ----------
    text:
        Cleaned text string from extract_document() — not yet written to disk.
    doc_name:
        Identifier shown in the report header (typically the PDF stem).
    config:
        Full config dict from extraction.yaml.
    test_set_path:
        Location of test_set.jsonl used for keyword coverage check.
    """
    cfg = config.get("quality_gate", {})
    report = GateReport(doc_name=doc_name, gate_name="p1_1_extraction")

    report.results.append(_check_min_content(text, cfg))
    report.results.append(_check_word_glue(text, cfg))
    report.results.append(_check_symbol_corruption(text))
    report.results.append(_check_short_lines(text, cfg))
    report.results.append(_check_excluded_sections(text, config))
    report.results.append(_check_keyword_coverage(text, cfg, test_set_path))

    report.print_summary()
    return report


# ── Individual check functions ────────────────────────────────────────────────


def _check_min_content(text: str, cfg: dict) -> CheckResult:
    """FAIL if the cleaned text is shorter than the minimum expected length."""
    threshold = cfg.get("min_content_chars", 5_000)
    n = len(text)
    return CheckResult(
        name="min_content_length",
        status=Status.PASS if n >= threshold else Status.FAIL,
        detail=f"{n:,} chars  (threshold ≥{threshold:,})",
    )


def _check_word_glue(text: str, cfg: dict) -> CheckResult:
    """
    FAIL if too many whitespace-separated tokens are suspiciously long
    all-letter strings.

    A 'glued' token is one that:
      - is longer than word_glue_min_len characters, AND
      - consists entirely of ASCII letters (no spaces, digits, or punctuation).

    These are a reliable proxy for words pdfplumber merged without spaces,
    which breaks embedding quality and keyword matching.
    """
    min_len   = cfg.get("word_glue_min_len", 25)
    max_ratio = cfg.get("word_glue_max_ratio", 0.03)
    pattern   = re.compile(rf"^[A-Za-z]{{{min_len},}}$")

    tokens = text.split()
    if not tokens:
        return CheckResult("word_glue_ratio", Status.FAIL, "text is empty")

    glued = [t for t in tokens if pattern.match(t)]
    ratio = len(glued) / len(tokens)
    return CheckResult(
        name="word_glue_ratio",
        status=Status.PASS if ratio <= max_ratio else Status.FAIL,
        detail=(
            f"{ratio:.2%} glued  "
            f"({len(glued)} / {len(tokens)} tokens ≥{min_len} letters;  "
            f"threshold ≤{max_ratio:.0%})"
        ),
    )


def _check_symbol_corruption(text: str) -> CheckResult:
    """
    FAIL if PDF font-encoding substitutions for mathematical symbols remain.

    Known corruption pattern for this document family:
      ≥  encoded as $  → regex  \\$\\d
      ≤  encoded as #  → regex  #\\d
    These should have been fixed by extractor.py's normalisation step.
    """
    patterns = {r"\$\d": "$ before digit", r"#\d": "# before digit"}
    found    = {desc for pat, desc in patterns.items() if re.search(pat, text)}
    return CheckResult(
        name="symbol_corruption",
        status=Status.PASS if not found else Status.FAIL,
        detail="none found" if not found else f"suspicious: {', '.join(sorted(found))}",
    )


def _check_short_lines(text: str, cfg: dict) -> CheckResult:
    """
    WARN (not FAIL) if an unusually high fraction of non-empty lines are
    very short.  Short lines are expected for captions, table rows, and
    page tags — this only becomes a concern at high ratios.
    """
    max_ratio   = cfg.get("short_line_max_ratio", 0.25)
    min_line_len = 20

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return CheckResult("short_line_ratio", Status.WARN, "no non-empty lines found")

    short = [ln for ln in lines if len(ln.strip()) < min_line_len]
    ratio = len(short) / len(lines)
    return CheckResult(
        name="short_line_ratio",
        status=Status.PASS if ratio <= max_ratio else Status.WARN,
        detail=(
            f"{ratio:.1%} short  "
            f"({len(short)} / {len(lines)} lines < {min_line_len} chars;  "
            f"threshold ≤{max_ratio:.0%})"
        ),
    )


def _check_excluded_sections(text: str, config: dict) -> CheckResult:
    """
    FAIL if any configured noise-section header still appears in the text
    *as a standalone section heading* (i.e. on its own line with nothing
    other than optional whitespace around it).

    Plain inline references such as "(Supplementary Table S1)" are ignored
    because they contain additional text after the keyword.  This prevents
    false positives while still catching a section that was not removed.
    """
    headers = config.get("extraction", {}).get("exclude_sections", [])
    still_present = []
    for h in headers:
        # Match the header only when it stands alone on a line — optional
        # trailing punctuation/space allowed, but no other words.
        pattern = re.compile(
            rf"(?:^|\n)\s*{re.escape(h.upper())}\s*[.:]?\s*(?:\n|$)",
            re.IGNORECASE | re.MULTILINE,
        )
        if pattern.search(text):
            still_present.append(h)

    return CheckResult(
        name="excluded_sections_absent",
        status=Status.PASS if not still_present else Status.FAIL,
        detail=(
            "all excluded sections removed"
            if not still_present
            else f"still present as section header: {', '.join(still_present)}"
        ),
    )


def _check_keyword_coverage(
    text: str,
    cfg: dict,
    test_set_path: Path,
) -> CheckResult:
    """
    FAIL if fewer than ``keyword_coverage_min`` of the test-set keywords
    appear in the cleaned text.

    Matching is:
      - Case-insensitive.
      - Hyphenated line-breaks are joined before matching (e.g. 'bio-\\n
        markers' → 'biomarkers') so soft-hyphens don't cause false misses.

    The keyword list is deduplicated across all test-set entries so each
    unique keyword counts once regardless of how many questions reference it.
    """
    min_coverage = cfg.get("keyword_coverage_min", 0.80)

    if not test_set_path.exists():
        return CheckResult(
            name="keyword_coverage",
            status=Status.WARN,
            detail=f"test_set not found at {test_set_path} — check skipped",
        )

    # Normalise the cleaned text once
    normalised = re.sub(r"-\n", "", text).lower()

    # Collect all keywords from the test set (deduplicated, order-preserved)
    seen: set[str] = set()
    keywords: list[str] = []
    with test_set_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            for kw in entry.get("keywords", []):
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    keywords.append(kw_lower)

    if not keywords:
        return CheckResult("keyword_coverage", Status.WARN, "no keywords found in test_set")

    found   = [kw for kw in keywords if kw in normalised]
    missing = [kw for kw in keywords if kw not in normalised]
    ratio   = len(found) / len(keywords)

    detail = f"{len(found)}/{len(keywords)} keywords found  ({ratio:.0%})"
    if missing:
        snippet = ", ".join(f'"{k}"' for k in missing[:5])
        if len(missing) > 5:
            snippet += f" … +{len(missing) - 5} more"
        detail += f";  missing: {snippet}"

    return CheckResult(
        name="keyword_coverage",
        status=Status.PASS if ratio >= min_coverage else Status.FAIL,
        detail=detail,
    )


# ── Standalone CLI ────────────────────────────────────────────────────────────


def main() -> None:
    """
    Validate an already-written cleaned file from the command line.

    Usage:
        python -m gates.p1_1_extraction data/cleaned/my_document.txt
    """
    if len(sys.argv) < 2:
        print("Usage: python -m gates.p1_1_extraction <cleaned_file.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    text   = path.read_text(encoding="utf-8")
    report = run(text, path.stem, config)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
