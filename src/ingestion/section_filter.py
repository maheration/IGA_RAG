"""
Config-driven section exclusion for extracted PDF text.

Public API
----------
filter_excluded_sections(text, exclude_headers) -> str
    Drops any section whose header matches an entry in the exclusion list.

Header detection — two tiers
-----------------------------
Tier 1 (strict): A line qualifies as a strict header if it consists
  only of uppercase letters and spaces and is ≤ 40 characters.
  Used to TRIGGER exclusion even when no blank line precedes the header —
  PDFs often render bold section headers without extra vertical whitespace.
  Examples: "DISCLOSURE", "REFERENCES", "CONFLICT OF INTEREST"

Tier 2 (lenient): A line qualifies as a lenient header if it is ≤ 80
  characters, not purely numeric, and either ALL CAPS or Title Cased without
  terminal punctuation AND is preceded by a blank line.
  Used to CANCEL exclusion when a new, non-excluded section begins.
  Examples: "Treating IgAN", "CONCLUSION"

This two-tier approach avoids false positives (body text words that happen
to be all-caps, e.g. abbreviations) while correctly catching tightly-packed
section headers that lack preceding whitespace.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def filter_excluded_sections(text: str, exclude_headers: list[str]) -> str:
    """
    Remove content belonging to sections whose headers appear in the
    exclusion list.

    Args:
        text: full extracted text of a document (may span multiple pages)
        exclude_headers: strings to match against section headers, e.g.
                         ["REFERENCES", "DISCLOSURE"].  Matching is
                         case-insensitive and substring-based so that
                         "DISCLOSURE" also matches "AUTHOR DISCLOSURES".

    Returns:
        Cleaned text with excluded sections removed and excess blank lines
        collapsed.
    """
    normalized_excludes = [h.upper().strip() for h in exclude_headers]
    lines = text.splitlines()

    result: list[str] = []
    excluding = False
    prev_was_blank = True  # treat document start as following a blank line

    for line in lines:
        stripped = line.strip()

        # — Blank line —
        if not stripped:
            if not excluding:
                result.append(line)
            prev_was_blank = True
            continue

        upper = stripped.upper()

        # — Tier 1: strict header (ALL CAPS letters + spaces only, ≤ 40 chars) —
        # Triggers exclusion regardless of whether a blank line preceded.
        if _is_strict_header(stripped):
            if any(exc in upper for exc in normalized_excludes):
                if not excluding:
                    logger.debug("Excluding section (strict match): %r", stripped)
                excluding = True
                prev_was_blank = False
                continue

        # — Tier 2: lenient header (after blank line, ≤ 80 chars, all-caps or title) —
        # Cancels exclusion when a new non-excluded section begins.
        if prev_was_blank and _is_lenient_header(stripped):
            if not any(exc in upper for exc in normalized_excludes):
                excluding = False  # new, non-excluded section → resume inclusion
            else:
                excluding = True  # excluded section header after blank line
                logger.debug("Excluding section (lenient match): %r", stripped)
                prev_was_blank = False
                continue

        # — Output —
        if not excluding:
            result.append(line)

        prev_was_blank = False

    filtered = "\n".join(result)
    filtered = re.sub(r"\n{3,}", "\n\n", filtered)
    return filtered.strip()


def _is_strict_header(line: str) -> bool:
    """
    Strict header: every character is an uppercase letter or a space,
    and the line is ≤ 40 characters long.

    This catches tightly-packed PDF section headers such as "DISCLOSURE"
    or "CONFLICT OF INTEREST" that may not be preceded by a blank line.
    """
    return bool(line) and len(line) <= 40 and all(c.isupper() or c == " " for c in line)


def _is_lenient_header(line: str) -> bool:
    """
    Lenient header (requires a preceding blank line from the caller):
    ≤ 80 characters, not purely numeric, no bracket markup (e.g. [Page N]
    tags must not reset the excluding state), and either ALL CAPS or Title
    Cased without a terminal period.
    """
    if len(line) > 80:
        return False
    if line.isdigit():
        return False
    if "[" in line or "]" in line:  # exclude [Page N] tags and similar markup
        return False
    if line.isupper():
        return True
    if line.istitle() and not line.endswith("."):
        return True
    return False
