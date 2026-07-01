"""
Table detection, linearization, and column-aware text extraction.

Public API
----------
detect_column_boundary(page) -> float | None
    Returns the x-coordinate of the gutter between two text columns, or None.

get_page_content_blocks(page, figure_bboxes=None) -> list[tuple[float, str]]
    Returns (top_y, text) pairs for body text and linearized tables, in
    correct top-to-bottom reading order, with figure and table regions
    excluded from body text.

Design notes
------------
Text is reconstructed from word bounding boxes (page.extract_words()) rather
than page.extract_text().  This approach:
  - Fixes word-glue: each word is a discrete element joined with an explicit
    space, regardless of how the PDF stores character spacing.
  - Fixes figure-text leakage: words whose centres fall inside a figure or
    table bounding box are filtered out before reconstruction, so flowchart
    labels and table cell text never bleed into the body text.
  - Preserves column reading order: words are split by the detected column
    boundary and each column is reconstructed independently, so left-column
    text is always output before right-column text.
"""

from __future__ import annotations

from collections import Counter

import pdfplumber

Bbox = tuple[float, float, float, float]  # (x0, top, x1, bottom)

# ── Public API ────────────────────────────────────────────────────────────────


def detect_column_boundary(page: pdfplumber.page.Page) -> float | None:
    """
    Return the x-coordinate of the column gutter if the page is two-column,
    or None for single-column pages.

    Algorithm: find the dominant word-start x-position in the right half of
    the page — in a two-column layout this corresponds to the right column's
    left margin, and the boundary is set just to its left.
    """
    words = page.extract_words()
    if len(words) < 20:
        return None

    pw = page.width
    right_x0s = [w["x0"] for w in words if w["x0"] > pw * 0.45]

    if len(right_x0s) < 10:
        return None

    bins: Counter[int] = Counter(int(x // 10) * 10 for x in right_x0s)
    top_count = max(bins.values())

    if top_count < 15:
        return None

    left_count = sum(1 for w in words if w["x0"] < pw * 0.45)
    if left_count < 15:
        return None

    top_bins = [x for x, c in bins.items() if c >= top_count * 0.70]
    right_col_margin = float(min(top_bins))

    boundary = right_col_margin - 5.0
    if not (pw * 0.40 <= boundary <= pw * 0.65):
        return None

    return boundary


def get_page_content_blocks(
    page: pdfplumber.page.Page,
    figure_bboxes: list[Bbox] | None = None,
    header_strip_top: float = 0.0,
) -> list[tuple[float, str]]:
    """
    Extract all text content from a page as (top_y, text) blocks.

    Steps:
    1. Find tables; linearize each one as its own block.
    2. Collect all word bboxes; drop words inside figure or table regions.
    3. Detect two-column layout; split remaining words by column.
    4. Reconstruct each column's text in reading order via word bboxes.
    5. Offset right-column y-coordinates by page.height so right-column
       content always sorts after left-column content.

    Args:
        page: pdfplumber Page object.
        figure_bboxes: bounding boxes of detected figure regions on this page.
                       Words whose centres fall inside these bboxes are
                       excluded from body text so flowchart labels do not
                       bleed into the extracted prose.
    """
    tables = page.find_tables()
    table_bboxes: list[Bbox] = [t.bbox for t in tables]
    exclude_bboxes: list[Bbox] = table_bboxes + (figure_bboxes or [])

    # Filter body-text words: drop running headers, figure words, and table words
    all_words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
    body_words = [
        w for w in all_words
        if w["top"] >= header_strip_top          # strip running headers
        and not _word_in_any_bbox(w, exclude_bboxes)  # strip figure / table words
    ]

    col_boundary = detect_column_boundary(page)

    blocks: list[tuple[float, str]] = []

    if col_boundary is None:
        text = _words_to_text(body_words)
        if text.strip():
            blocks.append((0.0, text.strip()))
    else:
        left_words  = [w for w in body_words if w["x0"] < col_boundary]
        right_words = [w for w in body_words if w["x0"] >= col_boundary]

        left_text = _words_to_text(left_words)
        if left_text.strip():
            blocks.append((0.0, left_text.strip()))

        right_text = _words_to_text(right_words)
        if right_text.strip():
            # Offset by page.height so all right-column content sorts after
            # all left-column content when the caller merges by top_y.
            blocks.append((page.height, right_text.strip()))

    # Linearized table blocks — inserted at their natural top_y
    for table in sorted(tables, key=lambda t: t.bbox[1]):
        linearized = _linearize_table(table)
        if linearized:
            blocks.append((table.bbox[1], linearized))

    return sorted(blocks, key=lambda b: b[0])


# ── Word-based text reconstruction ────────────────────────────────────────────


def _words_to_text(words: list[dict]) -> str:
    """
    Reconstruct a text string from word bounding boxes in reading order.

    Words are sorted top-to-bottom and grouped into lines using a y-tolerance
    of 5 pts (handles minor character-height variation within a line).
    Within each line, words are sorted left-to-right and joined with spaces.

    This avoids the word-glue problem that arises from extract_text(), where
    PDFs that store characters without explicit space glyphs produce output
    like "GlomerularDiseasesin2021".
    """
    if not words:
        return ""

    LINE_Y_TOL = 5.0
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    lines: list[list[dict]] = []
    current_line: list[dict] = []
    line_ref_top: float | None = None

    for word in sorted_words:
        if line_ref_top is None or abs(word["top"] - line_ref_top) <= LINE_Y_TOL:
            current_line.append(word)
            if line_ref_top is None:
                line_ref_top = word["top"]
        else:
            lines.append(sorted(current_line, key=lambda w: w["x0"]))
            current_line = [word]
            line_ref_top = word["top"]

    if current_line:
        lines.append(sorted(current_line, key=lambda w: w["x0"]))

    return "\n".join(" ".join(w["text"] for w in line) for line in lines)


# ── Spatial helpers ────────────────────────────────────────────────────────────


def _word_in_any_bbox(word: dict, bboxes: list[Bbox]) -> bool:
    """Return True if the word's centre point falls inside any bbox."""
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    return any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)


# ── Table linearization ───────────────────────────────────────────────────────


def _linearize_table(table: pdfplumber.table.Table) -> str:
    """
    Serialize a pdfplumber Table into a self-contained string.

    The first non-empty row is used as column headers.  Every subsequent
    non-empty row is emitted as "Header: value | Header: value …" so that
    each row carries full context and remains meaningful after chunking.

    Example output:
        [TABLE]
        Drug: Nefecon | Dose: 16 mg/day | Duration: 9 months
        Drug: Methylprednisolone | Dose: 0.4 mg/kg/d | Duration: 2 months
    """
    rows = table.extract()
    if not rows:
        return ""

    headers: list[str] = []
    data_start = 0
    for i, row in enumerate(rows):
        if any(cell for cell in row if cell):
            headers = [str(cell).strip() if cell else "" for cell in row]
            data_start = i + 1
            break

    if not headers:
        return ""

    lines = ["[TABLE]"]
    for row in rows[data_start:]:
        if not any(cell for cell in row if cell):
            continue
        cells = [str(cell).strip() if cell else "" for cell in row]
        parts = [f"{h}: {c}" for h, c in zip(headers, cells) if h or c]
        if parts:
            lines.append(" | ".join(parts))

    return "\n".join(lines) if len(lines) > 1 else ""
