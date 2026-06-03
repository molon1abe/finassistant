"""
Sberbank debit/credit card statement parser.

Text layout per transaction (tab-separated columns, 2+ lines):
    Line 1: {op_date}\t{op_time}\t{category}\t{amount_rub}\t{balance_rub}
    Line 2: {proc_date}\t{auth_code}\t{description}[\t{foreign_amount} {currency}]
    Line 3+: optional description continuation

Reference: https://github.com/Ev2geny/Sberbank2Excel
"""

import re
import fitz
import structlog
from langchain_core.documents import Document

from .loaders import file_uuid

logger = structlog.get_logger()

# Gap in points between words on the same line that triggers a tab separator.
# Within a cell (e.g. "6 698,00") gaps are <5pt; between columns gaps are >30pt.
_TAB_GAP = 20.0

# Repeated page header block inserted by Sberbank between pages — strip it out.
_PAGE_BREAK_RE = re.compile(
    r"Продолжение на следующей странице[\s\S]*?код авторизации.{0,3}операции.{0,3}\n",
    re.IGNORECASE,
)

# Each transaction:
#   line 1  — DD.MM.YYYY (space or tab) HH:MM …
#   line 2  — DD.MM.YYYY (space or tab) auth_code …
#   lines 3+ — continuation (no date+time at start)
_ENTRY_RE = re.compile(
    r"("
    r"\d{2}\.\d{2}\.\d{4}[ \t]\d{2}:\d{2}[^\n]*\n"  # line 1
    r"\d{2}\.\d{2}\.\d{4}[ \t][\d\-]{1,8}[^\n]*"  # line 2
    r"(?:\n(?!\d{2}\.\d{2}\.\d{4}[ \t]\d{2}:\d{2})[^\n]*)* "  # continuation
    r")"
)


def _page_to_text(page: fitz.Page) -> str:
    """
    Reconstruct tab-separated columnar text from PyMuPDF word positions.
    Words on the same y-row separated by a gap > _TAB_GAP become separate
    tab-delimited fields; smaller gaps become spaces.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, text, block, line, word)
    if not words:
        return ""

    # Snap words to rows by y-coordinate (4pt grid)
    rows: dict[int, list] = {}
    for w in words:
        y_key = round(w[1] / 4) * 4
        rows.setdefault(y_key, []).append(w)

    lines = []
    for y_key in sorted(rows):
        row_words = sorted(rows[y_key], key=lambda w: w[0])
        parts: list[str] = []
        buf: list[str] = []
        prev_x1: float | None = None

        for w in row_words:
            if prev_x1 is not None and (w[0] - prev_x1) > _TAB_GAP:
                parts.append(" ".join(buf))
                buf = []
            buf.append(w[4])
            prev_x1 = w[2]

        if buf:
            parts.append(" ".join(buf))
        lines.append("\t".join(parts))

    return "\n".join(lines)


def _parse_entry(entry: str, file_id: str, source: str) -> Document | None:
    """
    Convert one raw transaction entry string into a labelled Document.
    Handles both tab-split (date\\ttime) and space-split (date time) layouts
    since PyMuPDF may merge them depending on the gap threshold.
    """
    lines = [line for line in entry.split("\n") if line.strip()]
    if len(lines) < 2:
        return None

    cols1 = lines[0].split("\t")
    cols2 = lines[1].split("\t")

    # cols1[0] is either "05.11.2024" or "05.11.2024 14:32"
    # If date and time are separate tab fields, merge them.
    if (
        re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", cols1[0])
        and len(cols1) >= 2
        and re.fullmatch(r"\d{2}:\d{2}", cols1[1])
    ):
        op_datetime = cols1[0] + " " + cols1[1]
        cols1 = cols1[2:]
    else:
        op_datetime = cols1[0]
        cols1 = cols1[1:]

    # Same for cols2: processing date + auth code may be merged or separate
    if (
        re.fullmatch(r"\d{2}\.\d{2}\.\d{4}", cols2[0])
        and len(cols2) >= 2
        and re.fullmatch(r"[\d\-]{1,8}", cols2[1])
    ):
        proc_auth = cols2[0] + " " + cols2[1]
        cols2 = cols2[2:]
    else:
        proc_auth = cols2[0]
        cols2 = cols2[1:]

    category = cols1[0] if len(cols1) > 0 else ""
    amount = cols1[1] if len(cols1) > 1 else ""
    balance = cols1[2] if len(cols1) > 2 else ""
    description = cols2[0] if len(cols2) > 0 else ""

    # Append continuation lines to description
    for extra in lines[2:]:
        ep = extra.split("\t")
        if len(ep) == 1 and not re.match(r"\d{2}\.\d{2}\.\d{4}", ep[0]):
            description = description + " " + ep[0].strip()

    parts = [
        f"Transaction Date: {op_datetime}",
        f"Processing Date / Auth: {proc_auth}",
    ]
    if category:
        parts.append(f"Category: {category}")
    if description.strip():
        parts.append(f"Description: {description.strip()}")
    if amount:
        parts.append(f"Amount (RUB): {amount}")
    if balance:
        parts.append(f"Balance (RUB): {balance}")

    return Document(
        page_content="\n".join(parts), metadata={"file_id": file_id, "source": source}
    )


def load_sber_pdf(file_path: str) -> list[Document]:
    file_id = file_uuid(file_path)
    docs: list[Document] = []

    full_text_parts = []
    with fitz.open(file_path) as pdf:
        for page in pdf:
            full_text_parts.append(_page_to_text(page))
    full_text = "\n".join(full_text_parts)

    logger.debug("Raw extracted text (first 800 chars)", preview=full_text[:800])

    # Remove repeated inter-page header blocks
    full_text = _PAGE_BREAK_RE.sub("", full_text)

    entries = _ENTRY_RE.findall(full_text)
    logger.info("Transactions found", count=len(entries))

    if not entries:
        logger.debug("Text sample for debugging", text=full_text[:2000])

    for entry in entries:
        doc = _parse_entry(entry, file_id, file_path)
        if doc:
            docs.append(doc)

    if not docs:
        logger.warning(
            "Sber parser found no transactions, falling back to generic loader"
        )

    return docs
