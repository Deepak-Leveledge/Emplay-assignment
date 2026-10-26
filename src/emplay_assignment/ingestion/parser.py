
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
from bs4 import BeautifulSoup

from emplay_assignment.config import settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}


def normalize_text(text: str) -> str:
    """Clean extracted text while preserving useful line breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00ad", "")

    # Join words broken across PDF line endings, e.g. "dead-\nline".
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_tables(tables: list[Any]) -> str:
    """Convert extracted PDF/HTML tables into readable Markdown."""
    formatted_tables: list[str] = []

    for table_number, table in enumerate(tables, start=1):
        rows: list[list[str]] = []

        for row in table or []:
            cells = [
                normalize_text(str(cell or "")).replace("\n", " ")
                for cell in row
            ]

            while cells and not cells[-1]:
                cells.pop()

            if any(cells):
                rows.append(cells)

        if not rows:
            continue

        column_count = max(len(row) for row in rows)
        lines: list[str] = []

        for row_index, row in enumerate(rows):
            padded_row = row + [""] * (column_count - len(row))
            lines.append("| " + " | ".join(padded_row) + " |")

            if row_index == 0:
                lines.append(
                    "| " + " | ".join(["---"] * column_count) + " |"
                )

        formatted_tables.append(
            f"Table {table_number}\n" + "\n".join(lines)
        )

    return "\n\n".join(formatted_tables)


def infer_bid_id(path: Path) -> str | None:
    """Find a folder component such as Bid1 or Bid2."""
    for part in path.parts:
        if re.fullmatch(r"bid[_ -]?\d+", part, flags=re.IGNORECASE):
            return part.replace(" ", "").replace("_", "").replace("-", "")

    return None


def infer_doc_type(path: Path) -> str:
    """Classify a source document using its file type and filename."""
    name = path.name.lower()

    if path.suffix.lower() in {".html", ".htm"}:
        return "bid_page"

    if "addendum" in name:
        return "addendum"

    if "affidavit" in name:
        return "affidavit"

    if "spec" in name:
        return "specs"

    if any(word in name for word in ("rfp", "porfp", "final", "solicitation")):
        return "rfp"

    return "other"


def infer_addendum_number(path: Path) -> int | None:
    match = re.search(r"\baddendum[\s_-]*(\d+)\b", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_document_date(text: str) -> str | None:
    """
    Extract a date only when it follows a recognizable document-date label.
    Do not confuse an unlabeled due date with the document's own date.
    """
    pattern = re.compile(
        r"\b(?:solicitation|issue|issued|publication|posting|document)"
        r"\s+date\s*[:#\-]?\s*"
        r"("
        r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|\d{4}-\d{2}-\d{2}"
        r")",
        re.IGNORECASE,
    )

    match = pattern.search(text[:30000])
    return match.group(1).strip() if match else None


def build_record(
    path: Path,
    text: str,
    page_number: int | None,
    document_date: str | None,
) -> dict[str, Any]:
    """Attach citation-friendly metadata to an extracted record."""
    try:
        relative_path = path.resolve().relative_to(
            settings.raw_data_dir.resolve()
        ).as_posix()
    except ValueError:
        relative_path = path.name

    metadata = {
        "bid_id": infer_bid_id(path),
        "file_name": path.name,
        "relative_path": relative_path,
        "doc_type": infer_doc_type(path),
        "addendum_number": infer_addendum_number(path),
        "page_number": page_number,
        "document_date": document_date,
        "source_format": path.suffix.lower().lstrip("."),
        "is_empty": not bool(text.strip()),
    }

    unique_string = f"{relative_path}|{page_number}|{text}"
    doc_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()[:20]

    return {
        "doc_id": doc_id,
        "text": text,
        "metadata": metadata,
    }


def parse_pdf(path: Path) -> list[dict[str, Any]]:
    """Extract PDF text page-by-page and preserve tables where possible."""
    records: list[dict[str, Any]] = []
    pdf_tables = None

    try:
        pdf_tables = pdfplumber.open(str(path))
    except Exception as exc:
        # Text extraction should still work if table extraction fails.
        logger.warning("Table extraction unavailable for %s: %s", path.name, exc)

    try:
        with fitz.open(path) as pdf:
            page_texts = [
                page.get_text("text") or ""
                for page in pdf
            ]

            document_date = extract_document_date(
                "\n".join(page_texts[:3])
            )

            for index, page_text in enumerate(page_texts):
                table_text = ""

                if pdf_tables is not None:
                    try:
                        extracted = pdf_tables.pages[index].extract_tables() or []
                        table_text = format_tables(extracted)
                    except Exception as exc:
                        logger.warning(
                            "Table extraction failed: %s | page %s | %s",
                            path.name,
                            index + 1,
                            exc,
                        )

                combined_text = page_text

                if table_text:
                    combined_text += (
                        "\n\n[EXTRACTED TABLES]\n" + table_text
                    )

                cleaned_text = normalize_text(combined_text)

                if not cleaned_text:
                    logger.warning(
                        "Empty page detected: %s | page %s. OCR may be needed.",
                        path.name,
                        index + 1,
                    )

                records.append(
                    build_record(
                        path=path,
                        text=cleaned_text,
                        page_number=index + 1,
                        document_date=document_date,
                    )
                )

    finally:
        if pdf_tables is not None:
            pdf_tables.close()

    return records


def parse_html(path: Path) -> list[dict[str, Any]]:
    """Extract readable HTML text and retain table structure."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    title = ""
    if soup.title:
        title = normalize_text(soup.title.get_text(" ", strip=True))

    content_root = soup.body or soup
    text = normalize_text(content_root.get_text("\n", strip=True))

    # tables = [
    #     format_tables([table.find_all("tr")])
    #     for table in []
    # ]

    html_tables: list[list[list[str]]] = []

    for table in soup.find_all("table"):
        table_rows: list[list[str]] = []

        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if cells:
                table_rows.append([
                    cell.get_text(" ", strip=True)
                    for cell in cells
                ])

        if table_rows:
            html_tables.append(table_rows)

    tables_text = format_tables(html_tables)

    if title and title.lower() not in text[:len(title) + 50].lower():
        text = f"{title}\n\n{text}"

    if tables_text:
        text += "\n\n[EXTRACTED HTML TABLES]\n" + tables_text

    text = normalize_text(text)
    document_date = extract_document_date(text)

    if not text:
        logger.warning("HTML document contains no readable text: %s", path.name)

    return [
        build_record(
            path=path,
            text=text,
            page_number=None,
            document_date=document_date,
        )
    ]


def parse_file(path: Path) -> list[dict[str, Any]]:
    """Dispatch a file to its appropriate parser."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(path)

    if suffix in {".html", ".htm"}:
        return parse_html(path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def main() -> None:
    """Parse all supported files and write normalized JSONL output."""
    input_dir = settings.raw_data_dir
    output_dir = settings.processed_data_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    source_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_files:
        raise FileNotFoundError(
            f"No PDF/HTML files were found under: {input_dir}"
        )

    all_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in source_files:
        try:
            records = parse_file(path)
            all_records.extend(records)
            logger.info(
                "Parsed %s -> %s record(s)",
                path.name,
                len(records),
            )
        except Exception as exc:
            logger.exception("Failed to parse source file: %s", path)
            errors.append({
                "file": str(path),
                "error_type": type(exc).__name__,
                "message": str(exc),
            })

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "normalized_documents.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for record in all_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    error_file = output_dir / "parse_errors.jsonl"

    with error_file.open("w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")

    empty_records = sum(
        1 for record in all_records
        if record["metadata"]["is_empty"]
    )

    logger.info("Source files found: %s", len(source_files))
    logger.info("Records written: %s", len(all_records))
    logger.info("Empty records: %s", empty_records)
    logger.info("Files with parsing errors: %s", len(errors))
    logger.info("Normalized output: %s", output_file)
    logger.info("Error log: %s", error_file)


if __name__ == "__main__":
    main()
