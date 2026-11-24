"""Read study files (PDF, PPTX, DOCX, TXT/MD) and split them into chunks.

Every chunk remembers where it came from (file, page/slide/section) and the
nearest heading, so answers can cite sources and topics can follow the
structure of the original material.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass

CHUNK_SIZE = 1400  # characters
SUPPORTED = (".pdf", ".pptx", ".docx", ".txt", ".md")


@dataclass
class Chunk:
    id: int
    doc: str
    loc: str  # e.g. "p. 12", "slide 4", "section 2"
    heading: str
    text: str
    topic: str = ""
    parent: str = ""  # chapter title when the file has chapters > sections (PDF bookmarks)
    in_topics: bool = True   # False for answer keys, front/back matter: never a topic to study or quiz
    searchable: bool = True  # False for covers, copyright, index...: not even used to answer questions


# Keyword PRIOR for sections that are not study content (used when no AI is available, or if it fails).
#   "drop"    -> not searchable, not a topic (cover, copyright, preface, bibliography, index, credits ...)
#   "notopic" -> kept searchable for Q&A but never a topic (answer keys, glossary)
_DROP = re.compile(
    r"\b(cover|copyright|title page|half[- ]?title|preface|foreword|to the (student|instructor|reader|teacher)s?|"
    r"companion web ?site|about the authors?|acknowledg\w*|dedication|bibliograph\w*|credits|eula|"
    r"end user license|(suggested|further) readings?|list of (figures|tables|symbols))\b"
    r"|^(table of )?contents$|^index$|^references$", re.I)
_NOTOPIC = re.compile(r"^(answers?|solutions?|glossary|hints?)\b|\b(answers?|solutions?|hints?) to\b", re.I)


def _kind(title: str) -> str:
    t = re.sub(r"^[\d.\s:–-]+", "", (title or "").strip())  # ignore numbering like "7." or "A:"
    if _DROP.search(t):
        return "drop"
    return "notopic" if _NOTOPIC.search(t) else "keep"


# ----------------------------------------------------------- slide titles
_GENERIC = {"exercise", "exercises", "example", "examples", "solution", "solutions", "remark", "remarks",
            "note", "notes", "proof", "answer", "answers", "definition", "definitions", "theorem"}
_FILLER = {"contd", "cont", "continued", "part", "ii", "iii", "iv"}


def _is_generic(title: str) -> bool:
    words = [w for w in re.findall(r"[a-z]+", title.lower()) if w not in _FILLER]
    return bool(words) and all(w in _GENERIC for w in words)


def _inherit(titles):
    """'EXAMPLE' / 'EXERCISE' slides take the title of the concept slide before them."""
    out, last = [], ""
    for t in titles:
        if t and not _is_generic(t):
            last = t
        out.append(last or t)
    return out


def _pdf_title(page) -> str:
    """Slide-style title: the larger text at the top of the page (empty if there isn't one)."""
    lines = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        for l in b["lines"]:
            text = "".join(sp["text"] for sp in l["spans"]).strip()
            if text and l["spans"]:
                lines.append((l["bbox"][1], max(sp["size"] for sp in l["spans"]), text))
    if len(lines) < 2:
        return ""
    body = sorted(sz for _, sz, _ in lines)[len(lines) // 2]
    top = sorted((y, sz, t) for y, sz, t in lines if y < 0.25 * page.rect.height and sz >= body * 1.1)
    if not top:
        return ""
    y0, sz0, t0 = top[0]
    parts, last_y = [t0], y0
    for y, sz, t in top[1:]:
        if abs(sz - sz0) < 0.5 and y - last_y < sz0 * 1.8:  # second line of a long title
            parts.append(t)
            last_y = y
        else:
            break
    title = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return title if len(title) <= 90 else ""


# ---------------------------------------------------------------- readers
# Each reader returns a list of units: (location label, heading, text)

def _read_pdf(data: bytes):
    import pymupdf as fitz  # PyMuPDF

    pdf = fitz.open(stream=data, filetype="pdf")
    heading_at, parent_at = {}, {}
    for level, title, page in pdf.get_toc():  # the PDF's own bookmarks
        if level <= 2:
            heading_at[page] = title.strip()
        if level == 1:
            parent_at[page] = title.strip()
    pages = [(page.get_text().replace("\x00", "•").strip(), _pdf_title(page)) for page in pdf]
    titles = [t for _, t in pages]
    # slide-deck PDFs: most pages have a title. Normal books/articles: don't guess.
    use_titles = not heading_at and len(pages) >= 5 and sum(1 for t in titles if t) >= 0.5 * len(pages)
    effective = _inherit(titles) if use_titles else [""] * len(pages)
    units, current, parent = [], "", ""
    for i, ((text, _), eff) in enumerate(zip(pages, effective), start=1):
        current = heading_at.get(i, current)
        parent = parent_at.get(i, parent)
        if text:  # keep every page; the AI (or the keyword prior) decides later what counts as study content
            units.append((f"p. {i}", eff or current, text, parent, _kind(parent or current)))
    return units


def _read_pptx(data: bytes):
    from pptx import Presentation

    prs = Presentation(io.BytesIO(data))
    units, raw_titles = [], []
    for i, slide in enumerate(prs.slides, start=1):
        title = ""
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text_frame.text.strip()
        parts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = shape.text_frame.text.strip()
                if t:
                    parts.append(t)
            elif getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append("Speaker notes: " + notes)
        if parts:
            units.append((f"slide {i}", title, "\n".join(parts)))
    eff = _inherit([u[1] for u in units])  # Example/Exercise slides inherit the concept title
    return [(loc, e, text) for (loc, _, text), e in zip(units, eff)]


def _read_docx(data: bytes):
    from docx import Document

    doc = Document(io.BytesIO(data))
    units, heading, buf = [], "", []

    def flush():
        if buf:
            units.append((f"section {len(units) + 1}", heading, "\n".join(buf)))
            buf.clear()

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        if p.style is not None and p.style.name.lower().startswith("heading"):
            flush()
            heading = text
        buf.append(text)
    flush()
    return units


def _read_text(data: bytes):
    text = data.decode("utf-8", errors="ignore")
    units, heading, buf, size = [], "", [], 0

    def flush():
        nonlocal size
        if buf:
            units.append((f"part {len(units) + 1}", heading, "\n".join(buf)))
            buf.clear()
            size = 0

    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:  # markdown heading starts a new unit
            flush()
            heading = m.group(1).strip()
        elif size > 1500 and not line.strip():
            flush()
            continue
        if line.strip():
            buf.append(line.strip())
            size += len(line)
    flush()
    return units


READERS = {".pdf": _read_pdf, ".pptx": _read_pptx, ".docx": _read_docx,
           ".txt": _read_text, ".md": _read_text}


# --------------------------------------------------------------- chunking

def _split(text: str, size: int = CHUNK_SIZE):
    """Split text into pieces of at most `size` characters, on line breaks."""
    out, buf = [], ""
    for line in (l.strip() for l in text.splitlines()):
        if not line:
            continue
        while len(line) > size:  # very long line: hard split
            if buf:
                out.append(buf)
                buf = ""
            out.append(line[:size])
            line = line[size:]
        if buf and len(buf) + len(line) + 1 > size:
            out.append(buf)
            buf = ""
        buf = f"{buf}\n{line}" if buf else line
    if buf:
        out.append(buf)
    return out


def build_chunks(files):
    """files: list of (filename, bytes). Returns a list of Chunk objects."""
    chunks: list[Chunk] = []
    for name, data in files:
        ext = os.path.splitext(name)[1].lower()
        if ext not in READERS:
            raise ValueError(f"'{name}': unsupported file type. Use {', '.join(SUPPORTED)}.")
        units = READERS[ext](data)
        before = len(chunks)
        for loc, heading, text, *rest in units:
            parent = rest[0] if rest else ""
            kind = rest[1] if len(rest) > 1 else "keep"  # keyword prior: keep / notopic / drop
            for piece in _split(text):
                if len(piece) >= 20:
                    chunks.append(Chunk(len(chunks), name, loc, heading, piece, parent=parent,
                                        in_topics=kind == "keep", searchable=kind != "drop"))
        if len(chunks) == before:
            raise ValueError(f"'{name}': no readable text found (is it a scanned PDF?).")
    return chunks
