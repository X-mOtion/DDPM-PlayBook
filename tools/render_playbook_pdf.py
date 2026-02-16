#!/usr/bin/env python3
# Render the structured markdown playbook into a nicely formatted PDF.
# This is intentionally a tiny, purpose-built parser: it only supports the patterns used in
# `Books/Diffusion_6Papers_FirstPrinciples_Playbook_zh.md`.

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus import KeepTogether
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.fonts import addMapping


ZH_LABELS = ("怎样做", "为何这样做", "常见问题", "解决方式")
EN_LABELS = ("How", "Why", "Common Issues", "Fix")
EN_LABEL_ALIASES = (
    ("How", "How to do it"),
    ("Why", "Why this is done"),
    ("Common Issues", "Common problem", "Common problems", "FAQ"),
    ("Fix", "Solution"),
)

# Table fonts are resolved at runtime in `build_pdf`.
TABLE_CELL_FONT = "Helvetica"
TABLE_HEADER_FONT = "Helvetica-Bold"
MATH_SPAN_FONT = ""


@dataclass
class Item:
    title: str
    source: str
    fields: dict


def parse_md(md: str) -> Tuple[str, List[str], List[Tuple[str, List[str], List[Item]]]]:
    title = ""
    preface_paras: List[str] = []
    chapters: List[Tuple[str, List[str], List[Item]]] = []

    cur_ch_title: Optional[str] = None
    cur_ch_paras: List[str] = []
    cur_items: List[Item] = []

    cur_item: Optional[Item] = None
    in_preface = True
    last_field_key: Optional[str] = None

    def flush_item() -> None:
        nonlocal cur_item, cur_items, last_field_key
        if cur_item is None:
            return
        # Ensure all labels exist to keep the PDF layout stable.
        for k in ZH_LABELS:
            cur_item.fields.setdefault(k, "")
        cur_items.append(cur_item)
        cur_item = None
        last_field_key = None

    def flush_chapter() -> None:
        nonlocal cur_ch_title, cur_ch_paras, cur_items, chapters, last_field_key
        flush_item()
        if cur_ch_title is None:
            return
        chapters.append((cur_ch_title, cur_ch_paras, cur_items))
        cur_ch_title = None
        cur_ch_paras = []
        cur_items = []
        last_field_key = None

    h1 = re.compile(r"^#\s+(.*)\s*$")
    h2 = re.compile(r"^##\s+(.*)\s*$")
    h3 = re.compile(r"^###\s+(.*)\s*$")
    bullet = re.compile(r"^-\s*([^:]+):\s*(.*)\s*$")
    hr = re.compile(r"^\s*---\s*$")
    free_bullet = re.compile(r"^\s*-\s+(.*)\s*$")

    lines = md.splitlines()
    for raw in lines:
        line = raw.rstrip("\n")

        if not title:
            m = h1.match(line)
            if m:
                title = m.group(1).strip()
                continue

        if hr.match(line):
            # Treat as visual separator; we just end preface and ignore.
            in_preface = False
            continue

        m = h2.match(line)
        if m:
            in_preface = False
            flush_chapter()
            cur_ch_title = m.group(1).strip()
            continue

        m = h3.match(line)
        if m:
            in_preface = False
            flush_item()
            full = m.group(1).strip()
            # Expect "...  引文: ..." (or legacy "... 来源: ...")
            if "引文:" in full:
                left, right = full.split("引文:", 1)
                it_title = left.strip()
                it_source = right.strip()
            elif "来源:" in full:
                left, right = full.split("来源:", 1)
                it_title = left.strip()
                it_source = right.strip()
            else:
                it_title = full
                it_source = ""
            cur_item = Item(title=it_title, source=it_source, fields={})
            continue

        m = bullet.match(line)
        if m and cur_item is not None:
            k = m.group(1).strip()
            v = m.group(2).strip()
            cur_item.fields[k] = v
            last_field_key = k
            continue

        # Continuation lines for the last field of the current item.
        # Example:
        # - 怎样做: ...
        #   1) ...
        #   2) ...
        if cur_item is not None and last_field_key is not None:
            if line.startswith("  ") and line.strip():
                cur_item.fields[last_field_key] = (
                    cur_item.fields.get(last_field_key, "").rstrip() + "\n" + line.strip()
                ).strip()
                continue
            # Allow markdown pipe tables without indentation; keeps authoring convenient.
            # This is safe because pipe-table lines only make sense within the current cell.
            if line.startswith("|") and line.strip():
                cur_item.fields[last_field_key] = (
                    cur_item.fields.get(last_field_key, "").rstrip() + "\n" + line.strip()
                ).strip()
                continue
            if line.strip() == "":
                # Preserve paragraph breaks inside a cell.
                cur_item.fields[last_field_key] = (
                    cur_item.fields.get(last_field_key, "").rstrip() + "\n"
                )
                continue

        m = free_bullet.match(line)
        if m and cur_item is None:
            txt = m.group(1).strip()
            if in_preface:
                preface_paras.append(f"- {txt}")
            elif cur_ch_title is not None:
                cur_ch_paras.append(f"- {txt}")
            continue

        # Free text paragraphs
        if line.strip() == "":
            if in_preface:
                preface_paras.append("")
            continue

        if in_preface and line.strip():
            preface_paras.append(line.strip())
            continue

        if (not in_preface) and cur_item is None and cur_ch_title is not None and line.strip():
            cur_ch_paras.append(line.strip())

    flush_chapter()

    if not title:
        title = os.path.basename("playbook")
    return title, preface_paras, chapters


class PlaybookDoc(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        running_font: str,
        running_font_bold: str,
        cover_tagline: str,
        **kw,
    ):
        super().__init__(filename, **kw)
        self._current_chapter = ""
        self._running_font = running_font
        self._running_font_bold = running_font_bold
        self._cover_tagline = cover_tagline

        main_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="main",
        )

        cover_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="cover",
            showBoundary=0,
        )

        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[cover_frame], onPage=self._on_cover),
                # Use onPageEnd so headings placed on the page can update running headers.
                PageTemplate(id="main", frames=[main_frame], onPageEnd=self._on_page),
            ]
        )

    def beforeDocument(self):
        # multiBuild does multiple passes; reset running state each pass.
        self._current_chapter = ""

    def afterFlowable(self, flowable):
        # Capture Heading2 into TOC.
        if isinstance(flowable, Paragraph) and flowable.style.name == "PBHeading2":
            text = flowable.getPlainText()
            self._current_chapter = text
            self.notify("TOCEntry", (0, text, self.page))

    def _on_page(self, canvas, doc):
        canvas.saveState()
        # Header
        canvas.setFillColor(colors.HexColor("#5A5A5A"))
        header_font = self._running_font
        try:
            canvas.setFont(self._running_font, 9)
        except Exception:
            header_font = "STSong-Light"
            canvas.setFont(header_font, 9)
        canvas.drawString(doc.leftMargin, doc.pagesize[1] - 12 * mm, doc.title)

        # Footer
        canvas.setFillColor(colors.grey)
        try:
            canvas.setFont(self._running_font, 9)
        except Exception:
            canvas.setFont("STSong-Light", 9)
        if self._current_chapter:
            chapter = self._current_chapter
            max_width = doc.pagesize[0] - doc.leftMargin - doc.rightMargin - 30 * mm
            while chapter and canvas.stringWidth(chapter, header_font, 9) > max_width:
                chapter = chapter[:-1]
            if chapter != self._current_chapter:
                chapter = chapter.rstrip() + "..."
            canvas.drawString(doc.leftMargin, 10 * mm, chapter)
        canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 10 * mm, f"{doc.page}")
        canvas.restoreState()

    def _on_cover(self, canvas, doc):
        # Cover page: no page number, add a simple visual identity.
        canvas.saveState()
        w, h = doc.pagesize
        canvas.setFillColor(colors.HexColor("#0B3D2E"))
        canvas.rect(0, 0, 10 * mm, h, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor("#0B3D2E"))
        try:
            canvas.setFont(self._running_font_bold, 10)
        except Exception:
            canvas.setFont("STSong-Light", 10)
        canvas.drawString(doc.leftMargin, h - 12 * mm, self._cover_tagline)
        canvas.restoreState()


def _format_math_expr(expr: str) -> str:
    """
    Minimal inline "math" formatter for reportlab Paragraph XML.
    We interpret backtick-wrapped segments in the markdown as math-ish text.
    """
    from xml.sax.saxutils import escape as _xml_escape

    s = expr.strip()
    # Escape raw user content early to avoid breaking reportlab Paragraph XML.
    # We'll add <sub>/<super>/<i> tags later in this function.
    s = _xml_escape(s, entities={"'": "&apos;", '"': "&quot;"})

    # 1) Simple function-style prettification.
    s = re.sub(r"\bsqrt\s*\(", "√(", s)

    # 2) Special patterns that must run before generic underscore->subscript.
    # Keep underscores (no <sub> tags yet) so we can do operator spacing safely.
    # alpha_bar_t -> αˉ_t (avoid combining-mark rendering artifacts in some fonts)
    s = re.sub(
        r"\balpha_bar_([A-Za-z0-9+-]+)\b",
        lambda m: "αˉ_%s" % m.group(1),
        s,
    )
    s = re.sub(
        r"\bα_bar_([A-Za-z0-9+-]+)\b",
        lambda m: "αˉ_%s" % m.group(1),
        s,
    )
    s = re.sub(r"\balpha_bar\b", "αˉ", s)
    # beta_tilde (paper-style \u02DCbeta_t) and tilde_x
    s = re.sub(r"\bbeta_tilde\b", "β\u0303", s)
    s = re.sub(r"\btilde_([A-Za-z])\b", lambda m: m.group(1) + "\u0303", s)

    # 3) Operator spacing on raw text (do this before inserting any <sub>/<super> tags).
    # Avoid touching HTML tags by doing it before we introduce them.
    s = re.sub(r"\s*/\s*", " / ", s)
    s = re.sub(r"\s*\*\s*", " * ", s)
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*⊙\s*", " ⊙ ", s)
    s = s.replace("||", "‖")

    long_formula_mode = len(s) > 120

    # 4) Subscripts (ASCII + common Greek symbols).
    def _replace_braced_subscripts(text: str) -> str:
        """
        Replace balanced-brace subscripts like `alpha_{tau_{i-1}}` safely.
        """
        out: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if i + 2 < n and text[i].isalpha():
                j = i
                while j < n and (text[j].isalnum() or text[j] == "-"):
                    j += 1
                base = text[i:j]
                if j + 1 < n and text[j:j + 2] == "_{":
                    k = j + 2
                    depth = 1
                    while k < n and depth > 0:
                        if text[k] == "{":
                            depth += 1
                        elif text[k] == "}":
                            depth -= 1
                        k += 1
                    if depth == 0:
                        inner = text[j + 2 : k - 1]
                        # Keep one-level subscript only; flatten nested braces for readability.
                        inner = inner.replace("{", "(").replace("}", ")")
                        out.append(f"{base}<sub>{inner}</sub>")
                        i = k
                        continue
                out.append(base)
                i = j
                continue
            out.append(text[i])
            i += 1
        return "".join(out)

    # Always map _theta to Greek theta in both normal/long formulas.
    s = s.replace("_theta", "_θ")

    if not long_formula_mode:
        # Single pass on braced subscripts; avoid nested tiny subscripts.
        s = _replace_braced_subscripts(s)

        s = s.replace("_θ", "<sub>θ</sub>")
        # Greek letter (optionally with combining marks) + _t
        s = re.sub(
            r"([α-ωΑ-Ω][\u0300-\u036f]?)_([A-Za-z0-9+-]+)\b",
            lambda m: f"{m.group(1)}<sub>{_xml_escape(m.group(2))}</sub>",
            s,
        )
        s = re.sub(
            r"\b([A-Za-z]+)_\{([^}]+)\}\b",
            lambda m: f"{m.group(1)}<sub>{_xml_escape(m.group(2))}</sub>",
            s,
        )
        s = re.sub(
            r"\b([A-Za-z]+)_([A-Za-z0-9+-]+)\b",
            lambda m: f"{m.group(1)}<sub>{_xml_escape(m.group(2))}</sub>",
            s,
        )
        s = re.sub(
            r"αˉ_([A-Za-z0-9+-]+)\b",
            lambda m: f"αˉ<sub>{_xml_escape(m.group(1))}</sub>",
            s,
        )

    # 5) Superscripts.
    s = re.sub(r"\^(\d+)", lambda m: f"<super>{_xml_escape(m.group(1))}</super>", s)

    # 6) Greek letter names (also match identifiers like sigma<sub>t</sub>).
    # Use (?=\\b|<|_) so replacements work before tag boundaries and subscripts.
    def repl(name: str, greek: str) -> None:
        nonlocal s
        s = re.sub(rf"\b{name}(?=\b|<|_)", greek, s)

    repl("Sigma", "Σ")
    repl("mu", "μ")
    repl("alpha", "α")
    repl("beta", "β")
    repl("lamuda", "λ")  # common misspelling
    repl("lambda", "λ")
    repl("sigma", "σ")
    repl("gamma", "γ")
    repl("rho", "ρ")
    repl("eta", "η")
    repl("tau", "τ")
    repl("varphi", "ϕ")
    repl("phi", "φ")
    repl("epsilon", "ε")
    repl("eps", "ε")
    repl("theta", "θ")

    # 7) Normalize spacing but keep regular break opportunities.
    s = re.sub(r"\s*\|\s*", " | ", s)
    s = re.sub(r"\s*=\s*", " = ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # For long formulas, stop here to avoid over-formatting artifacts.
    if long_formula_mode:
        return s

    # 8) Light italics for common scalar/variable letters.
    s = re.sub(r"\b([qpx])\b", r"<i>\1</i>", s)
    return s


def format_cell_text(s: str) -> str:
    """
    Convert markdown-ish text into reportlab Paragraph XML.
    - Escapes XML for normal text.
    - Converts inline `...` segments using _format_math_expr().
    """
    def _format_math_plain(expr: str) -> str:
        t = expr.strip()
        # Stable, tag-free math normalization for PDF robustness.
        t = t.replace("sqrt(", "√(")
        # Keep math structure explicit but readable in plain-text rendering.
        t = t.replace("_{", "_(").replace("{", "(").replace("}", ")")
        t = re.sub(r"\balpha_bar_([A-Za-z0-9+-]+)\b", lambda m: f"αˉ_{m.group(1)}", t)
        t = re.sub(r"\bα_bar_([A-Za-z0-9+-]+)\b", lambda m: f"αˉ_{m.group(1)}", t)
        t = re.sub(r"\balpha_bar\b", "αˉ", t)
        t = t.replace("_theta", "_θ")
        greek_map = [
            ("Sigma", "Σ"),
            ("alpha", "α"),
            ("beta", "β"),
            ("sigma", "σ"),
            ("rho", "ρ"),
            ("lambda", "λ"),
            ("lamuda", "λ"),
            ("eps", "ε"),
            ("epsilon", "ε"),
            ("theta", "θ"),
            ("tau", "τ"),
            ("mu", "μ"),
            ("gamma", "γ"),
            ("varphi", "ϕ"),
            ("phi", "φ"),
            ("eta", "η"),
        ]
        for src, dst in greek_map:
            t = re.sub(rf"\b{src}(?=\b|_)", dst, t)
        t = t.replace("||", "‖")
        t = re.sub(r"\s+", " ", t).strip()
        txt = (
            t.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        if MATH_SPAN_FONT:
            return f'<font name="{MATH_SPAN_FONT}">{txt}</font>'
        return txt

    def _looks_like_math(span: str) -> bool:
        t = span.strip()
        if not t:
            return False
        # CLI flags / code-like spans should stay as code, not math.
        if "--" in t:
            return False
        greek_name_pattern = r"(?:alpha|beta|sigma|rho|lambda|eps|theta|mu|tau|eta|gamma|varphi|phi)"
        if re.search(rf"(?i){greek_name_pattern}", t):
            return True
        if re.search(r"^[A-Za-z{}]+(?:_[A-Za-z0-9+-]+)+(?:\([^)]*\))?$", t):
            return True
        if "/" in t and not any(op in t for op in ("=", "+", "*", "_", "^", "sqrt", "clip", "log")):
            return False
        # Heuristics for math-like spans used in this playbook.
        if any(k in t for k in ("sqrt", "clip", "log10", "||", "⊙", "^", "_{")):
            return True
        if re.search(r"\b(alpha|beta|sigma|rho|lambda|eps|theta|mu|tau|gamma|varphi)\b", t):
            return True
        if re.search(r"[A-Za-z]+_[A-Za-z0-9]+", t) and any(op in t for op in ("=", "+", "-", "*", "/")):
            return True
        if any(op in t for op in ("=", "+", "*")) and re.search(r"[A-Za-z]", t):
            return True
        return False

    def _format_code_span(span: str) -> str:
        esc = (
            span.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )
        # Use body font for robust glyph coverage in mixed content.
        return esc

    parts = s.split("`")
    out: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(
                part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )
        else:
            if _looks_like_math(part):
                out.append(_format_math_plain(part))
            else:
                out.append(_format_code_span(part))
    txt = "".join(out).strip()
    # Render newlines as hard breaks in the PDF.
    txt = txt.replace("\n", "<br/>")
    return txt


def _parse_md_table_block(
    lines: List[str], start: int, max_table_w: float
) -> Tuple[Optional[Table], int]:
    """
    Parse a GitHub-flavored markdown pipe table starting at `start`.
    Returns (table, next_index). If not a table, returns (None, start).

    Supported pattern:
      | h1 | h2 |
      |---|---:|
      | v1 | v2 |
    """
    if start >= len(lines):
        return None, start
    if not lines[start].lstrip().startswith("|"):
        return None, start
    if start + 1 >= len(lines):
        return None, start

    header = lines[start].strip()
    sep = lines[start + 1].strip()
    if not (sep.startswith("|") and "|" in sep and "-" in sep):
        return None, start
    # Must look like a separator row.
    if not re.search(r"\|\s*:?-{3,}:?\s*\|", sep):
        return None, start

    block: List[str] = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        block.append(lines[i].strip())
        i += 1

    def split_row(row: str) -> List[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    rows_raw = [split_row(r) for r in block]
    if len(rows_raw) < 3:
        return None, start

    header_cells = rows_raw[0]
    sep_cells = rows_raw[1]
    data_cells = rows_raw[2:]
    ncol = len(header_cells)
    if ncol < 2:
        return None, start
    # Normalize all rows to ncol.
    def norm(cells: List[str]) -> List[str]:
        if len(cells) < ncol:
            cells = cells + [""] * (ncol - len(cells))
        return cells[:ncol]

    header_cells = norm(header_cells)
    sep_cells = norm(sep_cells)
    data_cells = [norm(r) for r in data_cells]

    # Force all markdown table cells to center-aligned for consistent presentation.
    aligns: List[str] = ["CENTER"] * ncol

    # Convert to reportlab Paragraphs so backticks math formatting still works.
    # Keep the table readable within A4 margins.
    cell_style = ParagraphStyle(
        "PBTableCell",
        parent=getSampleStyleSheet()["BodyText"],
        fontName=TABLE_CELL_FONT,
        fontSize=7.6,
        leading=9.2,
    )
    header_style = ParagraphStyle(
        "PBTableHeader",
        parent=cell_style,
        fontName=TABLE_HEADER_FONT,
    )

    def p(txt: str, is_header: bool = False) -> Paragraph:
        return Paragraph(format_cell_text(txt) if txt else " ", header_style if is_header else cell_style)

    data: List[List[object]] = []
    data.append([p(c, True) for c in header_cells])
    for r in data_cells:
        data.append([p(c, False) for c in r])

    # Heuristic column widths: allocate by text length and fit to the caller-provided width.
    # This avoids pipe-tables overflowing the page when columns are many.
    def _plain_len(txt: str) -> int:
        t0 = re.sub(r"`([^`]*)`", r"\\1", txt or "")
        t0 = re.sub(r"\\s+", " ", t0).strip()
        return len(t0)

    col_lens: List[int] = []
    for ci in range(ncol):
        mx = _plain_len(header_cells[ci])
        for r in data_cells:
            mx = max(mx, _plain_len(r[ci]))
        # Clamp to avoid a single long string dominating the layout.
        mx = max(4, min(mx, 22))
        col_lens.append(mx)
    # Slightly favor the first column (usually "model").
    if col_lens:
        col_lens[0] = max(col_lens[0], 10)

    total = float(sum(col_lens)) if sum(col_lens) else 1.0
    col_widths = [max_table_w * (w / total) for w in col_lens]

    t = Table(data, hAlign="LEFT", repeatRows=1, colWidths=col_widths)
    # Publication-like "three-line table" look: no vertical rules, minimal ink.
    ts = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF5F1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FBFA")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 1.2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1.2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.HexColor("#9BA8A3")),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#9BA8A3")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.HexColor("#9BA8A3")),
        ]
    )
    for col, a in enumerate(aligns):
        ts.add("ALIGN", (col, 0), (col, -1), a)
    t.setStyle(ts)
    return t, i


def render_cell_flowables(
    s: str, base_style: ParagraphStyle, max_table_w: float
) -> List[object]:
    """
    Render a cell that may contain markdown tables into a list of flowables.
    Keeps normal text as Paragraphs and tables as reportlab Tables.
    """
    lines = s.splitlines()
    out: List[object] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf, out
        txt = "\n".join(buf).strip("\n")
        if txt.strip():
            out.append(Paragraph(format_cell_text(txt), base_style))
        buf = []

    i = 0
    while i < len(lines):
        tbl, nxt = _parse_md_table_block(lines, i, max_table_w=max_table_w)
        if tbl is not None:
            flush_buf()
            out.append(Spacer(1, 6))
            out.append(tbl)
            out.append(Spacer(1, 6))
            i = nxt
            continue
        buf.append(lines[i])
        i += 1

    flush_buf()
    if not out:
        out.append(Paragraph(" ", base_style))
    return out


def split_section_chunks(
    value: str,
    *,
    max_chars: int = 1200,
    max_lines: int = 24,
) -> List[str]:
    """
    Split a section body into medium chunks so each table row remains splittable across pages.
    - Keep markdown pipe tables as standalone chunks.
    - Split long prose blocks by soft boundaries (blank lines preferred).
    """
    text = value.strip("\n")
    if not text.strip():
        return [""]

    lines = text.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    buf_chars = 0
    i = 0

    def flush_buf() -> None:
        nonlocal buf, buf_chars, chunks
        if not buf:
            return
        chunk = "\n".join(buf).strip("\n")
        if chunk.strip():
            chunks.append(chunk)
        buf = []
        buf_chars = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Keep markdown table block together.
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
            and "-" in lines[i + 1]
        ):
            flush_buf()
            tbl: List[str] = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            chunks.append("\n".join(tbl).strip("\n"))
            continue

        buf.append(line)
        buf_chars += len(line) + 1
        needs_split = buf_chars >= max_chars or len(buf) >= max_lines
        if needs_split:
            # Prefer split at a recent blank line.
            split_idx = -1
            lo = max(0, len(buf) - 6)
            for j in range(len(buf) - 1, lo - 1, -1):
                if not buf[j].strip():
                    split_idx = j
                    break
            if 0 <= split_idx < len(buf) - 1:
                head = buf[: split_idx + 1]
                tail = buf[split_idx + 1 :]
                chunk = "\n".join(head).strip("\n")
                if chunk.strip():
                    chunks.append(chunk)
                buf = tail
                buf_chars = sum(len(x) + 1 for x in buf)
            else:
                flush_buf()
        i += 1

    flush_buf()
    return chunks or [""]


def build_pdf(input_md: str, output_pdf: str) -> None:
    # Prefer embedding an open-source TTF for print-ready output.
    # Fallback to CID font if the font files are missing.
    fonts_dir = os.path.join(os.path.dirname(__file__), "fonts")
    body_font = "NotoSansSC-Body-static"
    body_font_boldish = "NotoSansSC-Bold-static"
    title_font = "NotoSansSC-Bold-static"
    fallback_cid = "STSong-Light"

    with open(input_md, "r", encoding="utf-8") as f:
        md = f.read()

    input_name = os.path.basename(input_md).lower()
    is_english_doc = (
        input_name.endswith("_en.md")
        or input_name.endswith("-en.md")
        or bool(re.search(r"^\s*##\s+Abstract\s*$", md, flags=re.MULTILINE))
    )

    try:
        # Google Noto Latin.
        pdfmetrics.registerFont(
            TTFont("NotoSans-Regular", os.path.join(fonts_dir, "NotoSans-Regular.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("NotoSans-Bold", os.path.join(fonts_dir, "NotoSans-Bold.ttf"))
        )
        # Google Noto SC static instances.
        pdfmetrics.registerFont(
            TTFont("NotoSansSC-Body-static", os.path.join(fonts_dir, "NotoSansSC-Body-static.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("NotoSansSC-Bold-static", os.path.join(fonts_dir, "NotoSansSC-Bold-static.ttf"))
        )

        # Choose families by document language.
        if is_english_doc:
            body_font = "NotoSans-Regular"
            body_font_boldish = "NotoSans-Bold"
            title_font = "NotoSans-Bold"
        else:
            body_font = "NotoSansSC-Body-static"
            body_font_boldish = "NotoSansSC-Bold-static"
            title_font = "NotoSansSC-Bold-static"

        # Ensure ReportLab has sensible family mappings.
        addMapping("NotoSansSC", 0, 0, "NotoSansSC-Body-static")
        addMapping("NotoSansSC", 0, 1, "NotoSansSC-Bold-static")
        addMapping("NotoSansSC", 1, 0, "NotoSansSC-Bold-static")
        addMapping("NotoSansSC", 1, 1, "NotoSansSC-Bold-static")
        addMapping("NotoSans", 0, 0, "NotoSans-Regular")
        addMapping("NotoSans", 0, 1, "NotoSans-Bold")
        addMapping("NotoSans", 1, 0, "NotoSans-Bold")
        addMapping("NotoSans", 1, 1, "NotoSans-Bold")
        running_font = body_font
    except Exception:
        pdfmetrics.registerFont(UnicodeCIDFont(fallback_cid))
        body_font = fallback_cid
        body_font_boldish = fallback_cid
        title_font = fallback_cid
        running_font = fallback_cid

    # Resolve fonts for markdown pipe tables. Use a family name for TT fonts
    # so Paragraph parser can map bold/italic safely via addMapping().
    global TABLE_CELL_FONT, TABLE_HEADER_FONT, MATH_SPAN_FONT
    if running_font == fallback_cid:
        TABLE_CELL_FONT = fallback_cid
        TABLE_HEADER_FONT = fallback_cid
    else:
        TABLE_CELL_FONT = body_font
        TABLE_HEADER_FONT = body_font_boldish

    # Dedicated font for math spans to ensure Greek glyph coverage.
    MATH_SPAN_FONT = ""
    try:
        math_font_path = os.path.join(fonts_dir, "NotoSans-Regular.ttf")
        if os.path.exists(math_font_path):
            pdfmetrics.registerFont(TTFont("PB-Math-Unicode", math_font_path))
            MATH_SPAN_FONT = "PB-Math-Unicode"
    except Exception:
        MATH_SPAN_FONT = ""

    section_labels = EN_LABELS if is_english_doc else ZH_LABELS
    toc_title = "Contents" if is_english_doc else "目录"
    ref_prefix = "Ref" if is_english_doc else "引文"
    cover_subtitle = (
        "Engineering structured summary strictly based on 6 papers (publication layout edition)"
        if is_english_doc
        else "严格基于 6 篇论文的工程化条目式总结（出版排版版）"
    )
    cover_tagline = (
        "Structured handbook | strictly based on 6 papers"
        if is_english_doc
        else "条目式技术手册 | 仅基于 6 篇论文"
    )

    title, preface_paras, chapters = parse_md(md)
    author_name = ""
    for p in preface_paras:
        m = re.match(r"^\s*编者[:：]\s*(.+?)\s*$", p)
        if m:
            author_name = m.group(1).strip()
            break

    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "PBBase",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=10.6,
        leading=15.4,
        spaceAfter=4.2,
        textColor=colors.HexColor("#1F1F1F"),
        wordWrap="CJK",
    )
    small = ParagraphStyle(
        "PBSmall",
        parent=base,
        fontSize=9.6,
        leading=13.8,
        textColor=colors.HexColor("#2A2A2A"),
    )
    h1 = ParagraphStyle(
        "PBTitle",
        parent=base,
        fontName=title_font,
        fontSize=23,
        leading=29,
        spaceAfter=12,
        textColor=colors.HexColor("#1A1A1A"),
    )
    h2 = ParagraphStyle(
        "PBHeading2",
        parent=base,
        fontName=body_font_boldish,
        fontSize=14.2,
        leading=18.8,
        spaceBefore=11,
        spaceAfter=7.2,
        textColor=colors.HexColor("#0B3D2E"),
        keepWithNext=True,
    )
    h3 = ParagraphStyle(
        "PBHeading3",
        parent=base,
        fontName=body_font_boldish,
        fontSize=11.8,
        leading=16.4,
        spaceBefore=7.5,
        spaceAfter=5.2,
        textColor=colors.HexColor("#202020"),
        keepWithNext=True,
    )
    meta = ParagraphStyle(
        "PBMeta",
        parent=small,
        fontSize=8.9,
        leading=12.0,
        textColor=colors.HexColor("#595959"),
        spaceAfter=2,
    )

    doc = PlaybookDoc(
        output_pdf,
        running_font=running_font,
        running_font_bold=body_font_boldish,
        pagesize=A4,
        leftMargin=18.5 * mm,
        rightMargin=18.5 * mm,
        topMargin=16.5 * mm,
        bottomMargin=16.5 * mm,
        title=title,
        author=author_name,
        cover_tagline=cover_tagline,
    )

    story = []

    # Cover page
    story.append(Paragraph(title, h1))
    story.append(Paragraph(cover_subtitle, small))
    story.append(Spacer(1, 6 * mm))

    # Preface paragraphs: keep tight.
    for p in preface_paras:
        if not p.strip():
            continue
        if p.startswith("- "):
            story.append(Paragraph(p, small))
        else:
            # Slightly emphasize editor line on cover.
            if re.match(r"^\s*编者[:：]\s*", p):
                story.append(Paragraph(p, ParagraphStyle("PBEditor", parent=base, textColor=colors.HexColor("#0B3D2E"))))
            else:
                story.append(Paragraph(p, base))
    story.append(Spacer(1, 6 * mm))

    from reportlab.platypus import NextPageTemplate, PageBreak  # local import

    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # TOC page
    toc = TableOfContents(rightColumnWidth=20 * mm, dotsMinLevel=0)
    toc.levelStyles = [
        ParagraphStyle(
            "PBTOC0",
            parent=small,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=2,
            spaceAfter=2,
        )
    ]
    story.append(Paragraph(toc_title, h2))
    story.append(toc)
    story.append(PageBreak())

    # Chapters + items
    for ch_title, ch_paras, items in chapters:
        story.append(Paragraph(ch_title, h2))
        for p in ch_paras:
            if p.startswith("- "):
                story.append(Paragraph(p, small))
            else:
                story.append(Paragraph(p, base))
        if ch_paras:
            story.append(Spacer(1, 3 * mm))
        for it in items:
            # Item header with a subtle band
            header = it.title
            block: List[object] = []
            block.append(Paragraph(header, h3))
            if it.source:
                block.append(Paragraph(f"{ref_prefix}: {it.source}", meta))

            label_col_w = (33 * mm) if is_english_doc else (27 * mm)
            detail_col_w = doc.width - label_col_w
            item_sections: List[Tuple[str, str]] = []
            for idx, label in enumerate(section_labels):
                value = ""
                if is_english_doc:
                    for alias in EN_LABEL_ALIASES[idx]:
                        if it.fields.get(alias):
                            value = it.fields.get(alias, "").strip()
                            break
                    if not value:
                        value = it.fields.get(ZH_LABELS[idx], "").strip()
                else:
                    value = it.fields.get(ZH_LABELS[idx], "").strip()
                    if not value:
                        for alias in EN_LABEL_ALIASES[idx]:
                            if it.fields.get(alias):
                                value = it.fields.get(alias, "").strip()
                                break
                item_sections.append((label, value))

            label_style = ParagraphStyle(
                "PBLabel",
                parent=small,
                fontSize=9.6,
                leading=12.6,
                textColor=colors.HexColor("#0B3D2E"),
            )
            rows = []
            for label, value in item_sections:
                chunks = split_section_chunks(value) if value else [""]
                for chunk in chunks:
                    rows.append(
                        [
                            Paragraph(label, label_style),
                            render_cell_flowables(chunk, base, max_table_w=detail_col_w - 14)
                            if chunk.strip()
                            else [Paragraph(" ", base)],
                        ]
                    )

            table = Table(
                rows,
                colWidths=[label_col_w, detail_col_w],
                hAlign="LEFT",
                splitByRow=1,
            )
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("INNERGRID", (0, 0), (-1, -1), 0.30, colors.HexColor("#E2E6E4")),
                        ("BOX", (0, 0), (-1, -1), 0.66, colors.HexColor("#C0C8C4")),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF5F1")),
                        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#FAFCFB")]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4.8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.8),
                    ]
                )
            )
            block.append(table)
            block.append(Spacer(1, 5 * mm))
            story.extend(block)

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    # TableOfContents needs a multi-pass build to resolve page numbers.
    doc.multiBuild(story)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input markdown playbook")
    ap.add_argument("--output", required=True, help="Output PDF path")
    args = ap.parse_args(argv)

    build_pdf(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
