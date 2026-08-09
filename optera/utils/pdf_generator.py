"""
Optera PDF Report Builder & Compiler Utility

Uses ReportLab to generate publication-grade, executive PDF reports
for all Optera layers (ETL, Analytics, Optimization, Simulation).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger("OpteraPDFBuilder")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page count footer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#666666"))

        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(612 - 36, 20, page_text)
        self.drawString(36, 20, "Optera Quantitative Supply Chain Engine — Official Executive Report")
        self.line(36, 30, 612 - 36, 30)

        self.restoreState()


class OpteraPDFBuilder:
    """
    Executive PDF Report Builder.
    """

    def __init__(self, title: str = "Optera Framework Executive Report", subtitle: Optional[str] = None):
        self.title = title
        self.subtitle = subtitle
        self.story = []

        # Color Palette
        self.NAVY = colors.HexColor("#1B365D")
        self.PRIMARY = colors.HexColor("#1F77B4")
        self.TEXT_DARK = colors.HexColor("#2C3E50")
        self.LIGHT_BG = colors.HexColor("#F8F9FA")
        self.BORDER_COLOR = colors.HexColor("#E0E0E0")

        # Base Stylesheet
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        # Build initial header
        self._build_header()

    def _setup_custom_styles(self):
        """Sets up executive typography and text styles."""
        self.title_style = ParagraphStyle(
            "OpteraTitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=self.NAVY,
            spaceAfter=4
        )

        self.subtitle_style = ParagraphStyle(
            "OpteraSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12
        )

        self.h1_style = ParagraphStyle(
            "OpteraH1",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=self.NAVY,
            spaceBefore=12,
            spaceAfter=6
        )

        self.h2_style = ParagraphStyle(
            "OpteraH2",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=self.PRIMARY,
            spaceBefore=10,
            spaceAfter=4
        )

        self.body_style = ParagraphStyle(
            "OpteraBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=self.TEXT_DARK,
            spaceAfter=4
        )

        self.bullet_style = ParagraphStyle(
            "OpteraBullet",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=self.TEXT_DARK,
            leftIndent=12,
            spaceAfter=2
        )

        self.table_header_style = ParagraphStyle(
            "OpteraTableHeader",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=1
        )

        self.table_cell_style = ParagraphStyle(
            "OpteraTableCell",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=7.0,
            leading=9.0,
            textColor=self.TEXT_DARK,
            alignment=0
        )

    def _build_header(self):
        """Constructs report document banner."""
        self.story.append(Paragraph(self.title, self.title_style))
        if self.subtitle:
            self.story.append(Paragraph(self.subtitle, self.subtitle_style))
        self.story.append(HRFlowable(width="100%", thickness=1.5, color=self.NAVY, spaceAfter=10))

    def add_heading(self, text: str, level: int = 1):
        """Adds a section heading."""
        style = self.h1_style if level == 1 else self.h2_style
        self.story.append(Paragraph(text, style))

    def add_paragraph(self, text: str):
        """Adds a paragraph of text."""
        self.story.append(Paragraph(text, self.body_style))

    def add_bullet(self, key: str, value: str):
        """Adds a bullet point key-value string."""
        formatted = f"• <b>{key}</b>: {value}"
        self.story.append(Paragraph(formatted, self.bullet_style))

    def add_spacer(self, height: float = 8.0):
        """Adds vertical whitespace."""
        self.story.append(Spacer(1, height))

    def add_table(self, df: pd.DataFrame, max_width: float = 540.0):
        """Renders a pandas DataFrame as a styled ReportLab Table."""
        headers = [Paragraph(str(col), self.table_header_style) for col in df.columns]

        data_rows = []
        for _, row in df.iterrows():
            formatted_row = []
            for val in row:
                if isinstance(val, float):
                    val_str = f"{val:,.2f}"
                elif isinstance(val, int):
                    val_str = f"{val:,}"
                else:
                    val_str = str(val)
                formatted_row.append(Paragraph(val_str, self.table_cell_style))
            data_rows.append(formatted_row)

        table_data = [headers] + data_rows
        num_cols = len(df.columns)
        col_w = max_width / num_cols

        t = Table(table_data, colWidths=[col_w] * num_cols)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), self.NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.LIGHT_BG]),
            ("GRID", (0, 0), (-1, -1), 0.5, self.BORDER_COLOR)
        ]))

        self.story.append(t)
        self.story.append(Spacer(1, 8))

    def add_image(self, image_path: Union[str, Path], max_width: float = 530.0, max_height: float = 240.0):
        """Scales and embeds an image into the PDF."""
        img_path = Path(image_path)
        if not img_path.exists():
            logger.warning("Image file not found for PDF inclusion: %s", img_path)
            return

        try:
            with PILImage.open(img_path) as pil_img:
                orig_w, orig_h = pil_img.size

            aspect = orig_h / float(orig_w) if orig_w > 0 else 0.5
            target_w = max_width
            target_h = target_w * aspect

            if target_h > max_height:
                target_h = max_height
                target_w = target_h / aspect

            rl_image = Image(str(img_path), width=target_w, height=target_h)
            self.story.append(Spacer(1, 4))
            self.story.append(rl_image)
            self.story.append(Spacer(1, 8))
        except Exception as e:
            logger.error("Failed to embed image in PDF (%s): %s", img_path, e)

    def add_markdown(self, md_content: str, workspace_dir: Optional[Union[str, Path]] = None):
        """Parses full Markdown content into ReportLab flowables."""
        lines = md_content.splitlines()
        in_table = False
        table_lines = []

        def flush_table():
            nonlocal in_table, table_lines
            if not table_lines:
                return
            try:
                headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
                rows = []
                for row_line in table_lines[2:]:
                    cols = [c.strip() for c in row_line.strip("|").split("|")]
                    if len(cols) == len(headers):
                        rows.append(cols)
                if rows:
                    df = pd.DataFrame(rows, columns=headers)
                    self.add_table(df)
            except Exception as ex:
                logger.warning("Failed to parse markdown table: %s", ex)
            table_lines = []
            in_table = False

        ws = Path(workspace_dir).resolve() if workspace_dir else Path.cwd()

        for line in lines:
            s_line = line.strip()

            if s_line.startswith("|") and s_line.endswith("|"):
                in_table = True
                table_lines.append(s_line)
                continue
            elif in_table:
                flush_table()

            if not s_line:
                continue

            if s_line.startswith("# "):
                self.add_heading(s_line[2:].strip(), level=1)
            elif s_line.startswith("## "):
                self.add_heading(s_line[3:].strip(), level=2)
            elif s_line.startswith("### "):
                self.add_heading(s_line[4:].strip(), level=2)
            elif s_line.startswith("---"):
                self.story.append(HRFlowable(width="100%", thickness=0.8, color=self.BORDER_COLOR, spaceBefore=6, spaceAfter=6))
            elif s_line.startswith("![") and "](" in s_line and s_line.endswith(")"):
                img_path_str = s_line.split("](")[1][:-1].strip()
                img_path = Path(img_path_str)
                if not img_path.is_absolute():
                    img_path = ws / img_path_str
                self.add_image(img_path)
            elif s_line.startswith("- ") or s_line.startswith("* "):
                text = s_line[2:].strip()
                if ":" in text:
                    k, v = text.split(":", 1)
                    self.add_bullet(k.strip(), v.strip())
                else:
                    self.add_paragraph(f"• {text}")
            elif s_line.startswith("> "):
                self.add_paragraph(f"<i>{s_line[2:].strip()}</i>")
            else:
                formatted_line = re.sub(r"\\*\\*(.*?)\\*\\*", r"<b>\\1</b>", s_line)
                formatted_line = re.sub(r"`(.*?)`", r"<font face=\"Courier\">\\1</font>", formatted_line)
                self.add_paragraph(formatted_line)

        if in_table:
            flush_table()

    def build(self, output_pdf_path: Union[str, Path]) -> Path:
        """Compiles and writes the PDF file to disk."""
        out_path = Path(output_pdf_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=letter,
            leftMargin=36,
            rightMargin=36,
            topMargin=36,
            bottomMargin=45
        )

        doc.build(self.story, canvasmaker=NumberedCanvas)
        logger.info("Compiled PDF Report -> %s", out_path)
        return out_path
