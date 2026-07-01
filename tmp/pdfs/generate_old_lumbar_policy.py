"""Generate an older version of the sample lumbar MRI coverage policy PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "pdf"
DOWNLOADS_DIR = Path.home() / "Downloads"
FILENAME = "Sample_Lumbar_MRI_Coverage_Policy_Old.pdf"


def build_styles() -> dict[str, ParagraphStyle]:
    """Return document styles."""
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PolicyTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "PolicySubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=colors.HexColor("#555555"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "meta": ParagraphStyle(
            "PolicyMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "PolicySection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.HexColor("#16324f"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "PolicyBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "PolicyBullet",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "PolicyTable",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=9.5,
            spaceAfter=0,
        ),
    }


def add_header_footer(canvas, doc) -> None:  # noqa: ANN001
    """Draw repeated header and footer."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, LETTER[1] - 32, "Synthetic demo policy - no PHI - for AI platform testing only")
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 24, f"Page {doc.page}")
    canvas.restoreState()


def policy_story() -> list:
    """Build the PDF story."""
    styles = build_styles()
    story: list = []

    story.extend(
        [
            Paragraph("Sample Lumbar Spine MRI Coverage Policy", styles["title"]),
            Paragraph("Historical Reference Version for Version Comparison Demo", styles["subtitle"]),
            Paragraph("<b>Document Type:</b> Coverage Policy / Billing and Coding Policy", styles["meta"]),
            Paragraph("<b>Policy ID:</b> DEMO-MRI-2025-001", styles["meta"]),
            Paragraph("<b>Effective Date:</b> January 1, 2025", styles["meta"]),
            Paragraph("<b>Review Date:</b> June 1, 2025", styles["meta"]),
            Paragraph("<b>Version:</b> 2025.1", styles["meta"]),
            Spacer(1, 0.12 * inch),
            Paragraph("Purpose", styles["section"]),
            Paragraph(
                "This historical demo policy describes earlier coverage criteria, billing rules, documentation expectations, and prior authorization requirements for lumbar spine magnetic resonance imaging. It is intentionally more restrictive than the newer version so AI comparison workflows can detect meaningful policy change.",
                styles["body"],
            ),
            Paragraph("Covered Service", styles["section"]),
            Paragraph(
                "Lumbar spine MRI may be covered when medically necessary for lumbar radiculopathy, suspected spinal stenosis, progressive neurological deficit, trauma, malignancy, or post-surgical complications. Coverage for uncomplicated low back pain remains limited unless conservative management requirements are met.",
                styles["body"],
            ),
            Paragraph("Medical Necessity Criteria", styles["section"]),
            Paragraph(
                "• For uncomplicated low back pain, lumbar MRI is covered only after at least eight weeks of conservative therapy.",
                styles["bullet"],
            ),
            Paragraph(
                "• Conservative therapy should include physical therapy, medication management, activity modification, and provider-directed home exercise when clinically appropriate.",
                styles["bullet"],
            ),
            Paragraph(
                "• MRI may be approved before eight weeks only when progressive neurological deficit, suspected fracture, malignancy, or severe trauma is documented.",
                styles["bullet"],
            ),
            Paragraph(
                "• Repeat lumbar MRI is covered only when a documented change in symptoms is present and results are expected to alter treatment planning.",
                styles["bullet"],
            ),
            Paragraph("Prior Authorization", styles["section"]),
            Paragraph(
                "Prior authorization is required for all non-emergent outpatient lumbar spine MRI requests. Emergency department imaging may proceed without prior authorization only when urgent neurological compromise or major trauma is documented.",
                styles["body"],
            ),
        ]
    )

    story.append(PageBreak())

    coding_table = Table(
        [
            ["Type", "Code", "Description", "Coverage Note"],
            ["CPT", "72148", "MRI lumbar spine without contrast", "Covered when criteria are met"],
            ["CPT", "72149", "MRI lumbar spine with contrast", "Covered with infection, tumor, or post-operative indication"],
            ["ICD-10", "M54.16", "Radiculopathy, lumbar region", "Supported diagnosis"],
            ["ICD-10", "M48.061", "Spinal stenosis, lumbar region", "Supported diagnosis"],
            ["HCPCS", "A9575", "Injection, gadoterate meglumine", "Covered when contrast is medically necessary"],
        ],
        colWidths=[0.75 * inch, 0.85 * inch, 2.45 * inch, 2.05 * inch],
        repeatRows=1,
    )
    coding_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa9b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ]
        )
    )

    story.extend(
        [
            Paragraph("Billing and Coding Requirements", styles["section"]),
            Paragraph(
                "Claims must include a valid CPT or HCPCS procedure code, a supported ICD-10 diagnosis code, ordering provider information, rendering provider information, and documentation supporting medical necessity.",
                styles["body"],
            ),
            coding_table,
            Spacer(1, 0.16 * inch),
            Paragraph("Excluded Services", styles["section"]),
            Paragraph(
                "• Screening MRI for nonspecific low back pain without red flags is not covered.",
                styles["bullet"],
            ),
            Paragraph(
                "• Repeat imaging without a documented change in symptoms is not covered.",
                styles["bullet"],
            ),
            Paragraph(
                "• MRI requested solely for pre-employment, disability, or legal documentation review is not covered.",
                styles["bullet"],
            ),
            Paragraph("Frequency Limit", styles["section"]),
            Paragraph(
                "One lumbar spine MRI is allowed per eighteen-month period for the same clinical indication unless new trauma, suspected malignancy, or significant worsening neurological findings are documented.",
                styles["body"],
            ),
            Paragraph("Required Documentation", styles["section"]),
            Paragraph("• History and physical examination with date of symptom onset.", styles["bullet"]),
            Paragraph("• Clinical indication and suspected diagnosis.", styles["bullet"]),
            Paragraph("• Duration and type of conservative therapy when applicable.", styles["bullet"]),
            Paragraph("• Neurological examination findings when urgent imaging is requested.", styles["bullet"]),
            Paragraph("• Prior imaging results when repeat MRI is requested.", styles["bullet"]),
        ]
    )

    story.append(PageBreak())

    story.extend(
        [
            Paragraph("Synthetic Contract Terms", styles["section"]),
            Paragraph(
                "For demonstration only, the allowed amount for CPT 72148 is $700 when performed by an in-network freestanding imaging provider. Member cost share may include a $60 copay and 25% coinsurance, subject to benefit plan design.",
                styles["body"],
            ),
            Paragraph("Decision Examples", styles["section"]),
        ]
    )

    table_text = styles["table"]
    examples_table = Table(
        [
            ["Scenario", "Expected Policy Outcome", "Reason"],
            [
                Paragraph("M54.16, CPT 72148, eight weeks therapy, prior authorization present", table_text),
                Paragraph("Approve", table_text),
                Paragraph("Coverage and authorization criteria satisfied.", table_text),
            ],
            [
                Paragraph("Low back pain, four weeks therapy, no red flags", table_text),
                Paragraph("Deny", table_text),
                Paragraph("Conservative therapy requirement not met.", table_text),
            ],
            [
                Paragraph("Progressive neurological deficit, no prior authorization", table_text),
                Paragraph("Manual review", table_text),
                Paragraph("Urgent exception may apply, but documentation must support expedited imaging.", table_text),
            ],
        ],
        colWidths=[3.55 * inch, 1.25 * inch, 2.0 * inch],
        repeatRows=1,
    )
    examples_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#16324f")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa9b8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ]
        )
    )

    story.extend(
        [
            examples_table,
            Spacer(1, 0.16 * inch),
            Paragraph("Regulatory and Audit Note", styles["section"]),
            Paragraph(
                "Coverage determinations should comply with applicable federal and state requirements, medical necessity review standards, audit documentation policies, and member benefit language. This synthetic policy contains no PHI and is intended only for software demonstration.",
                styles["body"],
            ),
        ]
    )
    return story


def build_pdf(destination: Path) -> None:
    """Build the PDF at the given destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.55 * inch,
    )
    document.build(policy_story(), onFirstPage=add_header_footer, onLaterPages=add_header_footer)


def main() -> None:
    """Generate both repo and Downloads copies."""
    repo_pdf = OUTPUT_DIR / FILENAME
    downloads_pdf = DOWNLOADS_DIR / FILENAME
    build_pdf(repo_pdf)
    build_pdf(downloads_pdf)
    print(repo_pdf)
    print(downloads_pdf)


if __name__ == "__main__":
    main()
