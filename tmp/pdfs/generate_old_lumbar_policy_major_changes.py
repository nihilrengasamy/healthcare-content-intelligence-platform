"""Generate a historical lumbar MRI policy with major differences."""

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
FILENAME = "Sample_Lumbar_MRI_Coverage_Policy_Old_Major_Changes.pdf"


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
    canvas.drawString(
        doc.leftMargin,
        LETTER[1] - 32,
        "Synthetic demo policy - no PHI - historical reference version",
    )
    canvas.drawRightString(LETTER[0] - doc.rightMargin, 24, f"Page {doc.page}")
    canvas.restoreState()


def policy_story() -> list:
    """Build the PDF story."""
    styles = build_styles()
    story: list = []

    story.extend(
        [
            Paragraph("Sample Lumbar Spine MRI Coverage Policy", styles["title"]),
            Paragraph(
                "Historical Restrictive Version for AI Version Comparison Demo",
                styles["subtitle"],
            ),
            Paragraph(
                "<b>Document Type:</b> Coverage Policy / Billing and Coding Policy",
                styles["meta"],
            ),
            Paragraph("<b>Policy ID:</b> DEMO-MRI-2024-001", styles["meta"]),
            Paragraph("<b>Effective Date:</b> January 1, 2024", styles["meta"]),
            Paragraph("<b>Review Date:</b> June 1, 2024", styles["meta"]),
            Paragraph("<b>Version:</b> 2024.2", styles["meta"]),
            Spacer(1, 0.12 * inch),
            Paragraph("Purpose", styles["section"]),
            Paragraph(
                "This historical demo policy reflects an older and more restrictive approach to lumbar spine MRI utilization management. It is intentionally designed to differ materially from the newer policy so comparison workflows can identify clinical, reimbursement, coding, and authorization changes.",
                styles["body"],
            ),
            Paragraph("Covered Service", styles["section"]),
            Paragraph(
                "Lumbar spine MRI may be considered for severe lumbar radiculopathy, suspected spinal stenosis with neurological compromise, post-surgical complication assessment, documented trauma, or confirmed malignancy workup. Routine use for uncomplicated low back pain is generally not covered.",
                styles["body"],
            ),
            Paragraph("Medical Necessity Criteria", styles["section"]),
            Paragraph(
                "• For uncomplicated low back pain, lumbar MRI is covered only after at least twelve weeks of conservative therapy.",
                styles["bullet"],
            ),
            Paragraph(
                "• Conservative therapy must include supervised physical therapy and physician-directed medication management unless contraindicated.",
                styles["bullet"],
            ),
            Paragraph(
                "• MRI before twelve weeks requires documented motor weakness, bowel or bladder dysfunction, suspected fracture, or known malignancy.",
                styles["bullet"],
            ),
            Paragraph(
                "• Repeat lumbar MRI is covered only when surgery is being planned or when new neurological deficit is documented.",
                styles["bullet"],
            ),
            Paragraph("Prior Authorization", styles["section"]),
            Paragraph(
                "Prior authorization is required for all outpatient and freestanding imaging requests. Emergency department imaging without prior authorization requires retrospective documentation review.",
                styles["body"],
            ),
        ]
    )

    story.append(PageBreak())

    coding_table = Table(
        [
            ["Type", "Code", "Description", "Coverage Note"],
            ["CPT", "72148", "MRI lumbar spine without contrast", "Covered when strict criteria are met"],
            ["CPT", "72149", "MRI lumbar spine with contrast", "Covered only for infection or malignancy evaluation"],
            ["ICD-10", "M54.16", "Radiculopathy, lumbar region", "Supported diagnosis"],
            ["ICD-10", "M48.061", "Spinal stenosis, lumbar region", "Supported diagnosis"],
        ],
        colWidths=[0.75 * inch, 0.85 * inch, 2.55 * inch, 1.95 * inch],
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
                "Claims must include a valid CPT procedure code, supported ICD-10 diagnosis code, ordering provider information, prior authorization number, and documentation of completed conservative therapy. HCPCS contrast agent billing is not separately payable under this version.",
                styles["body"],
            ),
            coding_table,
            Spacer(1, 0.16 * inch),
            Paragraph("Excluded Services", styles["section"]),
            Paragraph(
                "• Screening MRI for nonspecific low back pain without objective neurological findings is not covered.",
                styles["bullet"],
            ),
            Paragraph(
                "• Repeat imaging within twenty-four months for the same indication is not covered unless surgery is planned.",
                styles["bullet"],
            ),
            Paragraph(
                "• Contrast-enhanced MRI for uncomplicated radiculopathy is not covered.",
                styles["bullet"],
            ),
            Paragraph(
                "• Experimental imaging protocols and out-of-network mobile MRI services are not covered.",
                styles["bullet"],
            ),
            Paragraph("Frequency Limit", styles["section"]),
            Paragraph(
                "One lumbar spine MRI is allowed per twenty-four-month period for the same clinical indication unless there is new trauma, confirmed malignancy, or planned surgical intervention.",
                styles["body"],
            ),
            Paragraph("Required Documentation", styles["section"]),
            Paragraph(
                "• History and physical examination with symptom duration and pain severity score.",
                styles["bullet"],
            ),
            Paragraph(
                "• Provider attestation confirming failure of conservative therapy for at least twelve weeks.",
                styles["bullet"],
            ),
            Paragraph(
                "• Physical therapy notes or equivalent supervised treatment documentation.",
                styles["bullet"],
            ),
            Paragraph(
                "• Neurological examination findings for any request submitted before twelve weeks.",
                styles["bullet"],
            ),
            Paragraph(
                "• Prior imaging report and operative planning note for repeat MRI requests.",
                styles["bullet"],
            ),
        ]
    )

    story.append(PageBreak())

    story.extend(
        [
            Paragraph("Synthetic Contract Terms", styles["section"]),
            Paragraph(
                "For demonstration only, the allowed amount for CPT 72148 is $620 when performed by an in-network freestanding imaging provider. Member cost share may include a $75 copay and 30% coinsurance, subject to benefit plan design. Out-of-network lumbar spine MRI reimbursement is not eligible under this historical version except for documented emergency care.",
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
                Paragraph("M54.16, CPT 72148, twelve weeks therapy, prior authorization present", table_text),
                Paragraph("Approve", table_text),
                Paragraph("Coverage and authorization criteria satisfied.", table_text),
            ],
            [
                Paragraph("Low back pain, six weeks therapy, no red flags", table_text),
                Paragraph("Deny", table_text),
                Paragraph("Conservative therapy requirement not met.", table_text),
            ],
            [
                Paragraph("Progressive neurological deficit, no prior authorization", table_text),
                Paragraph("Manual review", table_text),
                Paragraph("Urgent exception may apply, but retrospective review is required.", table_text),
            ],
        ],
        colWidths=[3.45 * inch, 1.35 * inch, 2.0 * inch],
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
                "Coverage determinations should comply with applicable federal and state requirements, medical necessity review standards, audit documentation policies, and member benefit language. This synthetic historical policy contains no PHI and is intended only for software demonstration.",
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
