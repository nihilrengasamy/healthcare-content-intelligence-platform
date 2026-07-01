from docx import Document
from pathlib import Path
path = Path(r"C:\Users\Nihil Rengasamy\Documents\Codex\2026-06-27\hi\outputs\healthcare-content-intelligence\Cotiviti_AI_Healthcare_Content_Intelligence_Report.docx")
doc = Document(path)
for i, p in enumerate(doc.paragraphs[:25], start=1):
    text = p.text.strip()
    if text:
        print(f"{i}: {text}")
