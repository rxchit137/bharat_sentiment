import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import os

def generate_pdf_report(json_report_path: str, sentiment_chart_path: str, language_chart_path: str) -> str:
    """Generates a PDF report from a JSON report file."""
    with open(json_report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pdf_path = os.path.splitext(json_report_path)[0] + ".pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Sentiment Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Summary
    summary_text = f"<b>Source File:</b> {data['source_file']}<br/>"
    summary_text += f"<b>Total Texts Analyzed:</b> {data['total_texts']}"
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Charts ---
    story.append(Image(sentiment_chart_path, width=6*inch, height=4*inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Image(language_chart_path, width=6*inch, height=4*inch))
    story.append(Spacer(1, 0.2 * inch))

    # --- Tables ---
    # Keywords Table
    story.append(Paragraph("Top Keywords by Sentiment", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))

    keywords_data = data["top_keywords"]
    header = list(keywords_data.keys())

    # Get all keywords and arrange them by column
    max_rows = max(len(v) for v in keywords_data.values()) if keywords_data else 0
    table_data = [header]
    for i in range(max_rows):
        row = [keywords_data.get(h, [''])[i] if i < len(keywords_data.get(h, [])) else '' for h in header]
        table_data.append(row)

    keyword_table = Table(table_data)
    keyword_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(keyword_table)

    # Build the PDF
    doc.build(story)

    return pdf_path
