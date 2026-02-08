from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import datetime

class PDFReportEngine:
    def create_report(self, inputs, risk_score, clinical_text, stent_labels):
        """
        Generates a PDF report.
        Args:
            inputs: Dict of patient values.
            risk_score: Float (0-1).
            clinical_text: String (AI interpretation).
            stent_labels: Dict mapping 0/1/2 to strings.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom Styles
        title_style = styles['Title']
        header_style = styles['Heading2']
        body_style = styles['BodyText']
        
        elements = []

        # 1. Header & Timestamp
        elements.append(Paragraph("STEMI 3-Year Mortality Risk Assessment", title_style))
        elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        elements.append(Spacer(1, 20))
        
        # 2. Risk Score Highlight
        risk_percent = f"{risk_score:.1%}"
        risk_color = "red" if risk_score >= 0.147 else "green"
        risk_text = f"<font color='{risk_color}'><b>{risk_percent}</b></font>"
        
        elements.append(Paragraph(f"Predicted 3-Year Mortality Risk: {risk_text}", header_style))
        elements.append(Spacer(1, 12))

        # 3. Patient Data Table
        # Handle Stent Label Display
        stent_val = inputs['Stent for IRA']
        stent_display = stent_labels.get(stent_val, str(stent_val))

        data = [
            ["Parameter", "Value", "Parameter", "Value"],
            ["Age", f"{inputs['Age']} years", "Hemoglobin", f"{inputs['Hb']} g/L"],
            ["AST", f"{inputs['AST']} U/L", "Respiratory Supp.", "Yes" if inputs['Respiratory support'] else "No"],
            ["Beta Blocker", "Yes" if inputs['Beta blocker'] else "No", "Cardiotonics", "Yes" if inputs['Cardiotonics'] else "No"],
            ["Statins", "Yes" if inputs['Statins'] else "No", "Stent Strategy", stent_display]
        ]
        
        # Table Styling
        table = Table(data, colWidths=[100, 100, 100, 150])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.6)), # Header Blue
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

        # 4. Clinical Interpretation
        elements.append(Paragraph("AI Clinical Interpretation:", header_style))
        # Formatting: Replace markdown bold with HTML bold, handle newlines
        formatted_text = clinical_text.replace("**", "<b>").replace("**", "</b>").replace("\n", "<br/>")
        # Simple replace for second ** closing tag (naive approach for PDF)
        while "**" in formatted_text:
            formatted_text = formatted_text.replace("**", "<b>", 1).replace("**", "</b>", 1)
            
        elements.append(Paragraph(formatted_text, body_style))
        
        # 5. Disclaimer
        elements.append(Spacer(1, 40))
        disclaimer = "DISCLAIMER: For Research Use Only. This tool is based on the GBM model (AUC 0.801) and should not replace clinical judgment."
        elements.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', parent=body_style, fontSize=8, textColor=colors.grey)))

        doc.build(elements)
        buffer.seek(0)
        return buffer
