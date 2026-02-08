# -*- coding: utf-8 -*-
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, mm
import datetime
import textwrap
import io

class PDFReportEngine:
    def __init__(self, buffer, patient_data, predict_result, nlg_report):
        self.buffer = buffer
        self.data = patient_data
        self.res = predict_result  # {'prob': float, 'threshold': float, 'risk_label': str}
        self.text = nlg_report
        
        self.c = canvas.Canvas(self.buffer, pagesize=A4)
        self.width, self.height = A4
        self.margin = 20 * mm
        self.y = self.height - self.margin
        
        self.hospital_name = "Hubei STEMI Multicenter Cohort AI"
        self.system_name = "STEMI 3-Year Mortality Predictor (GBM)"

    def _draw_header(self):
        # Blue Header Bar
        self.c.setFillColor(colors.HexColor("#003366"))
        self.c.rect(0, self.height - 30*mm, self.width, 30*mm, fill=1, stroke=0)
        
        # Title
        self.c.setFillColor(colors.white)
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawString(self.margin, self.height - 15*mm, "STEMI Post-PCI Risk Assessment")
        
        # Subtitle / Time
        self.c.setFont("Helvetica", 10)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.c.drawString(self.margin, self.height - 22*mm, f"Report Date: {date_str} | ID: {int(datetime.datetime.now().timestamp())}")

    def _draw_patient_table(self):
        self.y -= 40 * mm
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "1. Patient Clinical Profile")
        self.y -= 8 * mm
        
        # Define Data (8 Features)
        # Handle Stent Display logic
        stent_raw = self.data.get('Stent for IRA', 0)
        stent_map = {0: "Type 0 (No Stent/Other)", 1: "Type 1 (DES)", 2: "Type 2 (BMS)"}
        stent_str = stent_map.get(stent_raw, f"Type {stent_raw}")

        params = [
            ("Age", f"{self.data.get('Age', 'N/A')} years"),
            ("Hemoglobin (Hb)", f"{self.data.get('Hb', 'N/A')} g/L"),
            ("AST", f"{self.data.get('AST', 'N/A')} U/L"),
            ("Respiratory Supp.", "Yes" if self.data.get('Respiratory support') else "No"),
            ("Beta Blocker", "Yes" if self.data.get('Beta blocker') else "No"),
            ("Cardiotonics", "Yes" if self.data.get('Cardiotonics') else "No"),
            ("Statins", "Yes" if self.data.get('Statins') else "No"),
            ("Stent Strategy", stent_str),
        ]
        
        # Draw Grid
        row_height = 8 * mm
        col_width = 80 * mm
        x_left = self.margin
        x_right = self.margin + col_width
        
        self.c.setFont("Helvetica", 10)
        for i, (label, value) in enumerate(params):
            # Alternate background
            if i % 2 == 0:
                self.c.setFillColor(colors.HexColor("#f0f0f0"))
                self.c.rect(x_left, self.y - row_height + 2*mm, col_width*2, row_height, fill=1, stroke=0)
            
            self.c.setFillColor(colors.black)
            # Left Column
            if i < 4:
                cur_y = self.y - (i * row_height)
                self.c.drawString(x_left + 2*mm, cur_y, label)
                self.c.drawRightString(x_left + col_width - 5*mm, cur_y, str(value))
            # Right Column (adjust logic if you want 2 columns, currently listing vertically is cleaner for 8 items)
            else:
                # Let's actually do 2 columns: 4 rows x 2 cols
                pass

        # Re-implementing as 2-column layout for compactness
        self.y += 8*mm # Reset slightly
        col_w = 85 * mm
        row_h = 10 * mm
        
        for i in range(0, 8, 2): # 0, 2, 4, 6
            self.y -= row_h
            
            # Left Item
            l_key, l_val = params[i]
            self.c.drawString(self.margin + 2*mm, self.y, f"{l_key}:")
            self.c.setFont("Helvetica-Bold", 10)
            self.c.drawString(self.margin + 40*mm, self.y, str(l_val))
            self.c.setFont("Helvetica", 10)
            
            # Right Item
            if i+1 < len(params):
                r_key, r_val = params[i+1]
                self.c.drawString(self.margin + col_w + 2*mm, self.y, f"{r_key}:")
                self.c.setFont("Helvetica-Bold", 10)
                self.c.drawString(self.margin + col_w + 40*mm, self.y, str(r_val))
                self.c.setFont("Helvetica", 10)
                
            # Line
            self.c.setStrokeColor(colors.lightgrey)
            self.c.line(self.margin, self.y - 2*mm, self.width - self.margin, self.y - 2*mm)

    def _draw_result(self):
        self.y -= 15 * mm
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "2. Risk Stratification")
        self.y -= 10 * mm
        
        # Risk Box
        prob = self.res['prob']
        threshold = self.res['threshold']
        is_high = prob >= threshold
        
        box_color = colors.HexColor("#f8d7da") if is_high else colors.HexColor("#d4edda")
        text_color = colors.HexColor("#721c24") if is_high else colors.HexColor("#155724")
        
        self.c.setFillColor(box_color)
        self.c.roundRect(self.margin, self.y - 15*mm, self.width - 2*self.margin, 20*mm, 4, fill=1, stroke=0)
        
        self.c.setFillColor(text_color)
        self.c.setFont("Helvetica-Bold", 16)
        risk_text = f"Predicted 3-Year Mortality Risk: {prob:.1%}"
        self.c.drawCentredString(self.width/2, self.y - 8*mm, risk_text)
        
        self.c.setFont("Helvetica", 10)
        sub_text = f"Risk Group: {self.res['risk_label']} (Cut-off Value: {threshold:.1%})"
        self.c.drawCentredString(self.width/2, self.y - 13*mm, sub_text)

    def _draw_interpretation(self):
        self.y -= 25 * mm
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "3. AI Clinical Analysis")
        self.y -= 8 * mm
        
        self.c.setFont("Helvetica", 10)
        line_height = 5 * mm
        max_width = 90 # chars
        
        # Cleanup Markdown
        clean_text = self.text.replace("### ", "").replace("**", "").replace("#### ", "")
        
        lines = []
        for paragraph in clean_text.split('\n'):
            wrapped = textwrap.wrap(paragraph, width=max_width)
            if not wrapped:
                lines.append("")
            else:
                lines.extend(wrapped)
        
        for line in lines:
            if self.y < self.margin + 10*mm:
                self.c.showPage()
                self._draw_header()
                self.y = self.height - 50*mm
                self.c.setFont("Helvetica", 10)
            
            self.c.drawString(self.margin, self.y, line)
            self.y -= line_height

    def _draw_footer(self):
        self.c.saveState()
        self.c.setFont("Helvetica-Oblique", 8)
        self.c.setFillColor(colors.grey)
        footer_text = f"Generated by {self.system_name}. For Research Use Only."
        self.c.drawCentredString(self.width / 2, 10*mm, footer_text)
        self.c.restoreState()

    def generate(self):
        self._draw_header()
        self._draw_patient_table()
        self._draw_result()
        self._draw_interpretation()
        self._draw_footer()
        self.c.save()
        self.buffer.seek(0)
        return self.buffer
