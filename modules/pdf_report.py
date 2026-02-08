# -*- coding: utf-8 -*-
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import datetime

class PDFReportEngine:
    def __init__(self, buffer, patient_data, predict_result, nlg_report):
        self.buffer = buffer
        self.data = patient_data
        self.res = predict_result
        self.text = nlg_report
        
        self.c = canvas.Canvas(self.buffer, pagesize=A4)
        self.width, self.height = A4
        self.margin = 20 * mm
        self.y = self.height - 20 * mm
        
        self.hospital_name = "Hubei STEMI Multicenter Cohort AI"
        self.system_name = "STEMI 3-Year Mortality Predictor (GBM)"

    def _draw_header(self):
        # 顶部深蓝背景条
        self.c.setFillColor(colors.HexColor("#003366"))
        self.c.rect(0, self.height - 30*mm, self.width, 30*mm, fill=1, stroke=0)
        
        # 标题文字
        self.c.setFillColor(colors.white)
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawString(self.margin, self.height - 15*mm, "STEMI Post-PCI Risk Assessment")
        
        # 副标题/时间
        self.c.setFont("Helvetica", 10)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.c.drawString(self.margin, self.height - 22*mm, f"Report Date: {date_str}")
        
        # 下移坐标指针
        self.y -= 35 * mm

    def _draw_patient_table(self):
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "1. Patient Clinical Profile")
        self.y -= 10 * mm
        
        # === 构造表格数据 ===
        # 格式化 Stent 显示
        stent_val = self.data.get('Stent for IRA', 0)
        stent_str = f"Type {stent_val}" 
        
        # 每一行包含两对 Key-Value
        table_data = [
            # Header Row (Optional, removed for cleaner look)
            ["Parameter", "Value", "Parameter", "Value"],
            
            ["Age", f"{self.data.get('Age')} years", 
             "Hemoglobin", f"{self.data.get('Hb')} g/L"],
             
            ["AST", f"{self.data.get('AST')} U/L", 
             "Respiratory Supp.", "Yes" if self.data.get('Respiratory support') else "No"],
             
            ["Beta Blocker", "Yes" if self.data.get('Beta blocker') else "No", 
             "Cardiotonics", "Yes" if self.data.get('Cardiotonics') else "No"],
             
            ["Statins", "Yes" if self.data.get('Statins') else "No", 
             "Stent Strategy", stent_str]
        ]
        
        # === 表格样式 ===
        # 列宽设置：Key(40mm), Value(45mm), Key(40mm), Value(45mm)
        col_widths = [40*mm, 45*mm, 40*mm, 45*mm]
        
        t = Table(table_data, colWidths=col_widths, rowHeights=8*mm)
        
        t.setStyle(TableStyle([
            # 表头背景
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            
            # 第一列和第三列（Label列）加粗
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 1), (2, -1), 'Helvetica-Bold'),
            
            # 网格线
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # 字体大小
            ('FONTSIZE', (0, 0), (-1, -1), 10),
        ]))
        
        # 将表格绘制到 Canvas 上
        w, h = t.wrapOn(self.c, self.width, self.height)
        t.drawOn(self.c, self.margin, self.y - h)
        
        # 更新 y 坐标
        self.y -= (h + 15*mm)

    def _draw_result(self):
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "2. Risk Stratification")
        self.y -= 8 * mm
        
        prob = self.res['prob']
        threshold = self.res['threshold']
        is_high = prob >= threshold
        
        # 绘制结果框
        box_color = colors.HexColor("#f8d7da") if is_high else colors.HexColor("#d4edda")
        border_color = colors.HexColor("#dc3545") if is_high else colors.HexColor("#28a745")
        
        self.c.setStrokeColor(border_color)
        self.c.setFillColor(box_color)
        self.c.roundRect(self.margin, self.y - 15*mm, self.width - 2*self.margin, 15*mm, 4, fill=1, stroke=1)
        
        # 绘制文字
        text_color = colors.HexColor("#721c24") if is_high else colors.HexColor("#155724")
        self.c.setFillColor(text_color)
        
        self.c.setFont("Helvetica-Bold", 12)
        res_text = f"Predicted 3-Year Mortality Risk: {prob:.1%}  |  Group: {self.res['risk_label'].upper()}"
        self.c.drawCentredString(self.width/2, self.y - 10*mm, res_text)
        
        self.c.setFont("Helvetica", 9)
        self.c.drawCentredString(self.width/2, self.y - 14*mm, f"(Cut-off Value: {threshold:.1%})")
        
        self.y -= 25 * mm

    def _draw_interpretation(self):
        self.c.setFillColor(colors.black)
        self.c.setFont("Helvetica-Bold", 14)
        self.c.drawString(self.margin, self.y, "3. AI Clinical Analysis")
        self.y -= 5 * mm
        
        # 使用 Paragraph 处理自动换行
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        style.fontName = "Helvetica"
        style.fontSize = 10
        style.leading = 14  # 行间距
        
        # 清理 Markdown 标记，转为简单的文本
        clean_text = self.text.replace("**", "").replace("•", "<br/>•")
        
        p = Paragraph(clean_text, style)
        w, h = p.wrapOn(self.c, self.width - 2*self.margin, self.height)
        p.drawOn(self.c, self.margin, self.y - h)
        
        self.y -= (h + 10*mm)

    def _draw_footer(self):
        self.c.saveState()
        self.c.setFont("Helvetica-Oblique", 8)
        self.c.setFillColor(colors.grey)
        footer_text = f"Generated by {self.system_name}. For Research Use Only. Not for clinical diagnosis."
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
