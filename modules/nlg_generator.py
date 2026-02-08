# -*- coding: utf-8 -*-
import datetime
import numpy as np

class ClinicalReportGenerator:
    """
    STEMI Specific Clinical Report Generator (GBM Model)
    Generates natural language interpretation based on 8 risk factors.
    """

    def __init__(self, patient_data, prob, threshold, shap_values=None, feature_names=None, base_value=None):
        self.data = patient_data
        self.prob = prob
        self.threshold = threshold
        # SHAP 值用于高级解释 (可选)
        self.shap_values = shap_values 
        self.feature_names = feature_names
        
    def _get_risk_level_desc(self):
        # 基于 0.147 阈值的细分描述
        if self.prob < self.threshold * 0.5: return "Very Low Risk"
        elif self.prob < self.threshold: return "Low Risk"
        elif self.prob < self.threshold * 2.0: return "High Risk"
        else: return "Critical Risk"

    def _generate_clinical_advice(self):
        """
        核心逻辑引擎：根据手稿 Discussion 章节生成建议
        """
        advice = []
        inputs = self.data
        
        # --- 1. 危重症指标 (Critical Indicators) ---
        if inputs.get('Respiratory support') == 1:
            advice.append("• **Critical Warning (Respiratory)**: Patient requires mechanical ventilation. This is a strong marker of critical illness (e.g., cardiogenic shock, ARDS) and is independently associated with a >2-fold increase in mortality risk.")
            
        if inputs.get('Cardiotonics') == 1:
            advice.append("• **Hemodynamic Instability**: Usage of inotropes/cardiotonics indicates severe LV systolic dysfunction or persistent hypoperfusion, portending a poor long-term prognosis.")

        # --- 2. 实验室与生命体征 (Labs & Vitals) ---
        ast_val = inputs.get('AST', 0)
        if ast_val > 100: # 这里的 100 是基于常见临床异常值，手稿中 non-survivor 中位数为 131
            advice.append(f"• **Hepatic/Ischemic Injury**: Elevated AST ({ast_val} U/L) may reflect hypoxic liver injury (shock liver) due to systemic hypoperfusion or large infarct size.")
            
        hb_val = inputs.get('Hb', 0)
        if hb_val < 130: # 贫血标准
            advice.append(f"• **Anemia Risk**: Hemoglobin ({hb_val} g/L) is below normal (<130 g/L). Anemia reduces myocardial oxygen delivery and aggravates ischemic injury.")

        age_val = inputs.get('Age', 0)
        if age_val > 72: # 手稿中死亡组中位年龄 72
            advice.append(f"• **Age Factor**: Advanced age ({age_val} years) is a significant non-modifiable risk factor associated with complex vascular conditions and comorbidities.")

        # --- 3. 药物与治疗 (Medication & Procedure) ---
        # 缺失保护性药物的警告
        if inputs.get('Beta blocker') == 0:
            advice.append("• **Therapy Gap (Beta-blockers)**: Beta-blocker not prescribed. Unless contraindicated, initiation is recommended to reduce myocardial oxygen demand and prevent arrhythmias.")
            
        if inputs.get('Statins') == 0:
            advice.append("• **Therapy Gap (Statins)**: Statin therapy missing. High-intensity statins are crucial for plaque stabilization and reducing recurrent cardiovascular events.")
            
        # 支架策略
        stent_val = inputs.get('Stent for IRA', 0)
        if stent_val == 0:
            advice.append("• **Procedural Strategy**: No stent implantation ('Type 0') recorded for the infarct-related artery. Lack of stenting may be associated with higher rates of reocclusion compared to modern stenting strategies.")

        # 兜底逻辑
        if not advice:
            advice.append("• No specific high-risk red flags identified among the 8 key predictors. Continue standard guideline-directed medical therapy (GDMT).")
            
        return "\n\n".join(advice)

    def generate_full_report(self):
        """生成 Markdown 格式的完整报告内容"""
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        risk_desc = self._get_risk_level_desc()
        advice_text = self._generate_clinical_advice()
        
        # 构造 Markdown 文本
        report = f"""
### STEMI 3-Year Prognostic Report
**Session ID**: `{session_id}`  |  **Date**: {current_time}

---
#### 1. Risk Stratification
- **Predicted Mortality Risk**: <span style="color:{'#dc3545' if self.prob >= self.threshold else '#28a745'}">**{self.prob:.1%}**</span>
- **Risk Classification**: **{risk_desc}** (Cut-off: {self.threshold:.1%})

#### 2. Clinical Interpretation & Recommendations
{advice_text}

---
#### 3. Patient Parameters
| Parameter | Value | Parameter | Value |
| :--- | :--- | :--- | :--- |
| **Age** | {self.data.get('Age')} years | **Hemoglobin** | {self.data.get('Hb')} g/L |
| **AST** | {self.data.get('AST')} U/L | **Respiratory Supp.** | {"Yes" if self.data.get('Respiratory support') else "No"} |
| **Beta Blocker** | {"Yes" if self.data.get('Beta blocker') else "No"} | **Cardiotonics** | {"Yes" if self.data.get('Cardiotonics') else "No"} |
| **Statins** | {"Yes" if self.data.get('Statins') else "No"} | **Stent Strategy** | Type {self.data.get('Stent for IRA')} |

> **Disclaimer**: This AI tool is for research use only. The prediction is based on the GBM model (AUC 0.801) validated in the multicenter STEMI cohort.
"""
        return report
