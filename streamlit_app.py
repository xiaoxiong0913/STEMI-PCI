import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import io
import matplotlib.pyplot as plt  # 修复：必须显式导入 matplotlib
import shap
import plotly.graph_objects as go
import datetime

# ================= 1. 引用自定义模块 =================
# 确保您的 modules 文件夹下有这些文件
from modules.database import PatientDatabase
from modules.nlg_generator import ClinicalReportGenerator
from modules.pdf_report import PDFReportEngine
from modules.batch_processor import BatchProcessor
from modules.analytics import AnalyticsEngine

# ================= 2. 系统初始化与配置 =================
st.set_page_config(
    page_title="STEMI Mortality Risk Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载外部 CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass
    
    # 内置备用样式，防止 CSS 文件丢失导致界面丑陋
    st.markdown("""
    <style>
        html, body, [class*="css"] { font-family: 'Helvetica Neue', sans-serif; font-size: 18px; }
        .overview-card { 
            background-color: #ffffff; padding: 20px; border-radius: 8px; 
            border-left: 6px solid #dc3545; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;
        }
        .stButton>button { width: 100%; height: 3.5em; font-weight: bold; font-size: 1.2rem; margin-top: 15px; }
        .stNumberInput label, .stSelectbox label { font-weight: 600; font-size: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# 尝试加载 assets/style.css
local_css(os.path.join("assets", "style.css"))

# ================= 3. 资源加载 =================
@st.cache_resource
def load_system():
    # 优先查找根目录，其次查找 assets
    paths = [".", "assets"]
    model, scaler = None, None
    
    for p in paths:
        m_path = os.path.join(p, "gbm_model.pkl")
        s_path = os.path.join(p, "scaler.pkl")
        if os.path.exists(m_path) and os.path.exists(s_path):
            try:
                with open(m_path, 'rb') as f: model = pickle.load(f)
                with open(s_path, 'rb') as f: scaler = pickle.load(f)
                break # 找到即停止
            except Exception as e:
                st.error(f"Error loading files from {p}: {e}")
    
    return model, scaler

model, scaler = load_system()
db = PatientDatabase()

# === STEMI 核心配置 ===
THRESHOLD = 0.147  # Cut-off value (Youden Index)

# 支架类型定义
STENT_LABELS = {
    0: "Type 0 (No Stent/POBA)", 
    1: "Type 1 (DES)", 
    2: "Type 2 (BMS/Other)"
}

# ================= 4. 侧边栏导航 =================
with st.sidebar:
    # 尝试显示图标，如果网络不通则显示文字
    try:
        st.image("https://img.icons8.com/color/96/heart-monitor.png", width=80)
    except:
        st.write("❤️")
        
    st.header("STEMI Risk AI")
    page = st.radio("Navigation", ["Risk Assessment", "Batch Analysis", "Clinical Dashboard", "Project Introduction"])
    st.markdown("---")
    
    if model and scaler:
        st.success("System Online (GBM)")
    else:
        st.error("System Offline (Missing .pkl files)")
        st.info("Please verify 'gbm_model.pkl' and 'scaler.pkl' are in the root directory.")

# ================= 5. 页面路由逻辑 =================

# ----------------- PAGE 1: 风险评估 -----------------
if page == "Risk Assessment":
    
    # 1. 顶部卡片
    st.markdown(f"""
    <div class='overview-card'>
        <h3 style='margin-bottom:10px; margin-top:0;'>3-Year Mortality Prediction for STEMI Patients (Post-PCI)</h3>
        <h4 style='margin-bottom:10px; color:#555;'>Model Overview</h4>
        <p style='font-size:16px; line-height:1.5;'>
            This tool uses a <b>Gradient Boosting Machine (GBM)</b> model validated on a multicenter cohort.<br>
            - <b>AUC: 0.801</b> (External Validation)<br>
            - <b>Risk Threshold: {THRESHOLD:.1%}</b> (Patients ≥ {THRESHOLD:.1%} are classified as High Risk)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2. 输入表单 (8 Features)
    st.markdown("##### Patient Clinical Data")
    with st.form("input_form_stemi"):
        # 第一行：3个连续变量
        c1, c2, c3 = st.columns(3)
        with c1: age = st.number_input("Age (years)", 20, 100, 65)
        with c2: hb = st.number_input("Hemoglobin (g/L)", 30, 200, 130, help="Normal range: 120-160")
        with c3: ast = st.number_input("AST (U/L)", 5, 2000, 40)

        # 第二行：药物与治疗
        c4, c5, c6 = st.columns(3)
        with c4: beta = st.selectbox("Beta Blocker Use", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with c5: statins = st.selectbox("Statins Use", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with c6: cardio = st.selectbox("Cardiotonics (Inotropes)", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

        # 第三行：临床状态与手术
        c7, c8, c9 = st.columns(3)
        with c7: resp = st.selectbox("Respiratory Support", [0, 1], format_func=lambda x: "Yes (Ventilation)" if x==1 else "No")
        with c8: stent = st.selectbox("Stent for IRA", [0, 1, 2], format_func=lambda x: STENT_LABELS.get(x, f"Type {x}"))
        with c9: st.write("") 

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 CALCULATE RISK", type="primary")

    if submitted and model and scaler:
        # 构造输入字典
        inputs = {
            'Age': age, 'Hb': hb, 'AST': ast,
            'Respiratory support': resp, 'Beta blocker': beta,
            'Cardiotonics': cardio, 'Statins': statins, 'Stent for IRA': stent
        }
        
        try:
            # === 核心预测逻辑 (混合预处理) ===
            # 1. 连续变量标准化 [Age, Hb, AST]
            cont_features = np.array([[age, hb, ast]])
            cont_scaled = scaler.transform(cont_features)
            
            # 2. 分类变量保持原始值 (必须严格匹配训练顺序)
            # 顺序: Respiratory, Beta, Cardio, Statins, Stent
            cat_features = np.array([[resp, beta, cardio, statins, stent]])
            
            # 3. 拼接 (3+5=8 features)
            final_input = np.hstack((cont_scaled, cat_features))
            
            # 4. 预测
            prob = model.predict_proba(final_input)[:, 1][0]
            risk_label = "High Risk" if prob >= THRESHOLD else "Low Risk"
            
            # 5. 存入数据库
            db.add_record(inputs, prob, risk_label)
            
        except Exception as e:
            st.error(f"Computation Error: {e}")
            st.stop()

        st.divider()
        st.subheader("Prediction Results")

        res_c1, res_c2 = st.columns([1, 1])
        
        with res_c1:
            # 仪表盘绘制
            gauge_color = "#dc3545" if prob >= THRESHOLD else "#28a745"
            risk_text_color = "red" if prob >= THRESHOLD else "green"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                title = {'text': f"<b>Mortality Probability</b><br><span style='color:{risk_text_color};font-size:0.8em'>{risk_label}</span>"},
                gauge = {
                    'axis': {'range': [0, 100]}, 
                    'bar': {'color': gauge_color}, 
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': THRESHOLD*100},
                    'steps': [{'range': [0, THRESHOLD*100], 'color': "#e9ecef"}]
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # === SHAP 解释 ===
        with res_c2:
            st.markdown("**Feature Impact Analysis**")
            with st.spinner("Analyzing risk drivers..."):
                try:
                    # GBM 使用 TreeExplainer
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(final_input)
                    
                    # 兼容性处理: shape_values 可能是 list 或 array
                    if isinstance(shap_values, list): sv = shap_values[1][0] 
                    else: sv = shap_values[0]
                    
                    # 基准值处理
                    if hasattr(explainer, "expected_value"):
                        base_val = explainer.expected_value
                        if isinstance(base_val, (list, np.ndarray)): base_val = base_val[1]
                    else:
                        base_val = 0.0

                    feature_names = ['Age', 'Hb', 'AST', 'Respiratory', 'Beta blocker', 'Cardiotonics', 'Statins', 'Stent']
                    
                    # 构造 Explanation
                    exp = shap.Explanation(
                        values=sv,
                        base_values=base_val,
                        data=final_input[0],
                        feature_names=feature_names
                    )
                    
                    # 绘图
                    fig_shap, ax = plt.subplots(figsize=(6, 5))
                    shap.plots.waterfall(exp, max_display=8, show=False)
                    st.pyplot(fig_shap, bbox_inches='tight')
                    plt.clf()

                except Exception as shap_err:
                    st.warning(f"Feature Analysis Unavailable: {shap_err}")
                    # 创建全零数组作为备用，防止报告生成出错
                    sv = [0.0] * 8

        st.divider()
        # 生成报告
        # 注意: 这里传入 feature_names 列表给报告生成器
        feature_list = ['Age', 'Hb', 'AST', 'Respiratory support', 'Beta blocker', 'Cardiotonics', 'Statins', 'Stent for IRA']
        nlg = ClinicalReportGenerator(inputs, prob, THRESHOLD, sv if isinstance(sv, list) else sv.tolist(), feature_list, 0)
        full_report = nlg.generate_full_report()
        
        with st.expander("📄 View Clinical Report", expanded=True):
            st.markdown(full_report)
        
        # PDF 下载
        pdf_buffer = io.BytesIO()
        pdf_engine = PDFReportEngine(
            pdf_buffer, 
            inputs, 
            {'prob': prob, 'threshold': THRESHOLD, 'risk_label': risk_label}, 
            nlg._generate_clinical_advice()
        )
        
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        st.download_button(
            "📥 Download PDF Report", 
            pdf_engine.generate(), 
            f"STEMI_Report_{time_str}.pdf", 
            "application/pdf"
        )

# ----------------- PAGE 2: 批量处理 -----------------
elif page == "Batch Analysis":
    st.title("Batch Cohort Analysis")
    st.info("Supported formats: CSV, Excel.")
    
    with st.expander("Data Template Guide"):
        st.markdown("""
        **Required Columns (Case-Sensitive):**
        - `Age`, `Hb`, `AST` (Numerical)
        - `Respiratory support`, `Beta blocker`, `Cardiotonics`, `Statins` (0 or 1)
        - `Stent for IRA` (0, 1, or 2)
        """)
        cols = ['Age', 'Hb', 'AST', 'Respiratory support', 'Beta blocker', 'Cardiotonics', 'Statins', 'Stent for IRA']
        template = pd.DataFrame(columns=cols)
        st.download_button("Download CSV Template", template.to_csv(index=False).encode('utf-8'), "STEMI_Batch_Template.csv", "text/csv")

    uploaded_file = st.file_uploader("Upload File", type=['xlsx', 'csv'])
    if uploaded_file:
        if model and scaler:
            processor = BatchProcessor(model, scaler)
            if st.button("🚀 Start Processing"):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                        
                    res_df, err = processor.process_data(df)
                    
                    if err:
                        st.error(err)
                    else:
                        st.success(f"Successfully processed {len(res_df)} records.")
                        
                        high_risk_count = len(res_df[res_df['Risk_Group'] == 'High Risk'])
                        st.metric("High Risk Patients Identified", high_risk_count, f"{high_risk_count/len(res_df):.1%}")
                        
                        st.dataframe(res_df.head())
                        st.download_button(
                            "Download Results (Excel)", 
                            processor.convert_to_excel(res_df), 
                            "STEMI_Predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                except Exception as e:
                    st.error(f"File Processing Error: {e}")

# ----------------- PAGE 3: 临床看板 -----------------
elif page == "Clinical Dashboard":
    st.title("Clinical Analytics Dashboard")
    st.caption("Real-time statistics based on local history.")
    
    analytics = AnalyticsEngine(db)
    df_hist = analytics.get_data()
    
    if df_hist.empty: 
        st.info("No records found in database. Run some predictions first!")
    else:
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Assessments", len(df_hist))
        
        # 计算高危比例
        if 'risk_label' in df_hist.columns:
            high_risk_n = len(df_hist[df_hist['risk_label']=='High Risk'])
            k2.metric("High Risk Ratio", f"{high_risk_n / len(df_hist):.1%}")
        
        if 'risk_prob' in df_hist.columns:
            avg_risk = df_hist['risk_prob'].mean()
            k3.metric("Avg Cohort Risk", f"{avg_risk:.1%}")
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(analytics.plot_risk_distribution(), use_container_width=True)
        with c2:
            st.plotly_chart(analytics.plot_temporal_trend(), use_container_width=True)
            
        # 额外图表：年龄分布
        st.plotly_chart(analytics.plot_age_distribution(), use_container_width=True)

# ----------------- PAGE 4: 项目介绍 -----------------
elif page == "Project Introduction":
    st.title("STEMI 3-Year Mortality Prediction")
    st.markdown("""
    ### Study Abstract
    **Objective:** To construct and validate a machine learning model for predicting all-cause mortality at three years after PCI in STEMI patients.
    
    **Methods:** - **Cohort:** Multicenter study (Wuhan, Yichang, Enshi) involving 2,657 patients.
    - **Model:** Gradient Boosting Machine (GBM) was selected as the optimal model.
    - **Performance:** Achieved an **AUC of 0.801** in the independent external validation cohort.
    
    **Key Predictors:**
    1. Respiratory Support
    2. Age
    3. Cardiotonics
    4. AST
    5. Hemoglobin (Hb)
    6. Beta-blockers
    7. Statins
    8. Stent for IRA
    """)
    
    # 尝试提供说明书下载
    manual_path = os.path.join("assets", "STEMI_User_Manual.docx")
    if os.path.exists(manual_path):
        with open(manual_path, "rb") as f:
            st.download_button("📥 Download User Manual (Word)", f, "STEMI_User_Manual.docx")
    else:
        st.warning("User Manual not found in assets folder.")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 0.8em;'>© 2026 Clinical AI Lab. For Research Use Only.</div>", unsafe_allow_html=True)
