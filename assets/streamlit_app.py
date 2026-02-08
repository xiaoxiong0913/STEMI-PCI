import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load model, scaler, and feature list
model_path = r"gbm_model.pkl"
scaler_path = r"scaler.pkl"
features_path = r"features.txt"

# Load model and scaler using pickle
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Check if scaler is an instance of StandardScaler
    if not isinstance(scaler, StandardScaler):
        raise ValueError("Loaded scaler is not an instance of StandardScaler.")

    # Load feature list
    with open(features_path, 'r') as f:
        features = f.read().splitlines()

except Exception as e:
    st.error(f"Error loading model, scaler, or features: {e}")
    st.stop()

st.set_page_config(layout="wide", page_icon="❤️")
st.title("STEMI Post-PCI 3-Year Mortality Prediction System")

# Custom CSS styling for beautification
st.write("""
<style>
.protocol-card {
    padding: 15px;
    border-radius: 10px;
    margin: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.critical-card {
    border-left: 5px solid #dc3545;
    background-color: #fff5f5;
}
.warning-card {
    border-left: 5px solid #ffc107;
    background-color: #fff9e6;
}
.green-card {
    border-left: 5px solid #28a745;
    background-color: #f0fff4;
}
.blue-card {
    border-left: 5px solid #17a2b8;
    background-color: #d1ecf1;
}
.result-card {
    border-radius: 10px;
    padding: 20px;
    background-color: #f8f9fa;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# Introduction section
st.write("# Introduction")
st.write("""
This clinical decision support tool integrates clinical, laboratory, and procedural parameters 
to predict 3-year mortality risk in STEMI patients after primary PCI. Validated with **AUC 0.91 (0.867-0.944)** and **93.4% accuracy**.
""")

# Clinical pathway cards
cols = st.columns([2, 3])

# First column: High Risk Criteria, Laboratory Alerts, Medication Recommendations
with cols[0]:
    st.write("""
    <div class='protocol-card critical-card'>
        <h4 style='color:#dc3545;'>High Risk Criteria</h4>
        <ul style='padding-left:20px'>
            <li>Probability ≥14.7%</li>
            <li>Age >75 years</li>
            <li>Hb <100 g/L</li>
            <li>AST >200 U/L</li>
        </ul>
    </div>
    <div class='protocol-card blue-card'>
        <h4 style='color:#17a2b8;'>Laboratory Alerts</h4>
        <ul style='padding-left:20px'>
            <li>Hb <100 g/L → Hematology consult</li>
            <li>AST >200 U/L → Cardiac enzyme monitoring</li>
            <li>Creatinine >150 μmol/L → Renal function assessment</li>
        </ul>
    </div>
    <div class='protocol-card warning-card'>
        <h4 style='color:#ffc107;'>Medication Recommendations</h4>
        <ul style='padding-left:20px'>
            <li>High-risk patients: High-intensity statin therapy</li>
            <li>Beta-blocker continuation unless contraindicated</li>
            <li>Dual antiplatelet therapy for ≥12 months</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Second column: Monitoring & Standard Protocol
with cols[1]:
    st.write("""
    <div class='protocol-card green-card'>
        <h4 style='color:#28a745;'>Post-PCI Monitoring Protocol</h4>
        <ul style='padding-left:20px'>
            <li>Cardiac telemetry for 24-48 hours post-PCI</li>
            <li>Serial troponin measurements every 6 hours × 3</li>
            <li>Echocardiography within 24 hours</li>
            <li>Lipid profile assessment before discharge</li>
            <li>Cardiac rehabilitation referral</li>
            <li>Follow-up at 1, 3, 6, and 12 months</li>
            <li>Annual stress testing for high-risk patients</li>
            <li>Lifestyle modification counseling</li>
            <li>Smoking cessation support</li>
            <li>Diabetes management optimization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Define normal value ranges for continuous variables
normal_ranges = {
    'Age': (18, 120),  # Age range
    'Hb': (130, 175),  # Normal hemoglobin range (g/L) - male reference
    'AST': (10, 40),  # Normal AST range (U/L)
}

# Feature list (corrected)
features = [
    'Age', 'Hb', 'AST', 'Respiratory support', 'Beta blocker',
    'Cardiotonics', 'Statins', 'Stent for IRA'
]

with st.sidebar:
    st.write("## Patient Parameters")
    with st.form("input_form"):
        inputs = {}

        # Continuous variables with normal ranges
        inputs['Age'] = st.slider("Age (Years)", 30, 100, 65)
        inputs['Hb'] = st.slider("Hemoglobin (g/L)", 60, 200, 130)
        inputs['AST'] = st.slider("AST (U/L)", 5, 600, 30)

        # Binary categorical variables
        inputs['Respiratory support'] = st.selectbox("Respiratory support", ["No", "Yes"])
        inputs['Beta blocker'] = st.selectbox("Beta blocker at discharge", ["No", "Yes"])
        inputs['Cardiotonics'] = st.selectbox("Cardiotonics use", ["No", "Yes"])
        inputs['Statins'] = st.selectbox("Statins at discharge", ["No", "Yes"])

        # Three-category variable: Stent for IRA
        stent_options = ["No stent", "Drug-eluting stent (DES)", "Bare-metal stent (BMS)"]
        inputs['Stent for IRA'] = st.selectbox("Stent for infarct-related artery", stent_options)

        submitted = st.form_submit_button("Predict Risk")

# Process prediction
if submitted:
    try:
        # Data preprocessing
        input_data = {}
        for k, v in inputs.items():
            if k == 'Stent for IRA':
                # Map stent options to numerical values
                stent_mapping = {
                    "No stent": 0,
                    "Drug-eluting stent (DES)": 1,
                    "Bare-metal stent (BMS)": 2
                }
                input_data[k] = stent_mapping[v]
            elif isinstance(v, str):
                input_data[k] = 1 if v == "Yes" else 0
            else:
                input_data[k] = v

        # Create DataFrame with correct feature order
        df = pd.DataFrame([input_data], columns=features)

        # Apply scaling
        df_scaled = scaler.transform(df)

        # Predict probability
        prob = model.predict_proba(df_scaled)[:, 1][0]
        risk_status = "High Risk" if prob >= 0.147 else "Low Risk"
        color = "#dc3545" if risk_status == "High Risk" else "#28a745"

        # Risk message
        risk_message = "High risk of mortality within 3 years." if risk_status == "High Risk" else "Low risk of mortality within 3 years."

        # Check for abnormal values and highlight them
        abnormal_vars = []
        advice = []

        for var, value in inputs.items():
            if var in normal_ranges:
                lower, upper = normal_ranges[var]
                if value < lower or value > upper:
                    abnormal_vars.append(var)
                    # Add the advice message
                    if var == 'Hb':
                        if value < lower:
                            advice.append(
                                f"<b>Hemoglobin ({value} g/L)</b>: Below normal range (130-175 g/L). Consider anemia workup and iron studies.")
                        else:
                            advice.append(
                                f"<b>Hemoglobin ({value} g/L)</b>: Above normal range (130-175 g/L). Evaluate for polycythemia.")
                    elif var == 'AST':
                        if value > upper:
                            advice.append(
                                f"<b>AST ({value} U/L)</b>: Elevated above normal (10-40 U/L). May indicate ongoing myocardial injury or liver dysfunction.")
                    elif var == 'Age':
                        if value > 75:
                            advice.append(
                                f"<b>Age ({value} years)</b>: Advanced age is an independent risk factor for adverse outcomes in STEMI.")

        # Display results
        st.markdown(f"""
        <div class='result-card'>
            <h2 style='color:{color};'>Predicted 3-Year Mortality Risk: {prob * 100:.1f}% ({risk_status})</h2>
            <p>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)

        # Display abnormal variables with advice
        if abnormal_vars:
            st.markdown("<h4 style='color: red;'>Clinical Advisory:</h4>", unsafe_allow_html=True)
            st.markdown("<p style='color: red;'>Abnormal parameters detected requiring attention:</p>",
                        unsafe_allow_html=True)
            for adv in advice:
                st.markdown(f"<p>{adv}</p>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.write("---")
st.write("<div style='text-align: center; color: gray;'>Developed according to ESC 2023 STEMI Guidelines</div>",
         unsafe_allow_html=True)