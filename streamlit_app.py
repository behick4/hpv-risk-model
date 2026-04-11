import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. PAGE CONFIG & CUSTOM THEMING
st.set_page_config(page_title="HPV-Microbiota AI", page_icon="🎀", layout="wide")

st.markdown("""
    <style>
    .main-title { text-align: center; color: #D81B60; font-family: 'Helvetica Neue', sans-serif; font-weight: bold; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #880E4F; font-style: italic; margin-top: 0px; margin-bottom: 30px; }
    .stMetric { background-color: #FCE4EC; padding: 15px; border-radius: 10px; border: 1px solid #F06292; }
    .info-text { font-size: 16px; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# 2. CENTERED TITLE
st.markdown("<h1 class='main-title'>🎀 AI Cervical Cancer Risk Stratification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Evidence-Based Integration of Genomics & Vaginal Microbiota Analysis</p>", unsafe_allow_html=True)

# 3. SCIENTIFIC CONTEXT SECTION (IMAGE + BULLET POINTS)
col_img, col_txt = st.columns([1, 1])

with col_img:
    # Scientific illustration representing the progression from HPV to Cancer
    st.image("https://www.nccc-online.org/wp-content/uploads/2021/01/HPV_Infographic_Social_Media_1.png", use_container_width=True)

with col_txt:
    st.markdown("### 🧬 HPV Pathogenesis & Oncogenesis")
    st.markdown("""
    <div class='info-text'>
    <ul>
        <li><b>Viral Integration:</b> High-risk HPV (16/18) integrates its DNA into the host genome, leading to the overexpression of <b>E6 and E7 oncogenes</b>.</li>
        <li><b>Tumor Suppression:</b> E6/E7 proteins degrade p53 and pRb, neutralizing the cell's natural ability to stop cancer growth.</li>
        <li><b>The Microbiota Factor:</b> Emerging research (including our pilot study) suggests that <i>Gardnerella</i> and other anaerobes increase inflammation and vaginal pH, facilitating viral persistence.</li>
        <li><b>Risk Synergy:</b> Cervical cancer is rarely caused by HPV alone; the local microbial environment acts as a critical co-factor in lesion progression.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# 4. DATA & MODEL LOADING
@st.cache_data
def load_and_train():
    try:
        df = pd.read_csv('HPV_Microbiota_Risk_Model.xlsx - Sheet1.csv')
        df['Target'] = df['Risk Category'].apply(lambda x: 1 if x == 'High' else 0)
        features = ['Discharge', 'Odor', 'Itching', 'HPV16', 'HPV18', 'Gardnerella', 'Ureaplasma', 'Mycoplasma', 'Age']
        X = df[features]
        y = df['Target']
        scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[['Age']] = scaler.fit_transform(X[['Age']])
        model = LogisticRegression(solver='liblinear').fit(X_scaled, y)
        return model, scaler, features, model.coef_[0]
    except:
        # Fallback if file not found
        st.error("CSV File not found. Please upload it to GitHub.")
        st.stop()

model, scaler, feature_names, coefficients = load_and_train()

# 5. SIDEBAR INPUTS
st.sidebar.header("📝 Clinical Inputs")
age = st.sidebar.slider("Patient Age", 18, 75, 35)
h16 = st.sidebar.toggle("HPV 16 Positive")
h18 = st.sidebar.toggle("HPV 18 Positive")
gard = st.sidebar.toggle("Gardnerella Presence")
urea = st.sidebar.toggle("Ureaplasma Presence")
myco = st.sidebar.toggle("Mycoplasma Presence")
dis = st.sidebar.toggle("Abnormal Discharge")
odo = st.sidebar.toggle("Vaginal Malodor")
itc = st.sidebar.toggle("Pruritus (Itching)")

# 6. RISK CALCULATION
user_data = pd.DataFrame([[int(dis), int(odo), int(itc), int(h16), int(h18), int(gard), int(urea), int(myco), age]], columns=feature_names)
user_data[['Age']] = scaler.transform(user_data[['Age']])
prob = model.predict_proba(user_data)[0][1]

# 7. DYNAMIC RISK VISUALIZATION
col_res, col_weights = st.columns([1, 1])

with col_res:
    st.subheader("📊 AI Risk Assessment")
    if prob < 0.3:
        color, status = "#4CAF50", "LOW RISK"
    elif prob < 0.7:
        color, status = "#FF9800", "MODERATE RISK"
    else:
        color, status = "#D81B60", "HIGH RISK"

    st.markdown(f"<h2 style='color: {color}; text-align: center;'>{status}</h2>", unsafe_allow_html=True)
    st.progress(prob)
    st.metric(label="High-Risk Probability", value=f"{prob:.1%}")

with col_weights:
    st.subheader("💡 Statistical Weighting")
    importance_df = pd.DataFrame({"Factor": feature_names, "Weight": coefficients}).sort_values(by="Weight")
    st.bar_chart(importance_df.set_index("Factor"), color="#F06292")

st.divider()
st.info("💡 **Scientific Recommendation:** High probability scores indicate a need for immediate cytology or colposcopy, regardless of symptom severity.")



