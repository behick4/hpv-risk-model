import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. PAGE CONFIG & CUSTOM THEMING (PINK THEME)
st.set_page_config(page_title="HPV-Microbiota AI", page_icon="🎀", layout="wide")

st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #D81B60; /* Deep Pink */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #880E4F;
        font-style: italic;
    }
    .stMetric {
        background-color: #FCE4EC;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #F06292;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CENTERED TITLE & VISUALS
st.markdown("<h1 class='main-title'>🎀 AI Cervical Cancer Risk Stratification</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Integrated Genomics & Vaginal Microbiota Analysis</p>", unsafe_allow_html=True)

# Path to an illustrative image (Using a public URL for health graphics)
st.image("https://www.google.com/url?sa=t&source=web&rct=j&url=https%3A%2F%2Fwww.mmcdubai.ae%2Fhpv%2F&ved=0CBYQjRxqGAoTCLDL4pCP5pMDFQAAAAAdAAAAABCiAQ&opi=89978449", width=400)

# 3. DATA & MODEL LOADING
@st.cache_data
def load_and_train():
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

model, scaler, feature_names, coefficients = load_and_train()

# 4. SIDEBAR INPUTS
st.sidebar.header("📝 Clinical Dashboard")
st.sidebar.markdown("Toggle patient factors below to calculate the real-time risk ratio.")

age = st.sidebar.slider("Patient Age", 18, 75, 35)
st.sidebar.divider()
st.sidebar.markdown("**Molecular Markers**")
h16 = st.sidebar.toggle("HPV 16 Positive")
h18 = st.sidebar.toggle("HPV 18 Positive")
gard = st.sidebar.toggle("Gardnerella Presence")
urea = st.sidebar.toggle("Ureaplasma Presence")
myco = st.sidebar.toggle("Mycoplasma Presence")
st.sidebar.divider()
st.sidebar.markdown("**Clinical Symptoms**")
dis = st.sidebar.toggle("Abnormal Discharge")
odo = st.sidebar.toggle("Vaginal Malodor")
itc = st.sidebar.toggle("Pruritus (Itching)")

# 5. RISK CALCULATION
user_data = pd.DataFrame([[int(dis), int(odo), int(itc), int(h16), int(h18), int(gard), int(urea), int(myco), age]], columns=feature_names)
user_data[['Age']] = scaler.transform(user_data[['Age']])
prob = model.predict_proba(user_data)[0][1]

# 6. DYNAMIC RISK VISUALIZATION
col_main, col_guide = st.columns([2, 1])

with col_main:
    st.subheader("📊 Personalized Risk Assessment")
    
    # Dynamic Color based on Risk
    if prob < 0.3:
        color = "#4CAF50" # Green
        status = "LOW RISK"
    elif prob < 0.7:
        color = "#FF9800" # Orange
        status = "MODERATE RISK"
    else:
        color = "#D81B60" # Pink/Red
        status = "HIGH RISK"

    # Risk Ratio Progress Bar
    st.markdown(f"<h2 style='color: {color};'>{status}</h2>", unsafe_allow_html=True)
    st.progress(prob)
    st.write(f"Confidence Level: **{prob:.1%}**")
    
    # Feature Weights Bar Chart
    st.markdown("### How the AI decided:")
    importance_df = pd.DataFrame({"Factor": feature_names, "Weight": coefficients}).sort_values(by="Weight")
    st.bar_chart(importance_df.set_index("Factor"), color="#F06292")

with col_guide:
    st.subheader("📚 Decision Support")
    st.write("Below is how each feature impacts the result:")
    
    impact_list = [
        ("HPV 16", "Primary oncogenic driver. Increases risk significantly."),
        ("Gardnerella", "Strong co-factor. Presence suggests high dysbiosis."),
        ("Age", "Increases risk due to potential viral persistence."),
        ("HPV 18", "Currently weighted lower in this cohort (Pilot size N=26)."),
        ("Symptoms", "Odor/Discharge often align with microbial shifts.")
    ]
    
    for item, desc in impact_list:
        with st.expander(f"Effect of {item}"):
            st.write(desc)

st.divider()
st.info("💡 **Scientific Recommendation:** For patients with a probability > 60%, co-testing for microbial dysbiosis and HPV mRNA is advised to confirm persistent oncogenic activity.")


