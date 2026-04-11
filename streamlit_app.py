import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. PAGE SETUP
st.set_page_config(page_title="AI HPV-Microbiota Research Tool", layout="wide")
st.title("🔬 Advanced HPV-Microbiota Risk Engine")
st.markdown("---")

# 2. LOAD & TRAIN ON REAL DATA
@st.cache_data
def train_scientific_model():
    # Loading your exact pilot data
    df = pd.read_csv('HPV_Microbiota_Risk_Model.xlsx - Sheet1.csv')
    
    # Define Target: High Risk = 1, others = 0
    df['Target'] = df['Risk Category'].apply(lambda x: 1 if x == 'High' else 0)
    
    # EXACT column names from your CSV
    features = ['Discharge', 'Odor', 'Itching', 'HPV16', 'HPV18', 'Gardnerella', 'Ureaplasma', 'Mycoplasma', 'Age']
    X = df[features]
    y = df['Target']
    
    # Standardize Age so it doesn't skew the model
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[['Age']] = scaler.fit_transform(X[['Age']])
    
    # Train Logistic Regression
    model = LogisticRegression(solver='liblinear')
    model.fit(X_scaled, y)
    
    return model, scaler, features, model.coef_[0]

# Initialize Model
try:
    model, scaler, feature_names, coefficients = train_scientific_model()
except Exception as e:
    st.error(f"Error loading data: {e}. Please ensure 'HPV_Microbiota_Risk_Model.xlsx - Sheet1.csv' is in your GitHub repo.")
    st.stop()

# 3. INTERACTIVE INTERFACE
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("🧬 Patient Genomic & Microbial Profile")
    age = st.slider("Patient Age", 18, 75, 35)
    
    st.markdown("**Pathogen Detection (PCR)**")
    c1, c2 = st.columns(2)
    h16 = c1.checkbox("HPV 16 +")
    h18 = c1.checkbox("HPV 18 +")
    gard = c2.checkbox("Gardnerella +")
    urea = c2.checkbox("Ureaplasma +")
    myco = c2.checkbox("Mycoplasma +")
    
    st.markdown("**Clinical Presentation**")
    c3, c4 = st.columns(2)
    dis = c3.checkbox("Abnormal Discharge")
    odo = c3.checkbox("Malodor")
    itc = c4.checkbox("Pruritus (Itching)")

# 4. SCIENTIFIC PREDICTION LOGIC
# Must be in the EXACT order of 'features' list above:
# ['Discharge', 'Odor', 'Itching', 'HPV16', 'HPV18', 'Gardnerella', 'Ureaplasma', 'Mycoplasma', 'Age']
user_data = pd.DataFrame([[
    int(dis), int(odo), int(itc), 
    int(h16), int(h18), int(gard), 
    int(urea), int(myco), age
]], columns=feature_names)

# Scale the age column only
user_data[['Age']] = scaler.transform(user_data[['Age']])

# Predict Probability
prob = model.predict_proba(user_data)[0][1]

with col_output:
    st.subheader("📊 AI Prediction Analysis")
    st.metric(label="High-Risk Probability", value=f"{prob:.1%}")
    
    if prob > 0.70:
        st.error("🚨 HIGH RISK PROFILE")
    elif prob > 0.40:
        st.warning("⚠️ MODERATE RISK PROFILE")
    else:
        st.success("✅ LOW RISK PROFILE")

    # Feature Importance Visualization
    st.markdown("---")
    st.subheader("💡 Statistical Weights")
    importance_df = pd.DataFrame({"Marker": feature_names, "Weight": coefficients}).sort_values(by="Weight", ascending=False)
    st.bar_chart(importance_df.set_index("Marker"))
