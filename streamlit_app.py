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
    
    # Features include Pathogens + Clinical Symptoms + Age
    features = ['HPV16', 'HPV18', 'Gardnerella', 'Ureaplasma', 'Mycoplasma', 'Age', 'Discharge', 'Odor', 'Itching']
    X = df[features]
    y = df['Target']
    
    # Scaling Age is a scientific requirement so it doesn't 'overpower' binary 0/1 values
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled['Age'] = scaler.fit_transform(X[['Age']])
    
    # Train Logistic Regression
    # We use 'liblinear' solver which is better for small medical datasets (N=26)
    model = LogisticRegression(solver='liblinear', C=1.0)
    model.fit(X_scaled, y)
    
    return model, scaler, features, model.coef_[0]

# Initialize Model
model, scaler, feature_names, coefficients = train_scientific_model()

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
# Prepare input for model
user_input = np.array([[int(h16), int(h18), int(gard), int(urea), int(myco), age, int(dis), int(odo), int(itc)]])
# Scale age using the same scaler used during training
user_input_scaled = user_input.copy()
user_input_scaled[0, 5] = (age - scaler.mean_[5]) / scaler.scale_[5]

# Predict
prob = model.predict_proba(user_input_scaled)[0][1]

with col_output:
    st.subheader("📊 Model Output")
    
    # Probability Gauge Simulation
    st.metric(label="Predicted Probability of High-Risk Lesion", value=f"{prob:.1%}")
    
    if prob > 0.70:
        st.error("CRITICAL RISK: Clinical profile highly correlates with High-Risk categorization.")
    elif prob > 0.40:
        st.warning("ELEVATED RISK: Significant microbial/genomic markers present.")
    else:
        st.success("LOW RISK: Profile aligns with low-risk clinical markers.")

    # 5. FEATURE IMPORTANCE (The "Scientific Proof")
    st.markdown("---")
    st.subheader("💡 Feature Weights (AI Logic)")
    importance_df = pd.DataFrame({
        "Marker": feature_names,
        "Weight": coefficients
    }).sort_values(by="Weight", ascending=False)
    
    st.bar_chart(importance_df.set_index("Marker"))
    st.caption("Positive values increase risk probability; negative values decrease it.")
