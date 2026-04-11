import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import numpy as np

# Set Page Config for a professional biotech look
st.set_page_config(page_title="HPV-Microbiota AI Risk Tool", layout="centered")

st.title("🔬 HPV & Microbiota Risk Predictor")
st.markdown("### Personalised Cervical Cancer Risk Stratification")
st.write("This AI prototype uses logistic regression weights derived from clinical pilot data.")

# Sidebar for Inputs
st.sidebar.header("Patient Parameters")
age = st.sidebar.slider("Patient Age", 18, 65, 35)
hpv16 = st.sidebar.checkbox("HPV 16 Positive")
hpv18 = st.sidebar.checkbox("HPV 18 Positive")
gardnerella = st.sidebar.checkbox("Gardnerella Present")
ureaplasma = st.sidebar.checkbox("Ureaplasma Present")
mycoplasma = st.sidebar.checkbox("Mycoplasma Present")

# Your Model Coefficients
intercept = -5.9398
c_hpv16 = 2.0776
c_hpv18 = 1.1552
c_gard = 0.9171
c_urea = -0.1758
c_myco = 0.1041
c_age = 0.1296

# Calculation
z = (intercept + 
     (age * c_age) + 
     (int(hpv16) * c_hpv16) + 
     (int(hpv18) * c_hpv18) + 
     (int(gardnerella) * c_gard) + 
     (int(ureaplasma) * c_urea) + 
     (int(mycoplasma) * c_myco))

probability = 1 / (1 + np.exp(-z))

# Visual Display
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.metric(label="High-Risk Probability", value=f"{probability:.1%}")

with col2:
    if probability > 0.5:
        st.error("Status: HIGH RISK")
    elif probability > 0.2:
        st.warning("Status: MEDIUM RISK")
    else:
        st.success("Status: LOW RISK")

# Dynamic Clinical Insight
if hpv16 and gardnerella:
    st.info("**Clinical Note:** Synergistic risk detected. HPV16 presence combined with Gardnerella dysbiosis suggests high inflammatory potential.")
    
