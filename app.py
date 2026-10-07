import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cardio Predictor AI", page_icon="❤️", layout="wide")

# --- CUSTOM CSS (PREMIUM UI) ---
st.markdown("""
<style>
    /* Hide Streamlit default menu and footer for a clean app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Style the predict button */
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0px 4px 15px rgba(255, 75, 75, 0.4);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=100) # Generic heart icon
    st.title("System Info")
    st.info("**Model Engine:**\nTuned Random Forest Ensemble")
    st.info("**Recall Optimization:**\nThreshold lowered to 0.40 to minimize False Negatives.")
    st.success("**Baseline Accuracy:** ~70.35%")
    st.write("---")
    st.write("Developed for CACS452 Project III")

# --- HEADER ---
st.title("❤️ AI Cardiovascular Risk Predictor")
st.markdown("Enter patient vitals below. The AI will analyze the data in real-time to predict cardiovascular disease risk.")
st.markdown("---")

# --- MAIN LAYOUT (2 COLUMNS) ---
col_form, col_results = st.columns([1.2, 1])

with col_form:
    st.subheader("📋 Patient Vitals")
    with st.form("patient_data_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
            ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=60, max_value=250, value=120, help="Ideal is < 120")
            cholesterol = st.selectbox("Cholesterol", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
            smoke = st.selectbox("Smoker?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
            active = st.selectbox("Physically Active?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])

        with c2:
            gender_input = st.selectbox("Gender", options=[(1, 'Female'), (2, 'Male')], format_func=lambda x: x[1])
            weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
            ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80, help="Ideal is < 80")
            gluc = st.selectbox("Glucose Level", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
            alco = st.selectbox("Alcohol Intake?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🔍 Analyze Risk")

# --- PREDICTION LOGIC & UI ---
with col_results:
    st.subheader("🔬 AI Assessment")
    
    if not submit_button:
        st.info("Awaiting patient data. Fill out the form and click 'Analyze Risk' to see the AI's prediction.")
    else:
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender_input[0]],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol[0]],
            'gluc': [gluc[0]],
            'smoke': [smoke[0]],
            'alco': [alco[0]],
            'active': [active[0]]
        })
        
        # Run Prediction
        probability = model.predict_proba(input_data)[0][1]
        
        # 1. Dashboard Metrics
        st.metric(label="Disease Probability", value=f"{probability*100:.1f}%")
        st.progress(float(probability))
        
        if probability < 0.40:
            st.success("✅ **STATUS: LOW RISK**")
            st.write("The AI indicates a low risk of cardiovascular disease. Maintain a healthy lifestyle.")
        elif probability < 0.65:
            st.warning("⚠️ **STATUS: MODERATE RISK**")
            st.write("The AI detects moderate risk factors. Recommend lifestyle improvements and a routine medical checkup.")
        else:
            st.error("🚨 **STATUS: HIGH RISK**")
            st.write("The AI detected critical risk factors for cardiovascular disease. **Immediate medical consultation is strongly advised.**")
            
        # 2. Explainable AI: Feature Importance
        st.markdown("---")
        st.write("**Explainable AI: Risk Factor Analysis**")
        
        feature_names = ['Age', 'Gender', 'Height', 'Weight', 'Sys BP', 'Dia BP', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active']
        importances = model.feature_importances_
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=True)
        
        # Custom sleek plot
        fig, ax = plt.subplots(figsize=(6, 4))
        # Use a sleek color based on risk (Red if high/mod risk, green if low)
        bar_color = '#ff4b4b' if probability >= 0.4 else '#28a745'
        ax.barh(importance_df['Feature'], importance_df['Importance'], color=bar_color, edgecolor='none')
        ax.set_xlabel('Impact on AI Decision', fontsize=10)
        
        # Remove borders for a cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#dddddd')
        ax.spines['left'].set_color('#dddddd')
        
        st.pyplot(fig)
