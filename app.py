import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise Cardio System", page_icon="❤️", layout="wide")

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

# --- HEADER ---
st.title("🏥 Enterprise Cardiovascular Diagnostic System")
st.markdown("---")

# --- TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["👤 Single Patient Assessment", "📈 Treatment Simulator", "📊 Hospital Bulk Processing"])

# --- TAB 1: SINGLE PATIENT ---
with tab1:
    col_form, col_results = st.columns([1.2, 1])
    
    with col_form:
        st.subheader("📋 Patient Vitals")
        with st.form("patient_data_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
                ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=60, max_value=250, value=120)
                cholesterol = st.selectbox("Cholesterol", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
                smoke = st.selectbox("Smoker?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
                active = st.selectbox("Physically Active?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])

            with c2:
                gender_input = st.selectbox("Gender", options=[(1, 'Female'), (2, 'Male')], format_func=lambda x: x[1])
                weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
                ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
                gluc = st.selectbox("Glucose Level", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
                alco = st.selectbox("Alcohol Intake?", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="🔍 Analyze Risk")
            
    with col_results:
        st.subheader("🔬 AI Assessment")
        
        if not submit_button:
            st.info("Awaiting patient data. Fill out the form and click 'Analyze Risk' to see the AI's prediction.")
        else:
            # 1. Prepare data matching exact column names expected by the model
            input_df = pd.DataFrame({
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
            
            # 2. Predict Probability
            probability = model.predict_proba(input_df)[0][1]
            
            # 3. Apply custom threshold and display metrics
            st.metric(label="Disease Probability", value=f"{probability*100:.1f}%")
            st.progress(float(probability))
            
            if probability < 0.40:
                st.success("✅ **STATUS: LOW RISK**")
            elif probability < 0.65:
                st.warning("⚠️ **STATUS: MODERATE RISK**")
            else:
                st.error("🚨 **STATUS: HIGH RISK**")
                
            # 4. Explainable AI: Feature Importance
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
            bar_color = '#ff4b4b' if probability >= 0.40 else '#28a745'
            ax.barh(importance_df['Feature'], importance_df['Importance'], color=bar_color, edgecolor='none')
            ax.set_xlabel('Impact on AI Decision', fontsize=10)
            
            # Remove borders for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#dddddd')
            ax.spines['left'].set_color('#dddddd')
            
            st.pyplot(fig)
            plt.clf()
                
            # 5. Save the baseline data into session state so Tab 2 can simulate treatments
            st.session_state['baseline_patient'] = input_df
            st.session_state['baseline_prob'] = probability

# --- TAB 2: TREATMENT SIMULATOR ---
with tab2:
    st.subheader("📈 Medical Treatment Simulator")
    
    if 'baseline_patient' not in st.session_state:
        st.info("⚠️ Please analyze a patient in 'Tab 1: Single Patient Assessment' first to use the simulator.")
    else:
        st.write("Adjust the interventions below to simulate how lifestyle changes affect the patient's cardiovascular risk.")
        st.markdown("---")
        
        base_df = st.session_state['baseline_patient'].copy()
        original_prob = st.session_state['baseline_prob']
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### 🩺 Clinical Interventions")
            # Weight reduction slider
            weight_loss = st.slider("Target Weight Loss (kg)", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
            
            # BP Control slider
            bp_reduction = st.slider("Systolic BP Reduction (mmHg)", min_value=0, max_value=60, value=0, step=1)
            
            # Smoking cessation toggle
            original_smoke = base_df['smoke'].iloc[0]
            if original_smoke == 1:
                quit_smoking = st.checkbox("Simulate Quitting Smoking", value=False)
            else:
                st.success("🚭 Patient is already a non-smoker.")
                quit_smoking = False
                
        with col_sim2:
            st.markdown("#### 🔮 Projected Outcomes")
            
            # Apply the simulated changes to the dataframe
            sim_df = base_df.copy()
            sim_df['weight'] = sim_df['weight'] - weight_loss
            sim_df['ap_hi'] = sim_df['ap_hi'] - bp_reduction
            if quit_smoking:
                sim_df['smoke'] = 0
                
            # Calculate the new theoretical probability
            new_prob = model.predict_proba(sim_df)[0][1]
            
            # Calculate the difference (delta)
            delta = new_prob - original_prob
            
            # Display using Streamlit's awesome metric component
            st.metric(
                label="New Simulated Disease Probability", 
                value=f"{new_prob*100:.1f}%", 
                delta=f"{delta*100:.1f}%", 
                delta_color="inverse" # Inverse means negative delta (drop in risk) is colored Green
            )
            
            if new_prob < 0.40:
                st.success("✅ **Goal Achieved!** Patient drops back into the Low Risk category.")
            elif new_prob < original_prob:
                st.info("📉 **Improvement shown**, but patient remains at risk. Consider further interventions.")
            else:
                st.warning("No significant change in risk profile.")
