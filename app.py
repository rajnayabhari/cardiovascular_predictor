import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cardiovascular Diagnostic System", page_icon="plus", layout="wide")

# --- CUSTOM CSS (PREMIUM CORPORATE UI) ---
st.markdown("""
<style>
    /* Hide Streamlit default menu and footer for a clean app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Professional corporate styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Style the predict button with corporate blue */
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        height: 50px;
        transition: all 0.3s ease;
        border: none;
    }
    .stButton>button:hover {
        background-color: #34495e;
        box-shadow: 0px 4px 10px rgba(44, 62, 80, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- HEADER ---
st.title("Cardiovascular Diagnostic System")
st.markdown("Enterprise Edition | Version 2.1")
st.markdown("---")

# --- TABS SETUP ---
tab1, tab2, tab3 = st.tabs(["Patient Assessment", "Treatment Simulator", "Batch Processing"])

# --- TAB 1: SINGLE PATIENT ---
with tab1:
    col_form, col_results = st.columns([1.2, 1])
    
    with col_form:
        st.subheader("Patient Vitals Entry")
        with st.form("patient_data_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50)
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
                ap_hi = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
                cholesterol = st.selectbox("Cholesterol", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
                smoke = st.selectbox("Smoker History", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
                active = st.selectbox("Physically Active", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])

            with c2:
                gender_input = st.selectbox("Gender", options=[(1, 'Female'), (2, 'Male')], format_func=lambda x: x[1])
                weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
                ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=150, value=80)
                gluc = st.selectbox("Glucose Level", options=[(1, 'Normal'), (2, 'Above Normal'), (3, 'Well Above Normal')], format_func=lambda x: x[1])
                alco = st.selectbox("Alcohol Consumption", options=[(0, 'No'), (1, 'Yes')], format_func=lambda x: x[1])
                
            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button(label="Execute Analysis")
            
    with col_results:
        st.subheader("Diagnostic Assessment")
        
        if not submit_button:
            st.info("System idle. Please enter patient parameters and execute analysis.")
        elif ap_lo >= ap_hi:
            st.error("Validation Error: Diastolic pressure cannot exceed or equal Systolic pressure.")
        else:
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
            
            input_df['bmi'] = (input_df['weight'] / ((input_df['height'] / 100) ** 2)).round(2)
            input_df['pulse_pressure'] = input_df['ap_hi'] - input_df['ap_lo']
            input_df['map'] = ((input_df['ap_hi'] + 2 * input_df['ap_lo']) / 3).round(2)
            
            probability = model.predict_proba(input_df)[0][1]
            
            st.metric(label="Calculated Disease Probability", value=f"{probability*100:.1f}%")
            st.progress(float(probability))
            
            if probability < 0.40:
                st.success("STATUS: LOW RISK")
            elif probability < 0.65:
                st.warning("STATUS: MODERATE RISK")
            else:
                st.error("STATUS: HIGH RISK")
                
            st.markdown("---")
            st.write("**Feature Contribution Analysis**")
            
            feature_names = ['Age', 'Gender', 'Height', 'Weight', 'Sys BP', 'Dia BP', 'Cholesterol', 'Glucose', 'Smoke', 'Alcohol', 'Active', 'BMI', 'Pulse Pressure', 'MAP']
            importances = model.feature_importances_
            
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=True)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_color = '#e74c3c' if probability >= 0.40 else '#2c3e50'
            ax.barh(importance_df['Feature'], importance_df['Importance'], color=bar_color, edgecolor='none')
            ax.set_xlabel('Relative Impact Coefficient', fontsize=10)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#dddddd')
            ax.spines['left'].set_color('#dddddd')
            
            st.pyplot(fig)
            plt.clf()
                
            st.session_state['baseline_patient'] = input_df
            st.session_state['baseline_prob'] = probability

# --- TAB 2: TREATMENT SIMULATOR ---
with tab2:
    st.subheader("Intervention Simulator")
    
    if 'baseline_patient' not in st.session_state:
        st.info("Notice: Initialize a patient profile in the Assessment tab prior to simulation.")
    else:
        st.write("Configure intervention parameters to project changes in cardiovascular risk probability.")
        st.markdown("---")
        
        base_df = st.session_state['baseline_patient'].copy()
        original_prob = st.session_state['baseline_prob']
        
        col_sim1, col_sim2 = st.columns(2)
        
        with col_sim1:
            st.markdown("#### Clinical Targets")
            
            original_weight = float(base_df['weight'].iloc[0])
            max_weight_loss = max(0.0, float(original_weight - 40.0))
            max_slider_loss = min(50.0, max_weight_loss)
            
            if max_slider_loss > 0:
                weight_loss = st.slider("Target Weight Reduction (kg)", min_value=0.0, max_value=float(max_slider_loss), value=0.0, step=1.0)
            else:
                st.success("Weight parameter is at minimum safe threshold.")
                weight_loss = 0.0
            
            original_ap_hi = int(base_df['ap_hi'].iloc[0])
            original_ap_lo = int(base_df['ap_lo'].iloc[0])
            
            max_safe_reduction = max(0, original_ap_hi - max(90, original_ap_lo + 10))
            
            if max_safe_reduction > 0:
                bp_reduction = st.slider("Target Systolic Reduction (mmHg)", min_value=0, max_value=max_safe_reduction, value=0, step=1)
            else:
                st.success("Blood pressure parameter is at minimum safe threshold.")
                bp_reduction = 0
            
            original_smoke = base_df['smoke'].iloc[0]
            if original_smoke == 1:
                quit_smoking = st.checkbox("Apply Smoking Cessation", value=False)
            else:
                st.success("Patient is classified as non-smoker.")
                quit_smoking = False
                
        with col_sim2:
            st.markdown("#### Outcome Projection")
            
            sim_df = base_df.copy()
            sim_df['weight'] = sim_df['weight'] - weight_loss
            sim_df['ap_hi'] = sim_df['ap_hi'] - bp_reduction
            if quit_smoking:
                sim_df['smoke'] = 0
                
            sim_df['bmi'] = (sim_df['weight'] / ((sim_df['height'] / 100) ** 2)).round(2)
            sim_df['pulse_pressure'] = sim_df['ap_hi'] - sim_df['ap_lo']
            sim_df['map'] = ((sim_df['ap_hi'] + 2 * sim_df['ap_lo']) / 3).round(2)
                
            new_prob = model.predict_proba(sim_df)[0][1]
            delta = new_prob - original_prob
            
            st.metric(
                label="Projected Probability", 
                value=f"{new_prob*100:.1f}%", 
                delta=f"{delta*100:.1f}%", 
                delta_color="inverse"
            )
            
            if new_prob < 0.40:
                st.success("Target Reached: Patient reclassified to Low Risk.")
            elif new_prob < original_prob:
                st.info("Risk reduction achieved. Further intervention recommended.")
            else:
                st.warning("No significant variance detected.")

# --- TAB 3: BATCH PROCESSING ---
with tab3:
    st.subheader("Batch Patient Processing")
    st.write("Upload a formatted dataset to process multiple profiles simultaneously.")
    
    uploaded_file = st.file_uploader("Data Source (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            st.success(f"System loaded {len(bulk_df)} records.")
            
            required_cols = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
            
            if not all(col in bulk_df.columns for col in required_cols):
                st.error(f"Format Error: Dataset must contain the following schema: {', '.join(required_cols)}")
            else:
                initial_count = len(bulk_df)
                
                # 1. Purge records with missing/blank data (NaNs)
                bulk_df = bulk_df.dropna(subset=required_cols)
                nan_count = initial_count - len(bulk_df)
                if nan_count > 0:
                    st.warning(f"Validation Log: Purged {nan_count} records due to missing/blank values.")
                
                # 2. Purge records with biological impossibilities
                current_count = len(bulk_df)
                bulk_df = bulk_df[bulk_df['ap_lo'] < bulk_df['ap_hi']]
                invalid_count = current_count - len(bulk_df)
                
                if invalid_count > 0:
                    st.warning(f"Validation Log: Purged {invalid_count} invalid records (Diastolic >= Systolic).")
                
                if len(bulk_df) == 0:
                    st.error("Operation Aborted: No valid records found.")
                else:
                    with st.spinner("Executing analysis..."):
                        X_bulk = bulk_df[required_cols].copy()
                        
                        X_bulk['bmi'] = (X_bulk['weight'] / ((X_bulk['height'] / 100) ** 2)).round(2)
                        X_bulk['pulse_pressure'] = X_bulk['ap_hi'] - X_bulk['ap_lo']
                        X_bulk['map'] = ((X_bulk['ap_hi'] + 2 * X_bulk['ap_lo']) / 3).round(2)
                        
                        probabilities = model.predict_proba(X_bulk)[:, 1]
                        
                        diagnoses = ["Elevated Risk (Review)" if p >= 0.40 else "Standard Risk" for p in probabilities]
                        
                        result_df = bulk_df.copy()
                        result_df['Risk_Probability'] = (probabilities * 100).round(1).astype(str) + "%"
                        result_df['Diagnostic_Classification'] = diagnoses
                        
                        st.write("### Output Preview")
                        st.dataframe(result_df.head(50))
                        
                        csv_export = result_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="Export Full Report (CSV)",
                            data=csv_export,
                            file_name='diagnostic_report.csv',
                            mime='text/csv',
                            type="primary"
                        )
        except Exception as e:
            st.error(f"Processing Error: {e}")
