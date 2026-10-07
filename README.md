# ❤️ Enterprise Cardiovascular Diagnostic System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg)

An enterprise-grade, machine-learning-powered cardiovascular disease prediction system. Developed as the final deliverable for the **BCA CACS452 Project III** curriculum.

---

## 📖 Project Overview
This project implements a fully functional, end-to-end Machine Learning pipeline. It is designed for clinical use, utilizing a custom-tuned Random Forest ensemble to predict cardiovascular disease risk based on 11 patient vitals. 

Crucially, the classification threshold has been medically tuned to **0.40** to prioritize **Recall** (minimizing False Negatives). This ensures high-risk patients are accurately flagged for medical consultation, prioritizing patient safety over raw statistical accuracy.

## ✨ Key Features
*   **PostgreSQL ETL Pipeline:** Automated extraction, biological outlier cleaning, and database loading.
*   **Single Patient Assessment:** Real-time risk probability calculation featuring **Explainable AI** (dynamically generated Feature Importance visuals).
*   **Clinical Treatment Simulator:** Interactive sliders allowing doctors to simulate how weight loss, blood pressure reduction, and smoking cessation will impact the patient's risk in real-time.
*   **Hospital Bulk Processing:** Upload CSV files containing hundreds of patients for simultaneous processing and automatic triage report generation.
*   **Algorithm Challenger:** Extensive Jupyter Notebook analysis comparing Random Forest and XGBoost algorithms with detailed Confusion Matrices.

## 🛠️ Technology Stack
*   **Frontend UI:** Streamlit
*   **Machine Learning:** Scikit-Learn (Random Forest), XGBoost
*   **Data Processing:** Pandas, NumPy
*   **Database:** PostgreSQL (SQLAlchemy, Psycopg2)
*   **Visualizations:** Matplotlib, Seaborn

---

## 🚀 How to Run Locally

### 1. Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Database Setup
Ensure PostgreSQL is installed and running on your machine. Update the credentials in `db.py` and `etl_pipeline.py` if your local Postgres password is not default.

Run the script to create the database:
```bash
python db.py
```

### 3. Run the ETL Pipeline
Extract the Kaggle dataset (`cardio_train.csv`), clean out biological impossibilities (e.g., negative blood pressures), and load the clean data into PostgreSQL:
```bash
python etl_pipeline.py
```

### 4. Run the Web Application
Start the Streamlit hospital dashboard:
```bash
streamlit run app.py
```

---

## 📊 Data Source
The model was trained on the **Cardiovascular Disease dataset** by Svetlana Ulianova (70,000 records). 
Available on [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).
