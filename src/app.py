import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import load_preprocessor, FEATURE_COLS

st.set_page_config(
    page_title="Bemorlarga Dori Tavsiya Tizimi",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2a5298;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .recommendation-box {
        background: linear-gradient(135deg, #00b09b to #96c93d);
        background-color: #e8f5e9;
        border-radius: 12px;
        padding: 1.5rem;
        border: 2px solid #66bb6a;
        text-align: center;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_prep():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_path = os.path.join(base_dir, "models", "preprocessor.pkl")
    dt_path = os.path.join(base_dir, "models", "DecisionTree.pkl")
    rf_path = os.path.join(base_dir, "models", "RandomForest.pkl")
    
    if not os.path.exists(prep_path):
        prep_path = "models/preprocessor.pkl"
        dt_path = "models/DecisionTree.pkl"
        rf_path = "models/RandomForest.pkl"
        
    prep = load_preprocessor(prep_path)
    dt_model = joblib.load(dt_path)
    rf_model = joblib.load(rf_path)
    
    return prep, dt_model, rf_model

def main():
    st.markdown('<div class="main-header">💊 Bemorlarga Dori Tavsiya Tizimi</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Decision Tree & Random Forest Machine Learning modellari asosida sun\'iy intellektual dori tavsiya platformasi</div>', unsafe_allow_html=True)

    try:
        preprocessor, dt_model, rf_model = load_models_and_prep()
    except Exception as e:
        st.error(f"Modellarni yuklashda xatolik yuz berdi: {e}. Iltimos avval `python src/train.py` buyrug'ini yuriting.")
        return

    # Sidebar inputs
    st.sidebar.header("🩺 Bemor ko'rsatkichlari")
    
    age = st.sidebar.slider("Yosh (Age)", min_value=15, max_value=85, value=45, step=1)
    sex = st.sidebar.radio("Jins (Sex)", options=["M", "F"], format_func=lambda x: "Erkak (M)" if x=="M" else "Ayol (F)")
    bp = st.sidebar.selectbox("Qon bosimi (BP)", options=["LOW", "NORMAL", "HIGH"], index=2)
    cholesterol = st.sidebar.selectbox("Xolesterin (Cholesterol)", options=["NORMAL", "HIGH"], index=1)
    na_to_k = st.sidebar.number_input("Na_to_K (Natriy/Kaliy nisbati)", min_value=5.0, max_value=40.0, value=15.2, step=0.1)

    model_choice = st.sidebar.radio("🤖 Mashinali o'rgatish modeli", options=["Decision Tree", "Random Forest"])

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Tavsiya va Bashorat", "📊 Model Visualizatsiyasi", "📁 Dataset Tahlili", "ℹ️ Loyiha Haqida"])

    with tab1:
        st.subheader("📋 Bemor profili va tavsiya etilgan dori")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("##### Kiritilgan ko'rsatkichlar:")
            st.write(f"- **Yosh:** {age} yosh")
            st.write(f"- **Jins:** {'Erkak' if sex=='M' else 'Ayol'} ({sex})")
            st.write(f"- **Qon bosimi:** {bp}")
            st.write(f"- **Xolesterin:** {cholesterol}")
            st.write(f"- **Na/K nisbati:** {na_to_k:.3f}")

        # Prediction
        selected_model = dt_model if model_choice == "Decision Tree" else rf_model
        sample_features = preprocessor.transform_sample(age, sex, bp, cholesterol, na_to_k)
        
        prediction = selected_model.predict(sample_features)[0]
        probabilities = selected_model.predict_proba(sample_features)[0]
        classes = selected_model.classes_

        with col2:
            st.markdown("##### 🎯 Model tavsiyasi:")
            st.success(f"### Tavsiya etilgan dori: **{prediction}**")
            st.caption(f"Tanlangan model: {model_choice}")

        st.divider()
        st.subheader("📈 Har bir dori bo'yicha ishonch ehtimolligi (Probabilities)")
        
        prob_df = pd.DataFrame({
            "Dori turi": classes,
            "Ehtimollik (%)": [p * 100 for p in probabilities]
        }).sort_values(by="Ehtimollik (%)", ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sns.barplot(data=prob_df, x="Ehtimollik (%)", y="Dori turi", palette="crest", ax=ax)
        ax.set_xlim(0, 100)
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(f"{width:.1f}%", (width + 1, p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', fontsize=10, color='black', fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)

    with tab2:
        st.subheader("📊 Modellarning tuzilishi va baholanishi")
        
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            st.markdown("##### Decision Tree Daraxt Tuzilishi")
            tree_img_path = "reports/figures/decision_tree.png"
            if os.path.exists(tree_img_path):
                st.image(tree_img_path, use_container_width=True)
            else:
                st.info("Rasm topilmadi. `python src/train.py` ni ishga tushiring.")
                
        with fig_col2:
            st.markdown("##### Model Ehtimollik Matritsasi (Confusion Matrix)")
            cm_img_path = "reports/figures/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Rasm topilmadi.")

        st.markdown("##### Muhim alomatlar (Feature Importance)")
        fi_img_path = "reports/figures/feature_importance.png"
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, use_container_width=True)

    with tab3:
        st.subheader("📁 Drug200 Dataseti")
        raw_path = "data/raw/drug200.csv"
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path)
            st.dataframe(df_raw, use_container_width=True)
            
            st.markdown("##### Dataset statistikasi:")
            st.write(df_raw.describe())
        else:
            st.warning("Dataset fayli topilmadi.")

    with tab4:
        st.subheader("ℹ️ Loyiha haqida")
        st.markdown("""
        Ushbu loyiha bemorlarning fiziologik va labaratoriya ko'rsatkichlariga ko'ra eng mos keladigan dorini tavsiya etuvchi Decision Tree va Random Forest machine learning tizimidir.
        
        **Alomatlar:**
        - `Age`: Bemorning yoshi
        - `Sex`: Jinsi (F - Ayol, M - Erkak)
        - `BP`: Qon bosimi (LOW, NORMAL, HIGH)
        - `Cholesterol`: Xolesterin miqdori (NORMAL, HIGH)
        - `Na_to_K`: Qondagi natriyning kaliyga nisbati
        
        **Maqsad dorilar:** `drugA`, `drugB`, `drugC`, `drugX`, `drugY`
        """)

if __name__ == "__main__":
    main()
