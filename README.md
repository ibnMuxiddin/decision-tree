# 💊 Bemorlarga Dori Tavsiya Qilish Tizimi (Medicine Recommendation System)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ushbu loyiha bemorlarning yoshi, jinsi, qon bosimi, xolesterin miqdori hamda qondagi natriy-kaliy nisbati (`Na_to_K`) ko'rsatkichlari bo'yicha eng mos keluvchi dori vositasini (`drugA`, `drugB`, `drugC`, `drugX`, `drugY`) tavsiya qiluvchi **Decision Tree** va **Random Forest** machine learning tizimidir.

---

## 🎯 Asosiy Xususiyatlari (Key Features)

- **Qayta ishlanuvchi ma'lumotlar quvuri (`src/preprocessing.py`)**: Ordinal va katagorial alomatlarni to'g'ri kodlash hamda pipeline obyektlarini saqlash.
- **Yuqori Aniqlikdagi Modellash (`src/train.py`)**: Decision Tree (98.33% aniqlik) va Random Forest ensamble modellari.
- **Interaktiv CLI Interfeysi (`src/predict.py`)**: Buyruqlar satri orqali yangi bemor profilini kiritish va dori bashoratini olish.
- **Zamonaviy Web Ilova (`src/app.py`)**: Streamlit freymvorkida yaratilgan, grafiklar va ehtimolliklar taqsimotini ko'rsatuvchi foydalanuvchiga qulay interfeys.
- **Visual Hisobotlar (`reports/figures/`)**: Decision Tree daraxti, Confusion Matrix va alomatlar muhimligi grafiklari.

---

## 📊 Model Natijalari (Model Performance)

| Model | Aniqlik (Test Accuracy) | Weighted F1-Score | 5-Fold Cross-Validation |
|---|:---:|:---:|:---:|
| **Decision Tree Classifier** | **98.33%** | **0.9830** | **98.00%** |
| **Random Forest Classifier** | **98.33%** | **0.9830** | **98.50%** |

### Alomatlar Muhimligi (Feature Importance)
Model qaror qabul qilishida eng yuqori o'rinni **`Na_to_K` (Natriy/Kaliy nisbati)** egallaydi, undan keyin **`BP` (Qon bosimi)** va **`Age` (Yosh)** turadi.

---

## 📁 Loyiha Tuzilishi (Project Structure)

```text
decision-tree/
├── data/
│   ├── raw/
│   │   └── drug200.csv              # Asosiy xom dataset
│   └── processed/
│       └── drug200_transformed.csv  # Qayta ishlangan dataset
├── models/
│   ├── preprocessor.pkl             # Pipelines & Encoders
│   ├── DecisionTree.pkl             # O'rgatilgan Decision Tree modeli
│   └── RandomForest.pkl             # O'rgatilgan Random Forest modeli
├── notebooks/
│   ├── 01_eda.ipynb                 # Dastlabki tahlil va eksploratsiya
│   ├── 02_data_preparation.ipynb    # Ma'lumotlarni tozalash va tayyorlash
│   ├── 03_ML.ipynb                  # Decision Tree modelini qurish
│   └── 04_ML_RandomForest.ipynb     # Random Forest modeli va taqqoslash
├── reports/
│   ├── model_performance.md         # Batafsil model hisoboti
│   └── figures/
│       ├── confusion_matrix.png     # Matritsa grafigi
│       ├── decision_tree.png        # Daraxt visualizatsiyasi
│       └── feature_importance.png   # Alomatlar ko'rsatkichlari
├── src/
│   ├── preprocessing.py             # Preprocessing moduli
│   ├── train.py                     # O'rgatish va baholash skripti
│   ├── predict.py                   # CLI inference ilovasi
│   └── app.py                       # Interaktiv Web Ilova (Streamlit)
├── .gitignore
├── README.md                        # Loyiha hujjatlari
└── requirements.txt                 # Kutubxonalar ro'yxati
```

---

## 🚀 O'rnatish va Ishga Tushirish (Installation & Setup)

### 1. Repozitoriyani klonlash
```bash
git clone https://github.com/ibnMuxiddin/decision-tree.git
cd decision-tree
```

### 2. Virtual muhitni yaratish va kutubxonalarni o'rnatish
```bash
python -m venv env
# Windows:
env\Scripts\activate
# Linux/macOS:
source env/bin/activate

pip install -r requirements.txt
```

### 3. Modelni o'rgatish va baholash
```bash
python src/train.py
```

---

## 💻 Ishlatish (Usage)

### CLI (Buyruqlar Satri Orqali):
```bash
python src/predict.py --age 47 --sex M --bp LOW --cholesterol HIGH --na_to_k 13.093
```

### Web Ilovani Ishga Tushirish:
```bash
streamlit run src/app.py
```
Brauzerda avtomatik ravishda `http://localhost:8501` manzili ochiladi.

---

## 📝 Muallif
- **Ibn Muxiddin** - [GitHub Profile](https://github.com/ibnMuxiddin)
