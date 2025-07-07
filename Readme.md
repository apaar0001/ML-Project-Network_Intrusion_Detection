# 🚨 Network Intrusion Detection using Machine Learning

![Network Security Banner](https://img.freepik.com/free-vector/cyber-security-concept_23-2148532223.jpg)

<p align="center">
  <a href="https://github.com/your-repo/ML-Project-Network_Intrusion_Detection"><img src="https://img.shields.io/github/stars/your-repo/ML-Project-Network_Intrusion_Detection?style=social" alt="Stars"></a>
  <a href="https://github.com/your-repo/ML-Project-Network_Intrusion_Detection"><img src="https://img.shields.io/github/forks/your-repo/ML-Project-Network_Intrusion_Detection?style=social" alt="Forks"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Feature Engineering & Selection](#2-feature-engineering--selection)
  - [3. Model Training & Evaluation](#3-model-training--evaluation)
- [Results & Visualizations](#results--visualizations)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [References](#references)

---

## 📝 Introduction

This project leverages advanced machine learning techniques to detect network intrusions, classifying network traffic as either **normal** or **malicious**. The goal is to help organizations identify potential security threats and protect their networks from unauthorized access.

---

## 📊 Dataset

- **Source:** KDD Cup 1999 dataset (10% subset)
- **Features:** 41 attributes per record (e.g., protocol type, service, flag, bytes, etc.)
- **Classes:** Normal, DOS, R2L, U2R, Probing
- **Files:**
  - `kddcup.data_10_percent/kddcup.data_10_percent`
  - `data_with_column_names.csv`

---

## 🔄 Project Pipeline

### 1. Data Preprocessing

- **Missing Values:** Checked and confirmed none.
- **Class Imbalance:** Addressed using **SMOTEENN** (hybrid of SMOTE and Edited Nearest Neighbors).
- **Categorical Encoding:** One-hot encoding for `protocol_type`, `service`, and `flag`.
- **Scaling:** RobustScaler applied to features for outlier resistance.
- **Attack Categorization:** All attack types mapped to 5 main categories.

```python
# Example: Attack Categorization
attack_types = {
    'dos': ['smurf.', 'neptune.', 'back.', 'teardrop.', 'pod.', 'land.'],
    'r2l': ['warezclient.', 'guess_passwd.', 'imap.', 'warezmaster.', 'ftp_write.', 'phf.', 'spy.', 'multihop.'],
    'u2r': ['buffer_overflow.', 'loadmodule.', 'rootkit.', 'perl.'],
    'probe': ['satan.', 'ipsweep.', 'portsweep.', 'nmap.']
}
def categorize_attack_type(label):
    for k, v in attack_types.items():
        if label in v:
            return k.upper()
    return 'NORMAL'
df['outcome'] = df['outcome'].apply(categorize_attack_type)
```

### 2. Feature Engineering & Selection

- **Variance Threshold:** Remove low-variance features.
- **Correlation Analysis:** Remove highly correlated features.
- **Statistical Tests:** Rank Sum and Chi-Square tests for feature significance.
- **Boruta Algorithm:** Random Forest-based feature selection for robustness.

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
# Remove low-variance features
vt = VarianceThreshold(threshold=0.01)
X_var = vt.fit_transform(X)
# Select K best features
selector = SelectKBest(mutual_info_classif, k=20)
X_selected = selector.fit_transform(X_var, y)
```

### 3. Model Training & Evaluation

- **Models Explored:**
  - Decision Tree (J48)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Softmax Regression (Multinomial Logistic Regression)
  - Bagging & Boosting (AdaBoost, XGBoost, LightGBM)
  - Perceptron
  - Multi-layer Perceptron (MLP)
- **Hyperparameter Tuning:** GridSearchCV for key models.
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- **Ensemble Voting:** Combined predictions for improved accuracy.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
gs = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
gs.fit(X_train, y_train)
print('Best Params:', gs.best_params_)
```

---

## 📈 Results & Visualizations

- **Best Model Accuracy:** Up to ~85% (MLP, Decision Tree, LightGBM)
- **Confusion Matrix & ROC Curves:** Plotted for detailed analysis.
- **Class Distribution:** Visualized before and after resampling.

![Confusion Matrix Example](https://user-images.githubusercontent.com/your-image.png)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/ML-Project-Network_Intrusion_Detection.git
   cd ML-Project-Network_Intrusion_Detection
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main notebook:**
   - Open `final_main.ipynb` or `Main.ipynb` in Jupyter Notebook or JupyterLab.
   - Follow the cells sequentially to preprocess data, train models, and evaluate results.
4. **Use Pretrained Models:**
   - Pretrained models are available in the `models/` directory.
   - Example to load and predict:

```python
import joblib
model = joblib.load('models/decisiontreemodel.pkl')
pred = model.predict(X_test)
```

---

## 🗂 Project Structure

```
ML-Project-Network_Intrusion_Detection/
│
├── data_with_column_names.csv      # Preprocessed dataset with headers
├── kddcup.data_10_percent/        # Raw dataset
├── models/                        # Saved models and encoders
│   ├── decisiontreemodel.pkl
│   ├── lightgbm_model.pkl
│   └── ...
├── final_main.ipynb               # Main pipeline notebook
├── Main.ipynb                     # Data exploration and preprocessing
├── profiling.ipynb                # Data profiling and EDA
├── data_profiling_report.html     # HTML report from profiling
├── Project_Report_Final_9.pdf     # Project report
└── Readme.md                      # This file
```

---

## 👥 Contributors

- [Your Name](https://github.com/your-github)
- [Collaborator 1](https://github.com/collaborator1)
- [Collaborator 2](https://github.com/collaborator2)

---

## 📚 References

- [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [SMOTEENN Paper](https://doi.org/10.1109/ICMLA.2010.47)
- [Boruta Feature Selection](https://doi.org/10.1093/bioinformatics/btq134)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 🌟 Acknowledgements

Special thanks to the open-source community and contributors to the libraries used in this project.

---

<p align="center"><b>⭐️ If you find this project helpful, please star the repo! ⭐️</b></p>
