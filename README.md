# 🏠 California Housing Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-FF6F00)
![Flask](https://img.shields.io/badge/Flask-WebApp-black)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An end-to-end **Machine Learning project** that predicts California housing prices using supervised, unsupervised, and deep learning models.

This project is built for **academic coursework** and **professional portfolio demonstration**, showcasing a complete ML lifecycle from data preprocessing to deployment.

---

## 📌 Project Highlights

✔ End-to-End ML Pipeline  
✔ Regression & Classification Models  
✔ Support Vector Machine (SVM)  
✔ Neural Network (TensorFlow/Keras)  
✔ Clustering & PCA  
✔ Model Serialization  
✔ Flask Web Deployment  
✔ Modular & Scalable Structure  

---

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Models Implemented](#-models-implemented)
- [Web Application](#-web-application)
- [Technologies Used](#-technologies-used)
- [Example Prediction](#-example-prediction)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## 📖 Introduction

The **California Housing ML Project** analyzes housing data and builds predictive models to estimate median house values.

The project follows a structured machine learning workflow:

1. Data Cleaning & Preprocessing  
2. Exploratory Data Analysis  
3. Feature Engineering  
4. Model Training & Evaluation  
5. Model Serialization  
6. Web Deployment with Flask  

---

## 🗂 Project Structure

```
California_Housing_ML_Project/
│
├── models/
│   ├── regression_model.pkl
│   ├── classifier_model.pkl
│   ├── svm_model.pkl
│   ├── neural_network_model.h5
│   ├── scaler.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing_eda.ipynb
│   ├── 02_regression_models.ipynb
│   ├── 03_classification_models.ipynb
│   ├── 04_svm_model.ipynb
│   ├── 05_neural_network.ipynb
│   └── 06_clustering_pca.ipynb
│
├── utils/
│   └── data_loader.py
│
├── web_app/
│   ├── app.py
│   ├── templates/
│   └── static/
│
└── report/
```

---

## ⚙ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/california-housing-ml.git
cd california-housing-ml
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow flask joblib
```

---

## 🚀 Usage

### 📊 Run Jupyter Notebooks

```bash
jupyter notebook
```

Run notebooks in the following order:

1. Data Preprocessing & EDA  
2. Regression Models  
3. Classification Models  
4. SVM Model  
5. Neural Network  
6. Clustering & PCA  

---

### 🌐 Run Flask Web Application

```bash
cd web_app
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 🔬 Machine Learning Workflow

### 🧹 Data Preprocessing
- Handling missing values  
- Feature scaling (StandardScaler)  
- Feature engineering  
- Train-test split  

### 📊 Exploratory Data Analysis
- Correlation matrix  
- Distribution visualization  
- Outlier detection  
- Feature importance analysis  

### 🤖 Model Training
- Cross-validation  
- Hyperparameter tuning  
- Model comparison  

### 📈 Model Evaluation

**Regression Metrics**
- MAE  
- MSE  
- RMSE  

**Classification Metrics**
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 🤖 Models Implemented

### 📈 Regression
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  

### 📊 Classification
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

### 🧮 Support Vector Machine
- SVM Classifier  

### 🧠 Neural Network
- Fully Connected Deep Neural Network  
- Implemented using TensorFlow / Keras  

### 📌 Unsupervised Learning
- K-Means Clustering  
- Principal Component Analysis (PCA)  

---

## 🌐 Web Application

The Flask web application allows users to:

- Input housing features  
- Select prediction model  
- Generate real-time predictions  
- View results in a clean web interface  

Models are dynamically loaded from serialized files in the `models/` directory.

---

## 🛠 Technologies Used

- Python 3.8+  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Flask  
- Joblib  

---

## 💡 Example Prediction

**Input Features**
- Median Income  
- House Age  
- Average Rooms  
- Average Bedrooms  
- Population  
- Average Occupancy  
- Latitude  
- Longitude  

**Output**
- Predicted Median House Value  
- Classification Label (High / Low Value)  

---

## 🛠 Troubleshooting

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Activate virtual environment |
| Model not loading | Verify file paths in `app.py` |
| TensorFlow errors | Install compatible TensorFlow version |
| Flask not running | Ensure port 5000 is available |

---

## 🚀 Future Improvements

- Hyperparameter Optimization (GridSearch / RandomSearch)  
- Docker Containerization  
- CI/CD Integration  
- Cloud Deployment (AWS / Azure / GCP)  
- REST API Versioning  

---

## 👨‍💻 Author

**Tanishq Sinha**  
Machine Learning Project 

---

## 📜 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, consider giving it a star!
