# EduPredict
# EduPredict
### Explainable AI-Based Student Performance Prediction System

EduPredict is a web-based student performance regression and early-warning system that predicts a student's final continuous-assessment score (0–100) using academic information, course enrolment details, online learning activity, and early assessment performance from the first 60 days of a course.

The project combines Machine Learning, Deep Learning, Explainable AI, and a Flask-based web application to provide data-driven insights into student performance.

## 🚀 Features

- **Individual Student Prediction:** Predict student performance using academic and engagement information.
- **Early Warning System:** Categorises predicted performance into At Risk, Average, and Good.
- **Explainable AI:** Uses permutation feature importance to identify influential predictors.
- **Batch Prediction:** Upload CSV files to generate predictions for multiple students.
- **Performance Dashboard:** Displays model evaluation metrics and visualisations.
- **Deep Learning Integration:** Implements a tabular residual Multilayer Perceptron (MLP).
- **Hybrid Ensemble:** Combines XGBoost, HistGradientBoosting, and Deep Learning predictions.
- **Interactive Interface:** Modern, responsive dashboard with real-time predictions.

## 🧠 How It Works

1. Collects student profile, course registration, assessment, and online learning activity data.
2. Processes student engagement records from the first 60 days.
3. Extracts and engineers 43 predictive features.
4. Applies preprocessing, including numerical scaling and categorical encoding.
5. Trains and evaluates multiple regression models.
6. Combines selected models using a weighted ensemble.
7. Generates predicted continuous-assessment scores.
8. Displays performance categories and insights through the web interface.

## 📊 Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset contains student demographic information, course registration records, assessment submissions, and Virtual Learning Environment (VLE) activity.

### Dataset Statistics

| Description | Count |
|---|---:|
| Unique Students | 23,985 |
| Student-Course Records | 26,361 |
| Raw Student Records | 32,593 |
| VLE Click Records | 10,655,280 |
| Module Presentations | 22 |

The modelling dataset contains 26,361 student-course records representing 23,985 unique students.

## ⚙️ Feature Engineering

The system uses **43 engineered features**, divided into:

- 8 categorical features
- 35 numerical features

Features include:

- Student demographic information
- Previous module attempts
- Registration timing
- Total platform clicks
- Active learning days
- Learning resource interactions
- Forum and quiz engagement
- Early assessment scores
- Assessment completion rate
- Submission timing
- Learning engagement trends

VLE activities are grouped into six categories:

- Forum
- Content
- Quiz
- Resource
- Homepage
- Other

## 🤖 Machine Learning Models

Six regression configurations are compared:

1. Linear Regression
2. Random Forest Regressor
3. HistGradientBoosting Regressor
4. XGBoost Regressor
5. Deep Learning (Tabular Residual MLP)
6. Hybrid Ensemble (DL + Boost)

### Deep Learning Architecture

The neural network is a tabular residual Multilayer Perceptron built using TensorFlow and Keras.

It incorporates:

- Dense layers
- Batch Normalisation
- SiLU activation functions
- Dropout regularisation
- Residual connections
- AdamW optimiser
- Mean Squared Error loss
- Early Stopping

### Hybrid Ensemble

The final ensemble combines three models using fixed weights:

```text
Ensemble Prediction =
    0.40 × XGBoost
  + 0.30 × HistGradientBoosting
  + 0.30 × Deep Learning MLP
```

## 📈 Model Evaluation

The models are evaluated using a student-grouped training, validation, and testing strategy.

### Performance Results

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.6042 | 14.300 | 19.150 |
| Random Forest | 0.6188 | 13.982 | 18.794 |
| HistGradientBoosting | 0.6284 | 13.777 | 18.556 |
| XGBoost | 0.6305 | 13.682 | 18.504 |
| Deep Learning (MLP) | 0.6095 | 14.349 | 19.023 |
| **Hybrid Ensemble** | **0.6333** | **13.755** | **18.434** |

The Hybrid Ensemble achieved the highest R² value among the evaluated configurations.

**Note:** R² is a regression evaluation metric, not classification accuracy. The model's predictions are estimates and can differ from actual student scores.

## 🔍 Explainable AI

EduPredict implements permutation feature importance to understand which input features influence model performance.

The technique works by:

1. Calculating baseline model performance.
2. Randomly shuffling one feature.
3. Measuring the change in prediction error.
4. Repeating the process multiple times.
5. Ranking features according to their importance.

### Top Predictive Features

| Feature | Importance (MAE Increase) |
|---|---:|
| Early Weight Ratio | 8.066 |
| Course Module | 4.414 |
| Average Days Early | 0.781 |
| Early Points Earned | 0.671 |
| Active Days | 0.637 |

The current implementation calculates global permutation importance for the XGBoost component. It does not currently implement SHAP, LIME, or individual student-level feature attribution.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Visualisation | Matplotlib, Seaborn |
| Model Persistence | Joblib, Keras |

## 📁 Project Structure

```text
EduPredict/
│
├── data/
│   ├── anonymisedData.zip
│   ├── oulad_sample.csv
│   └── oulad/
│
├── models/
│   ├── preprocessor.joblib
│   ├── xgboost.joblib
│   ├── hgb.joblib
│   ├── mlp.keras
│   └── meta.json
│
├── outputs/
│   ├── results.json
│   └── evaluation plots
│
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── train.py
├── train_oulad.py
├── server.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AkshKumar2/EduPredict.git
cd EduPredict
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python server.py
```

Open the local URL displayed in the terminal to access EduPredict.

### 5. Train the Models (Optional)

To train the models again using the OULAD dataset:

```bash
python train.py
```

Alternatively:

```bash
python train_oulad.py
```

The training process generates model artefacts, evaluation results, and visualisation files.

## 🌐 Application Modules

The web application provides four main sections:

| Module | Description |
|---|---|
| Live Predict | Individual student performance prediction |
| Batch Prediction | CSV-based prediction for multiple students |
| Performance | Model evaluation and performance visualisations |
| About | Project and dataset information |

## ⚠️ Limitations

- Predictions are based on historical OULAD data.
- The model predicts a continuous-assessment aggregate, not examination marks.
- Students who withdrew on or before day 60 are excluded from the modelling dataset.
- The live form exposes 25 of the 43 model features. Missing features use saved training-data defaults.
- The current explainability implementation provides global XGBoost feature importance rather than individual explanations.
- Model performance on other institutions or future academic years has not been independently established.

## 🔐 Responsible Use

EduPredict is designed as an educational decision-support tool.

Predictions should be treated as advisory estimates and should not replace human judgement. The system should not be used as the sole basis for grading, admission, academic progression, or disciplinary decisions.

Student information should be handled securely and used responsibly.

## 👨‍💻 Author

**Aksh Kumar**

School of Computer Science and Engineering  
Lovely Professional University  
Phagwara, India

📧 Email: akshkumarnew@gmail.com

🔗 GitHub: [AkshKumar2](https://github.com/AkshKumar2)

🔗 Repository: [EduPredict](https://github.com/AkshKumar2/EduPredict)

---

⭐ If you find this project interesting, consider giving the repository a star!
