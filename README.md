# EduPredict
EduPredict
Explainable AI-Based Student Performance Prediction System
EduPredict is a web-based student performance regression and early-warning system. It estimates a student's final continuous-assessment score (0–100) using student profile information, course enrolment details, online learning activity, and assessment information available during the first 60 days of a course.

The project combines a Python machine-learning pipeline, a Flask inference server, and a browser-based dashboard. It is intended to support early review and academic intervention. Its predictions are advisory estimates, not official grades or decisions.
Important: EduPredict predicts a continuous-assessment aggregate. It does not predict an examination-only mark, and it is not a pass/fail classification model.

Table of Contents
- Key Features
- How It Works
- Dataset
- Feature Engineering
- Models
- Evaluation Results
- Explainable AI
- Technology Stack
- Project Structure
- Installation
- Running the Project
- API Overview
- Prediction Categories
- Limitations
- Responsible Use
- Author
Key Features
- Individual prediction: enter student, course, engagement, and early-assessment details to estimate a score.
- Early-warning status: translates the predicted score into At risk, Average, or Good categories.
- Rule-based insight flags: presents supporting indicators based on the supplied information.
- Batch prediction: upload a CSV containing multiple student records and receive predictions for the cohort.
- Performance dashboard: view saved model evaluation results and generated analysis visualisations.
- Global feature importance: shows permutation feature importance for the XGBoost model component.
- Responsive browser interface: a dark, glass-style dashboard built with HTML, CSS, and JavaScript.
How It Works
1. Load source data. The training pipeline reads OULAD student, registration, course, assessment, and VLE tables.
2. Construct the target. It calculates a continuous-assessment aggregate from submitted non-exam assessments.
3. Build early-course features. It aggregates VLE activity and assessment information available up to day 60.
4. Prepare the data. It cleans and joins records, creates 43 model features, and removes records withdrawn on or before day 60.
5. Split by student. Student IDs are grouped so records belonging to the same student stay in one partition.
6. Train and compare models. Six regression configurations are evaluated using the same data partitions.
7. Save model artefacts. The preprocessing pipeline, selected model components, metadata, results, and plots are saved.
8. Serve predictions. Flask loads the saved artefacts, processes requests, and returns predictions to the browser.
9. Display results. The frontend presents the estimated score, status, insights, batch output, and performance information.
Dataset
EduPredict uses the Open University Learning Analytics Dataset (OULAD), which contains information about students taking online distance-learning courses.

The project uses these source tables:
OULAD table	Role in EduPredict
studentInfo.csv	Student background and module enrolment information
studentRegistration.csv	Registration timing and early-withdrawal filtering
courses.csv	Course presentation duration
assessments.csv	Assessment types, dates, and weights
studentAssessment.csv	Submitted assessment scores and dates
vle.csv	Maps learning-site IDs to activity categories
studentVle.csv	Daily virtual learning environment click records


Dataset scale in the documented training run
Measure	Count
Raw studentInfo.csv rows	32,593
Raw unique student IDs	28,785
Usable student-course records	26,361
Unique students represented in modelling data	23,985
Student VLE click records	10,655,280
Module presentations	22


A student can appear in multiple student-course records. Therefore, 26,361 is the number of modelling records, not the number of unique students.

The training pipeline processes the large studentVle.csv file in chunks to reduce memory usage.
What the Model Predicts
The target is named final_score in the training code. It is a score from 0 to 100 created from submitted non-exam assessments:
- Exam assessments are excluded.
- For most modules, submitted non-exam scores are aggregated using assessment weights.
- Module GGG receives special handling because its assessment weights are recorded as zero; the project uses the mean of submitted non-exam scores for that module.
- The result is clipped to the 0–100 range.
This target is not OULAD's final_result label and is not an examination mark.
Feature Engineering
The model uses 43 engineered inputs:
- 8 categorical features: gender, age_band, highest_education, region, imd_band, disability, code_module, and semester.
- 35 numerical features covering previous attempts, studied credits, registration timing, course duration, VLE engagement, activity patterns, early assessment performance, submission timing, and assessment completion.
VLE activity types are grouped into six categories: forum, content, quiz, resource, home, and other.

The live form exposes 25 fields for manual entry. The server fills the remaining 18 model inputs from saved training-data defaults when they are not supplied. These defaults are cohort-derived fallback values, not personalised values. In particular, changing a visible total does not automatically recompute every hidden derived feature.
Models
The training pipeline compares six regression configurations:
1. Linear Regression
2. Random Forest Regressor
3. HistGradientBoosting Regressor
4. XGBoost Regressor
5. Deep Learning tabular residual MLP
6. Hybrid Ensemble (XGBoost + HistGradientBoosting + MLP)
The neural network is a tabular multilayer perceptron with dense layers, batch normalisation, SiLU activations, dropout, and residual connections. It is not a CNN, RNN, or transformer.

The saved hybrid prediction is a fixed weighted combination:
Ensemble score = 0.40 × XGBoost
               + 0.30 × HistGradientBoosting
               + 0.30 × MLP
The weights are fixed in the training and inference code; they are not learned by a separate ensemble-fitting process.
Evaluation Results
The documented holdout-test results are:
Model	R²	MAE	RMSE
Linear Regression	0.6042	14.300	19.150
Random Forest	0.6188	13.982	18.794
HistGradientBoosting	0.6284	13.777	18.556
XGBoost	0.6305	13.682	18.504
Deep Learning (MLP)	0.6095	14.349	19.023
Hybrid Ensemble (DL + Boost)	0.6333	13.755	18.434


The selected configuration is the Hybrid Ensemble because it has the highest stored test R² in this run.

Metric interpretation: R² = 0.6333 is not “63.33% accuracy.” R² is a regression goodness-of-fit metric. The hybrid model's MAE is about 13.76 score points, so individual predictions can differ meaningfully from actual outcomes. The documented percentage of predictions within ±10 points is 47.8%.

The data split is grouped by student ID into approximately 70% training, 15% validation, and 15% holdout test records. This evaluates held-out students within the dataset distribution; it does not establish performance at another institution or in a future academic year.
Explainable AI
The project calculates global permutation feature importance for the XGBoost component:
1. Measure baseline MAE on the holdout test data.
2. Shuffle one input feature while leaving other features unchanged.
3. Transform the modified data with the fitted preprocessor.
4. Recalculate XGBoost predictions and MAE.
5. Repeat the shuffle three times and average the MAE increase.
The largest documented importance values are:
Feature	MAE increase after shuffling
early_weight_ratio	8.066
code_module	4.414
avg_days_early	0.781
early_pts_earned	0.671
active_days	0.637


These are global model-importance measurements, not causal effects. The current implementation does not provide SHAP, LIME, counterfactual explanations, or per-student feature attribution. The importance scores are calculated for XGBoost, not for the hybrid ensemble as a whole.
Technology Stack
- Language: Python
- Machine learning: scikit-learn, XGBoost
- Deep learning: TensorFlow / Keras
- Data processing: pandas, NumPy
- Model persistence: joblib, Keras model files
- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- Visualisation: Matplotlib, Seaborn
Project Structure
EduPredict/
├── data/
│   ├── anonymisedData.zip
│   ├── oulad_sample.csv
│   └── oulad/
│       ├── studentInfo.csv
│       ├── studentRegistration.csv
│       ├── courses.csv
│       ├── assessments.csv
│       ├── studentAssessment.csv
│       ├── vle.csv
│       └── studentVle.csv
├── models/
│   ├── preprocessor.joblib
│   ├── xgboost.joblib
│   ├── hgb.joblib
│   ├── mlp.keras
│   └── meta.json
├── models_oulad/          # Fallback/duplicate model artefacts
├── outputs/               # Evaluation results and plots
├── outputs_oulad/         # Fallback/duplicate output artefacts
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── train.py
├── train_oulad.py
├── server.py
├── requirements.txt
├── LICENSE
└── README.md
The exact files present may vary by repository revision. The documented project archive contains the paths shown above. The models_oulad/ and outputs_oulad/ directories are fallback copies, not independently trained model versions.
Installation
1. Clone the repository
git clone https://github.com/AkshKumar2/EduPredict.git
cd EduPredict
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Use a Python and library environment compatible with the versions in requirements.txt and the saved model artefacts.
4. Check the dataset
The training pipeline looks for the extracted OULAD CSV files in recognised data directories. If the extracted files are not present but data/anonymisedData.zip is available, the training code can extract the archive into data/oulad.

If the dataset is not included in your clone, obtain OULAD from its official source and place the CSV files in data/oulad/ using the filenames listed above.
Running the Project
Run the web application
python server.py
Open the local address printed by Flask in your browser. The server loads the saved model artefacts and metadata; make sure the expected model files are present.
Train the models again
python train.py
train_oulad.py is a small entry-point wrapper that imports and runs the main training function from train.py:
python train_oulad.py
Training requires the OULAD source tables and can take substantially longer than starting the already-trained web application.
API Overview
The Flask backend serves the browser interface and exposes endpoints for metadata, prediction, batch processing, and saved evaluation results. The documented application includes these API routes:
Endpoint	Purpose
GET /api/meta	Returns form metadata, feature definitions, and defaults
POST /api/predict	Generates an individual prediction
POST /api/batch	Processes a batch of student records
GET /api/results	Returns saved model evaluation results


Check server.py for the exact request and response schemas in the current code revision.
Prediction Categories
The interface applies these thresholds after the regression model produces a numerical prediction:
Predicted score	Interface status
Below 40	At risk
40 to below 70	Average
70 to 100	Good / Distinction in the individual prediction view


These thresholds are application rules, not a separately trained classification model.
Limitations
- The system does not connect to a live university student information system.
- Records withdrawn on or before day 60 are excluded, so the model is not an early-withdrawal predictor.
- Early assessment marks are used as predictors and also contribute to the constructed continuous-assessment target. The task is therefore forecasting the final coursework aggregate from partial coursework information, not predicting an entirely unseen examination result.
- The manual form exposes only 25 of the 43 features. Missing inputs are filled using stored cohort defaults, and hidden derived fields are not automatically recalculated from every visible form value.
- The reported test results do not prove generalisation to other universities, courses, or future academic years.
- Global permutation importance is not a causal explanation or an individual student's explanation.
- Predictions may be affected by data quality, missing context, and circumstances not represented in OULAD.
Responsible Use
EduPredict should be used as a decision-support and educational review aid only. A low predicted score should prompt supportive human review, not an automatic penalty or judgement about a student's ability, effort, or motivation. Do not use the model as the sole basis for grading, admission, progression, disciplinary action, or other high-impact decisions.

Handle student information securely, minimise personal data, and obtain appropriate permission before using real student records.
Author
Aksh Kumar
School of Computer Science and Engineering
Lovely Professional University
Phagwara, India
Email: akshkumarnew@gmail.com
License
This repository includes an MIT license. See LICENSE for the license text.
