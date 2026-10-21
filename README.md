# HVTM CARE: AI-Driven Pharmaceutical Inventory Forecasting System

HVTM CARE is an AI-powered pharmaceutical inventory forecasting and shortage prediction system designed to support smarter stock planning for essential medicines. The system combines machine learning, deep learning, and ensemble-based forecasting to predict future drug demand, identify possible shortage risks, and support data-driven procurement decisions.

This project focuses on improving pharmaceutical inventory visibility by analyzing historical drug usage patterns, forecasting future demand, and providing intelligent planning support through a unified web-based dashboard.

---

## Overview

Pharmaceutical inventory management is a critical challenge for hospitals, clinics, and pharmacies. Stock shortages can directly affect patient care, while overstocking can increase storage costs, wastage, and expiry-related losses.

HVTM CARE addresses these challenges by using AI-based forecasting models to estimate future pharmaceutical demand and support better inventory planning. The system is especially useful for identifying demand patterns, forecasting monthly requirements, and assisting healthcare institutions in making informed stock management decisions.

The system includes a frontend dashboard, a backend prediction API, trained forecasting models, data preprocessing workflows, model evaluation notebooks, and intelligent prediction logic.

---

## Key Features

### AI-Based Drug Demand Forecasting

Predicts future pharmaceutical demand using historical inventory and consumption data. The forecasting process supports monthly demand estimation and helps reduce both understocking and overstocking.

### Shortage Risk Support

Uses prediction outputs and inventory-related logic to support shortage risk identification. This helps users detect possible stock issues earlier and plan procurement decisions more effectively.

### Smart Inventory Planning

Supports inventory planning by estimating future required quantities based on demand patterns and forecasting results. This helps improve stock control and reduce unnecessary stock accumulation.

### NeuroStack Ensemble Forecasting

Uses a meta-learning inspired ensemble approach that combines multiple model outputs to improve prediction reliability. This allows the system to benefit from both classical machine learning models and deep learning models.

### Deep Learning Forecasting

Implements sequence-based forecasting models such as BiLSTM and GRU to capture time-dependent patterns, seasonal behavior, and changing demand trends.

### Classical Machine Learning Comparison

Includes traditional machine learning models such as Linear Regression, Random Forest, XGBoost, and optimized ensemble models to compare performance and strengthen forecasting decisions.

### Self-Learning Oriented Design

The system is designed with future self-learning capability in mind, allowing prediction logic to be improved over time using new inventory records and real-world consumption patterns.

### Unified Web Dashboard

Provides a simple and interactive interface for prediction, inventory record handling, result visualization, and decision support.

---

## AI and Forecasting Approach

HVTM CARE follows a multi-model forecasting approach instead of depending on a single prediction model.

The system uses:

- Historical pharmaceutical inventory data
- Data cleaning and preprocessing
- Feature engineering
- Classical machine learning models
- Deep learning forecasting models
- Ensemble and meta-learning logic
- Backend prediction engine
- Frontend result visualization

This approach improves forecasting stability and allows model outputs to be compared, combined, and evaluated before being used in the final prediction workflow.

---

## Models Used

The project includes several machine learning and deep learning models:

- BiLSTM
- GRU
- NeuroStack MetaLearner
- Linear Regression
- Random Forest
- XGBoost
- Optimized Random Forest
- Base ensemble models

These models are used to analyze pharmaceutical demand patterns and support accurate inventory forecasting.

---

## System Workflow

Historical Drug Inventory Data  
↓  
Data Cleaning and Preprocessing  
↓  
Feature Engineering  
↓  
Machine Learning Model Training  
↓  
Deep Learning Model Training  
↓  
Model Evaluation and Comparison  
↓  
NeuroStack Meta-Learning Integration  
↓  
FastAPI Prediction Engine  
↓  
Web-Based Forecasting Dashboard  
↓  
Inventory Forecasting and Shortage Risk Support  

---

## Technical Stack

### Artificial Intelligence and Machine Learning

- Python
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- BiLSTM
- GRU
- Random Forest
- Linear Regression
- Ensemble Learning
- Meta-Learning Concepts

### Backend

- FastAPI
- Python
- Pandas
- NumPy
- Joblib
- REST API Architecture

### Frontend

- HTML
- CSS
- JavaScript
- Responsive Dashboard Interface

### Data Processing and Experimentation

- Jupyter Notebook
- Pandas
- NumPy
- Feature Engineering
- Data Preprocessing
- Model Evaluation
- Model Comparison

---

## Repository Structure

```text
hvtm-care-pharma-forecasting/
├── backend/
│   ├── app.py                # FastAPI prediction & data API
│   ├── config.py             # Path configuration & environment settings
│   └── engine_logic.py       # NeuroStack prediction & model orchestration
├── data/
│   ├── drugs.csv             # Primary pharmaceutical inventory dataset
│   └── drugs_featured.csv    # Feature-engineered dataset
├── frontend/
│   ├── css/                  # UI stylesheet & styling
│   ├── js/                   # UI handlers, state & API client
│   └── index.html            # Unified inventory web dashboard
├── models/                   # Serialized ML & DL model binaries (.pkl, .h5, .npy)
├── notebooks/                # Jupyter research, training & evaluation notebooks
├── scripts/                  # Utility scripts (brand check, date fixes)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # MIT License
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hashan-7/hvtm-care-pharma-forecasting.git
cd hvtm-care-pharma-forecasting

```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS / Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Backend Server

```bash
cd backend
uvicorn app:app --reload
```

The backend will run locally at:

```text
http://127.0.0.1:8000
```

### 6. Run the Frontend

Open the following file in a browser:

```text
frontend/index.html
```

You can also use a local development server such as the VS Code Live Server extension.

---

## Usage

1. Start the backend server.
2. Open the frontend dashboard.
3. Enter the required pharmaceutical inventory details.
4. Run the prediction.
5. View the forecasted demand and inventory planning results.
6. Use the output to support stock planning and shortage risk analysis.

---

## Dataset

The project uses pharmaceutical inventory datasets containing historical drug-related records. These datasets are used for data exploration, preprocessing, feature engineering, model training, evaluation, and forecasting.

The dataset supports the development of demand prediction models by capturing inventory-related patterns over time.

Main dataset files:

- `data/drugs.csv`
- `data/drugs_featured.csv`

---

## Model Evaluation

The project includes evaluation workflows to compare the performance of different forecasting models. Model evaluation helps identify which approaches provide better prediction behavior for pharmaceutical inventory demand.

Evaluation areas include:

- Forecasting accuracy
- Prediction error comparison
- Classical ML vs deep learning performance
- Ensemble model behavior
- Demand pattern analysis

---

## Security and Production Notes

This project is currently designed for academic, research, and development purposes.

Before production deployment, the following improvements are recommended:

- Replace demo authentication with secure backend authentication
- Restrict CORS origins
- Store sensitive configurations using environment variables
- Validate all user inputs
- Use a production-ready database
- Add proper logging and monitoring
- Add role-based access control
- Deploy backend and frontend using secure hosting services
- Add automated testing and CI/CD workflows

---

## Future Improvements

Possible future enhancements include:

- Real-time economic data integration
- Live currency rate monitoring
- Automated procurement cost estimation
- Database integration
- Advanced authentication
- Role-based dashboard access
- Cloud deployment
- Continuous model retraining
- Self-learning model updates
- Advanced shortage alert system
- Interactive analytics dashboard
- Report generation for pharmacy decision-making

---

## Project Purpose

HVTM CARE demonstrates how artificial intelligence can support pharmaceutical inventory forecasting, drug demand prediction, and shortage risk reduction.

The project highlights the practical use of machine learning, deep learning, and ensemble forecasting techniques in healthcare-related inventory management.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.