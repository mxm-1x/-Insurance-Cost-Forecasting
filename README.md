# Insurance Cost Predictor

A production-grade Streamlit application for predicting health insurance costs based on individual profiles. The model is built using a Ridge Regression algorithm trained on US insurance data with log-transformed charges.

## Features 🚀

- **Personalized Predictions:** Enter your age, sex, number of children, smoking status, BMI, and region to get an estimated annual premium.
- **Feature Contributions (Explainable AI):** The app uses exact linear decomposition to show the precise impact of each feature on your final premium (e.g., how much your BMI or smoking status adds or subtracts from the cost).
- **What-If Analysis:** Automatically analyzes actionable changes (like quitting smoking or lowering BMI) and visualizes potential future savings.
- **Personalized Insights:** Provides dynamic, tailored tips based on user inputs to help mitigate insurance costs.
- **Modern UI:** Features a custom, premium glassmorphism-inspired dark mode UI with modern typography, interactive widgets, and clean data visualizations.

## Installation & Usage 💻

1. **Navigate to the core directory:**
   ```bash
   cd /path/to/ml
   ```

2. **Install the dependencies:**
   Make sure you have `pip` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

   *Note: Ensure that your trained model (`ridge_model.pkl` or `linear_model.pkl`), as well as your optionally fitted `scaler.pkl`, are placed in the same directory as the app to guarantee accurate execution.*

## Project Structure 📁

- `app.py`: The core Streamlit web application dashboard.
- `requirements.txt`: Python library dependencies (`streamlit`, `pandas`, `numpy`, `scikit-learn`).
- `Insurance Data.csv`: Raw medical insurance dataset.
- `ridge_model.pkl` / `redge_model.pkl`: Pre-trained regression models.
- `Insurance-Report.pdf`: Related research/report documents.

## Model Details 📊

- **Algorithms used:** Ridge Regression (fallback: Linear Regression)
- **Target Variable Transformation:** Expected to use `log1p` on target `charges` during training.
- **Preprocessing steps:** Includes one-hot encoding for categoricals (`sex`, `smoker`, `region`) and numerical input standardization using `StandardScaler`.
- **Feature Engineering:** Captured interactions such as `BMI × Smoking` and `Age × Smoking` to map nuanced real-world relationships.
