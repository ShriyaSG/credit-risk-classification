# Credit Risk Classification

## Project Overview
This project focuses on **credit risk assessment** using machine learning models. It leverages customer and payment data to predict the likelihood of a customer defaulting on a loan. Various classification models, including logistic regression, decision trees, random forests, Naïve Bayes, XGBoost, and neural networks, are trained and evaluated to find the most accurate predictor.

## Dataset
The project uses two datasets:  
1. **Customer Data (`customer_data.csv`)** – Contains customer demographics and financial details.  
2. **Payment Data (`payment_data.csv`)** – Includes historical payment records and credit behavior.

## Steps Involved
### 1. Data Preprocessing
- Handling missing values using **KNN Imputer**.
- Converting date columns to datetime format.
- Merging customer and payment datasets on a common `id` field.
- Removing irrelevant or highly correlated features.

### 2. Exploratory Data Analysis (EDA)
- Checking for missing values.
- Visualizing feature distributions with **box plots**.
- Generating a **correlation heatmap** to understand relationships.

### 3. Feature Engineering
- Rounding off numerical features for consistency.
- Encoding categorical variables if necessary.

### 4. Model Training & Evaluation
- Splitting data into training (80%) and testing (20%) sets.
- Training the following models:
  - Logistic Regression
  - Naïve Bayes
  - Decision Tree
  - Random Forest
  - XGBoost
  - Neural Network (Sequential model with Dense layers)
- Evaluating models using **accuracy score**.

### 5. Model Comparison & Visualization
- Storing model accuracy scores.
- Plotting **bar charts** to compare model performance.

### 6. Model Persistence
- Saving the best-performing models using `pickle` for future use.

## Results
- **XGBoost** and **Neural Networks** performed the best, achieving high accuracy scores.
- A **Neural Network with 11 nodes and 300 epochs** produced competitive results.

## Requirements
- Python 3.x  
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `pickle`

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow keras
   ```
2. Run the script:
   ```bash
   python credit_risk_ml.py
   ```
3. The trained models (`model.pkl`, `model2.pkl`) will be saved for future predictions.

## Author
- **Shriya Sengupta**  
