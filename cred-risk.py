import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import pickle
from keras.models import Sequential
from keras.layers import Dense

# Load datasets
df1 = pd.read_csv("/kaggle/input/credit-risk-classification-dataset/customer_data.csv")
df2 = pd.read_csv("/kaggle/input/credit-risk-classification-dataset/payment_data.csv")

# Merge on 'id'
df2['update_date'] = pd.to_datetime(df2['update_date'])
df2['report_date'] = pd.to_datetime(df2['report_date'])
df2['year'] = df2['report_date'].dt.year
df2['month'] = df2['report_date'].dt.month
df2['day'] = df2['report_date'].dt.day
merged_df = df1.merge(df2, on='id').drop(columns=['update_date', 'report_date'])

# Handle missing values using KNN imputation
imputer = KNNImputer(n_neighbors=3)
cols_to_impute = ['prod_limit', 'highest_balance', 'fea_2']
merged_df[cols_to_impute] = imputer.fit_transform(merged_df[cols_to_impute])

# Correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(merged_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation")
plt.show()

# Split dataset into features (X) and target variable (y)
X = merged_df.drop(columns=['label'])
y = merged_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42)
}

# Train and evaluate models
scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores[name] = round(accuracy_score(y_test, y_pred) * 100, 2)
    pickle.dump(model, open(f'{name.replace(" ", "_").lower()}.pkl', 'wb'))

# Train a simple neural network
nn_model = Sequential()
nn_model.add(Dense(11, activation='relu', input_dim=X_train.shape[1]))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=300, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
scores["Neural Network"] = round(accuracy_score(y_test, y_pred_nn) * 100, 2)
nn_model.save("neural_network_model.h5")

# Visualize accuracy scores
plt.figure(figsize=(10, 5))
sns.barplot(x=list(scores.keys()), y=list(scores.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.show()

# Print accuracy scores
for model, score in scores.items():
    print(f"The accuracy score achieved using {model} is: {score}%")
