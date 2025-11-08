# ==========================================================
# Predict Diabetes — End-to-End Pipeline
# Using predict_diabetes library
# ==========================================================
from predict_diabetes.src.predict_diabetes.load_and_split import load_data, split_data
from predict_diabetes.src.predict_diabetes.preprocess import preprocess_data
from predict_diabetes.src.predict_diabetes.train import train_model
from predict_diabetes.src.predict_diabetes.predict import evaluate_model, add_prediction_probabilities
from predict_diabetes.src.predict_diabetes.metrics import compute_roc_auc

df = load_data("sample_diabetes_mellitus_data.csv")

train, test = split_data(df)
train, test = preprocess_data(train, test)
model, X_train, X_test, y_train, y_test = train_model(train, test)
accuracy = evaluate_model(model, X_test, y_test)
train, test = add_prediction_probabilities(model, X_train, X_test, train, test)
train_auc, test_auc = compute_roc_auc(y_train, y_test, train, test)
