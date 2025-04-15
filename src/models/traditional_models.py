import pandas as pd
import numpy as np
import optuna  # Thêm Optuna cho hyperparameter tuning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import mlflow
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def objective(trial, X_train, y_train, X_val, y_val):
    """Hàm mục tiêu cho Optuna hyperparameter tuning"""
    # Đề xuất hyperparameters
    classifier_name = trial.suggest_categorical("classifier", ["logistic_regression", "svm", "naive_bayes"])
    max_features = trial.suggest_int("max_features", 5000, 20000)
    ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    
    # Classifier
    if classifier_name == "logistic_regression":
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        classifier = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif classifier_name == "svm":
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        classifier = LinearSVC(C=C, random_state=42, max_iter=1000)
    else:
        alpha = trial.suggest_float("alpha", 0.1, 10.0, log=True)
        classifier = MultinomialNB(alpha=alpha)
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_val)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def train_and_evaluate_models():
    # Start MLflow run
    with mlflow.start_run(run_name="traditional_models_training"):
        # Load processed data
        print("Đang tải dữ liệu đã xử lý...")
        try:
            train_data = pd.read_csv("data/train.csv")
            val_data = pd.read_csv("data/val.csv")
            test_data = pd.read_csv("data/test.csv")
        except FileNotFoundError:
            print("Không tìm thấy dữ liệu. Vui lòng chạy tiền xử lý trước!")
            return
        
        # Log dataset info
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("validation_samples", len(val_data))
        mlflow.log_param("test_samples", len(test_data))
        
        print("🔍 Bắt đầu hyperparameter tuning với Optuna...")
        # Hyperparameter tuning với Optuna
        study = optuna.create_study(direction="maximize", study_name="traditional_models")
        study.optimize(lambda trial: objective(
            trial, train_data['text'], train_data['label'], val_data['text'], val_data['label']
        ), n_trials=5)
        
        # Log optimized parameters
        best_params = study.best_params
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("best_validation_accuracy", study.best_value)
        
        print(f"Hyperparameter tốt nhất: {best_params}")
        print(f"Độ chính xác tốt nhất: {study.best_value:.4f}")
        
        # Build the best model
        print("Đang xây dựng mô hình tốt nhất...")
        
        max_features = best_params["max_features"]
        ngram_range = best_params["ngram_range"]
        classifier_name = best_params["classifier"]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        
        # Create classifier
        if classifier_name == "logistic_regression":
            classifier = LogisticRegression(C=best_params["C"], max_iter=1000, random_state=42)
            model_name = "logistic_regression"
        elif classifier_name == "svm":
            classifier = LinearSVC(C=best_params["C"], random_state=42, max_iter=1000)
            model_name = "svm"
        else:
            classifier = MultinomialNB(alpha=best_params["alpha"])
            model_name = "naive_bayes"
        
        # Create pipeline
        best_pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
        
        # Train on combined training and validation data
        print("🏋️‍♂️ Huấn luyện mô hình tốt nhất...")
        X_train_full = pd.concat([train_data['text'], val_data['text']])
        y_train_full = pd.concat([train_data['label'], val_data['label']])
        best_pipeline.fit(X_train_full, y_train_full)
        
        # Evaluate on test set
        print("Đánh giá trên tập test...")
        y_pred = best_pipeline.predict(test_data['text'])
        test_accuracy = accuracy_score(test_data['label'], y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_data['label'], y_pred, average='binary'
        )
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # Generate confusion matrix
        cm = confusion_matrix(test_data['label'], y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Save the confusion matrix
        os.makedirs("models", exist_ok=True)
        cm_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Save model
        model_path = f"models/{model_name}.joblib"
        joblib.dump(best_pipeline, model_path)
        mlflow.sklearn.log_model(best_pipeline, model_name)
        
        # Generate classification report
        report = classification_report(test_data['label'], y_pred, target_names=['Negative', 'Positive'])
        with open(f"models/{model_name}_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact(f"models/{model_name}_report.txt")
        
        print(f"Mô hình {model_name} đã được huấn luyện thành công!")
        print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    train_and_evaluate_models()
