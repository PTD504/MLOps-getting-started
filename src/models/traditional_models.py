import pandas as pd
import numpy as np
import optuna  # Thêm Optuna cho tuning hyperparameter
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

def objective(trial, X_train, y_train, X_val, y_val, classifier_name):
    """Hàm mục tiêu cho Optuna hyperparameter tuning cho một classifier cố định"""
    # Khởi tạo hyperparameters chung
    max_features = trial.suggest_int("max_features", 5000, 20000)
    ngram_range = trial.suggest_categorical("ngram_range", [(1, 1), (1, 2), (1, 3)])
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    
    # Classifier với hyperparameter riêng tương ứng
    if classifier_name == "logistic_regression":
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        classifier = LogisticRegression(C=C, max_iter=1000, random_state=42)
    elif classifier_name == "svm":
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        classifier = LinearSVC(C=C, random_state=42, max_iter=1000)
    elif classifier_name == "naive_bayes":
        alpha = trial.suggest_float("alpha", 0.1, 10.0, log=True)
        classifier = MultinomialNB(alpha=alpha)
    else:
        raise ValueError(f"Classifier {classifier_name} không được hỗ trợ.")
    
    # Tạo pipeline
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    # Huấn luyện mô hình
    pipeline.fit(X_train, y_train)
    
    # Dự đoán trên tập validation
    y_pred = pipeline.predict(X_val)
    
    # Tính accuracy
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy

def train_and_evaluate_models():
    # Tạo thư mục lưu kết quả nếu chưa có
    os.makedirs("models", exist_ok=True)
    
    # Tải dữ liệu
    print("Đang tải dữ liệu đã xử lý...")
    try:
        train_data = pd.read_csv("data/train.csv")
        val_data = pd.read_csv("data/val.csv")
        test_data = pd.read_csv("data/test.csv")
    except FileNotFoundError:
        print("Không tìm thấy dữ liệu. Vui lòng chạy tiền xử lý trước!")
        return
    
    # Ghi nhận thông tin dataset qua MLflow (cho toàn bộ quá trình)
    mlflow.set_experiment("Traditional_Models_All")
    with mlflow.start_run(run_name="multiple_models_training"):
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("validation_samples", len(val_data))
        mlflow.log_param("test_samples", len(test_data))
        
        # Danh sách các classifier muốn thử
        classifiers = ["logistic_regression", "svm", "naive_bayes"]
        
        # Dữ liệu train và validation riêng
        X_train = train_data['text']
        y_train = train_data['label']
        X_val = val_data['text']
        y_val = val_data['label']
        
        # Dữ liệu train full (train + validation)
        X_train_full = pd.concat([train_data['text'], val_data['text']])
        y_train_full = pd.concat([train_data['label'], val_data['label']])
        
        # Duyệt qua từng loại mô hình
        for classifier_name in classifiers:
            print(f"\n🔍 Bắt đầu tuning cho mô hình {classifier_name}...")
            study = optuna.create_study(direction="maximize", study_name=f"traditional_models_{classifier_name}")
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, classifier_name), n_trials=5)
            
            best_params = study.best_params
            best_accuracy = study.best_value
            mlflow.log_param(f"{classifier_name}_best_params", best_params)
            mlflow.log_metric(f"{classifier_name}_best_validation_accuracy", best_accuracy)
            
            print(f"Hyperparameter tốt nhất cho {classifier_name}: {best_params}")
            print(f"Độ chính xác tốt nhất trên tập validation: {best_accuracy:.4f}")
            
            # Xây dựng lại mô hình tốt nhất dựa trên best_params
            max_features = best_params["max_features"]
            ngram_range = best_params["ngram_range"]
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            
            if classifier_name == "logistic_regression":
                classifier = LogisticRegression(C=best_params["C"], max_iter=1000, random_state=42)
            elif classifier_name == "svm":
                classifier = LinearSVC(C=best_params["C"], random_state=42, max_iter=1000)
            elif classifier_name == "naive_bayes":
                classifier = MultinomialNB(alpha=best_params["alpha"])
            
            # Tạo pipeline cho mô hình tốt nhất
            pipeline = Pipeline([
                ('tfidf', vectorizer),
                ('classifier', classifier)
            ])
            
            # Huấn luyện mô hình trên tập full (train + validation)
            print(f"🏋️‍♂️ Huấn luyện mô hình {classifier_name} trên tập train và validation...")
            pipeline.fit(X_train_full, y_train_full)
            
            # Đánh giá trên tập test
            print(f"Đang đánh giá mô hình {classifier_name} trên tập test...")
            y_pred = pipeline.predict(test_data['text'])
            test_accuracy = accuracy_score(test_data['label'], y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_data['label'], y_pred, average='binary'
            )
            
            mlflow.log_metric(f"{classifier_name}_test_accuracy", test_accuracy)
            mlflow.log_metric(f"{classifier_name}_test_precision", precision)
            mlflow.log_metric(f"{classifier_name}_test_recall", recall)
            mlflow.log_metric(f"{classifier_name}_test_f1", f1)
            
            # Sinh confusion matrix và lưu lại
            cm = confusion_matrix(test_data['label'], y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {classifier_name}')
            cm_path = f"{classifier_name}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()  # Đóng figure sau khi lưu
            mlflow.log_artifact(cm_path)
            
            # Lưu model
            model_path = f"models/{classifier_name}.joblib"
            joblib.dump(pipeline, model_path)
            mlflow.sklearn.log_model(pipeline, classifier_name)
            
            # Sinh báo cáo phân loại và lưu lại
            report = classification_report(test_data['label'], y_pred, target_names=['Negative', 'Positive'])
            report_path = f"models/{classifier_name}_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            
            # In kết quả đánh giá ra màn hình
            print(f"Mô hình {classifier_name} đã được huấn luyện thành công!")
            print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
if __name__ == "__main__":
    train_and_evaluate_models()