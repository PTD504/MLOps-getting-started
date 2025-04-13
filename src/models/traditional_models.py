import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import mlflow
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_models():
    # Start MLflow run
    with mlflow.start_run(run_name="traditional_models_training"):
        # Load processed data
        print("Loading processed data...")
        train_data = pd.read_csv("data/train.csv")
        val_data = pd.read_csv("data/val.csv")
        test_data = pd.read_csv("data/test.csv")
        
        # Log dataset info
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("validation_samples", len(val_data))
        mlflow.log_param("test_samples", len(test_data))
        
        # Define models to train
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': LinearSVC(random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        # TF-IDF parameters
        max_features = 10000
        ngram_range = (1, 2)
        
        # Log TF-IDF parameters
        mlflow.log_param("tfidf_max_features", max_features)
        mlflow.log_param("tfidf_ngram_range", str(ngram_range))
        
        # Model training and evaluation
        best_model = None
        best_accuracy = 0
        
        for model_name, model in models.items():
            print(f"Training {model_name}...")
            
            # Create a child run for each model
            with mlflow.start_run(run_name=model_name, nested=True):
                # Create pipeline with TF-IDF and model
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)),
                    ('model', model)
                ])
                
                # Train the model
                pipeline.fit(train_data['text'], train_data['label'])
                
                # Evaluate on validation set
                val_predictions = pipeline.predict(val_data['text'])
                val_accuracy = accuracy_score(val_data['label'], val_predictions)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_data['label'], val_predictions, average='binary'
                )
                
                # Log metrics
                mlflow.log_metric("validation_accuracy", val_accuracy)
                mlflow.log_metric("validation_precision", precision)
                mlflow.log_metric("validation_recall", recall)
                mlflow.log_metric("validation_f1", f1)
                
                # Create and log confusion matrix
                cm = confusion_matrix(val_data['label'], val_predictions)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - {model_name}')
                plt.savefig(f"{model_name}_confusion_matrix.png")
                mlflow.log_artifact(f"{model_name}_confusion_matrix.png")
                
                # Save model
                model_path = f"models/{model_name}.joblib"
                joblib.dump(pipeline, model_path)
                mlflow.log_artifact(model_path)
                
                # Track best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model = model_name
                
                print(f"{model_name} trained. Validation accuracy: {val_accuracy:.4f}")
                
        # Log best model info
        mlflow.log_param("best_traditional_model", best_model)
        mlflow.log_metric("best_traditional_model_accuracy", best_accuracy)
        
        print(f"Best traditional model: {best_model} with accuracy: {best_accuracy:.4f}")
        
        # Evaluate best model on test set
        best_pipeline = joblib.load(f"models/{best_model}.joblib")
        test_predictions = best_pipeline.predict(test_data['text'])
        test_accuracy = accuracy_score(test_data['label'], test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_data['label'], test_predictions, average='binary'
        )
        
        # Log test metrics
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        print(f"Test accuracy of best model: {test_accuracy:.4f}")

if __name__ == "__main__":
    # Create directories if they don't exist
    import os
    os.makedirs("models", exist_ok=True)
    
    train_and_evaluate_models()