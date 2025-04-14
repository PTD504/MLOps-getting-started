import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Split the dataset into train and test
def split_data(df, text_column='cleaned_review', label_column='sentiment'):
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Objective function for Optuna
def objective(trial, X_train, y_train, X_test, y_test):
    # Hyperparameter search space
    model_type = trial.suggest_categorical('model_type', ['LogisticRegression', 'SVM', 'NaiveBayes'])
    
    if model_type == 'LogisticRegression':
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        penalty = trial.suggest_categorical('penalty', ['l2'])
        model = LogisticRegression(C=C, penalty=penalty, max_iter=1000)
    elif model_type == 'SVM':
        C = trial.suggest_loguniform('C', 1e-5, 1e5)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        model = SVC(C=C, kernel=kernel)
    elif model_type == 'NaiveBayes':
        alpha = trial.suggest_loguniform('alpha', 1e-5, 1e5)
        model = MultinomialNB(alpha=alpha)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log the metrics and parameters using MLflow
    mlflow.log_params({
        'model_type': model_type,
        'C': C if model_type in ['LogisticRegression', 'SVM'] else None,
        'penalty': penalty if model_type == 'LogisticRegression' else None,
        'kernel': kernel if model_type == 'SVM' else None,
        'alpha': alpha if model_type == 'NaiveBayes' else None
    })
    
    mlflow.log_metric('accuracy', accuracy)
    
    return accuracy

# Main function to run the Optuna optimization and MLflow tracking
def main():
    # Load data
    input_file = '/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/data/IMDB-Dataset_preprocessed.csv'
    df = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Start MLflow tracking
    mlflow.start_run()
    
    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=100)
    
    # Log the best parameters and model
    print("Best hyperparameters:", study.best_params)
    
    # Final training with best parameters
    best_params = study.best_params
    model_type = best_params['model_type']
    
    if model_type == 'LogisticRegression':
        model = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], max_iter=1000)
    elif model_type == 'SVM':
        model = SVC(C=best_params['C'], kernel=best_params['kernel'])
    elif model_type == 'NaiveBayes':
        model = MultinomialNB(alpha=best_params['alpha'])
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Final training with the best parameters
    model.fit(X_train_tfidf, y_train)
    
    # Save the best model and vectorizer to MLflow
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact('tfidf_vectorizer.pkl')  # Save vectorizer as artifact

    # Evaluate final model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {accuracy}")
    
    # End the MLflow run
    mlflow.end_run()

if __name__ == '__main__':
    # main()
    import pickle
    tokenizer = pickle.load(open('/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/src/data/tokenizer.pkl', 'rb'))
    print(len(tokenizer.word_index))
    X_train = np.load('/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/data/X_train.npy', allow_pickle=True)
    X_test = np.load('/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/data/X_test.npy', allow_pickle=True)
    y_train = np.load('/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/data/y_train.npy', allow_pickle=True)
    y_test = np.load('/home/phucuy2025/School_Stuff/CS317_MLOps/MLOps-getting-started/data/y_test.npy', allow_pickle=True)

    print(f"X train shape = {X_train.shape}")
    print(f"y train shape = {y_train.shape}")
    print(f"X test shape = {X_test.shape}")
    print(f"y test shape = {y_test.shape}")