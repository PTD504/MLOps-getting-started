# preprocess_with_mlflow.py
import mlflow
import mlflow.tensorflow
import os
import shutil

from src.data.preprocessing_data import preprocess_dataset, load_data, save_data_after_preprocessing
from src.data.splitting_data import read_data, save_data


# Set experiment name
mlflow.set_experiment("Sentiment Data Preprocessing")

with mlflow.start_run(run_name="IMDB-preprocessing-run") as run:
    # Log parameters
    mlflow.log_param("remove_duplicates", True)
    mlflow.log_param("text_cleaning", 
                     "lowercase, html tags removal, urls removal, punctuation removal," \
                     "chat_conversion, remove stopwords, emoji removal, expand contractions, lemmatizing and tokenizing")
    mlflow.log_param("tokenizer_oov", "<oov>")
    mlflow.log_param("max_words", 5000)
    mlflow.log_param("max_sequence_length", 318)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("val_size", 0.16)
    mlflow.log_param('train_size', 0.64)

    # Define file paths
    raw_file = "/MLOps-getting-started/data/IMDB-Dataset.csv"
    cleaned_file = "/MLOs-getting-started/data/IMDB-Dataset-Processed.parquet"
    output_dir = '/MLOps-getting-started/data/'
    
    # Step 1: Preprocess raw data
    # Load the dataset
    df = load_data(raw_file)
    # Preprocess the review texts
    df = preprocess_dataset(df)
    # save the preprocessed data
    save_data_after_preprocessing(df, cleaned_file)
    mlflow.log_artifact(cleaned_file, artifact_path="processed_data")

    # Step 2: Split + tokenize + save to .npy
    df = read_data(cleaned_file)
    tokenizer_file = '/MLOps-getting-started/src/data/tokenizer.pkl'
    save_data(df, output_dir, tokenizer_file=tokenizer_file)

    # Step 3: Log .npy files
    for fname in ["X_train.npy", "X_test.npy", "X_val.npy", "y_train.npy", "y_test.npy", "y_val.npy", "tokenizer.pkl"]:
        mlflow.log_artifact(os.path.join(output_dir, fname), artifact_path="npy_data")
