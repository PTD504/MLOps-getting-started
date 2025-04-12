import pandas as pd
import optuna
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
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

# Preprocessing for LSTM: Convert text to sequences and pad them
def preprocess_for_lstm(X_train, X_test, tokenizer, max_len=100):
    # Tokenize the text data using the tokenizer (for LSTM)
    X_train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=max_len)
    X_test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=max_len)
    
    # Return tokenized sequences
    return X_train_encodings, X_test_encodings

# Build LSTM model
def build_lstm_model(input_dim, max_len=100):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=max_len))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Objective function for Optuna (fine-tuning LSTM)
def objective_lstm(trial, X_train, y_train, X_test, y_test, tokenizer, max_len=100):
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_categorical('epochs', [3, 5, 10])
    
    # Tokenize the input data for LSTM
    X_train_encodings, X_test_encodings = preprocess_for_lstm(X_train, X_test, tokenizer, max_len)
    
    # Convert tokenized sequences to tensor datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train_encodings), y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test_encodings), y_test))
    
    # Build LSTM model
    model = build_lstm_model(input_dim=5000, max_len=max_len)
    model.fit(train_dataset.batch(batch_size), epochs=epochs, batch_size=batch_size, validation_data=test_dataset)
    
    # Evaluate the model
    y_pred = model.predict(X_test_encodings['input_ids'])
    accuracy = accuracy_score(y_test, np.round(y_pred))
    
    # Log parameters and accuracy using MLflow
    mlflow.log_params({
        'batch_size': batch_size,
        'epochs': epochs
    })
    mlflow.log_metric('accuracy', accuracy)
    
    return accuracy

# Fine-tuning BERT and DistilBERT models
def objective_bert_distilbert(trial, X_train, y_train, X_test, y_test, model_name='bert-base-uncased'):
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-4)
    
    # Load BERT or DistilBERT model and tokenizer
    if model_name == 'bert-base-uncased':
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_name == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize and pad sequences
    X_train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
    X_test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

    # Convert to TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((dict(X_train_encodings), y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((dict(X_test_encodings), y_test))
    
    # Training with specified batch_size and learning_rate
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(train_dataset.batch(batch_size), epochs=3, batch_size=batch_size)
    
    # Evaluate the model
    y_pred = model.predict(X_test_encodings)
    accuracy = accuracy_score(y_test, np.round(y_pred))
    
    # Log parameters and accuracy using MLflow
    mlflow.log_params({
        'batch_size': batch_size,
        'learning_rate': learning_rate
    })
    mlflow.log_metric('accuracy', accuracy)
    
    return accuracy

# Main function to run the Optuna optimization and MLflow tracking
def main():
    # Load data
    input_file = 'data/IMDB-Dataset_preprocessed.csv'
    df = load_data(input_file)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Start MLflow tracking
    mlflow.start_run()
    
    # Create an Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_lstm(trial, X_train, y_train, X_test, y_test, tokenizer, max_len=100), n_trials=50)
    
    # Log the best parameters and model
    print("Best hyperparameters:", study.best_params)
    
    # Final training with best parameters for LSTM
    best_params = study.best_params
    model = build_lstm_model(input_dim=5000)
    model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_test, y_test))
    
    # Save the best model to MLflow
    mlflow.keras.log_model(model, 'LSTM_model')
    
    # Evaluate the final model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, np.round(y_pred))
    print(f"Final Accuracy: {accuracy}")
    
    # End the MLflow run
    mlflow.end_run()

if __name__ == '__main__':
    main()
