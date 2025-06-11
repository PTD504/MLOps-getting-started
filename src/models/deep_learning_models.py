import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import os

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected. Training will use GPU.")
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")
else:
    print("No GPU detected. Training will use CPU.")

def train_lstm_model():
    """Train mô hình LSTM cho phân loại văn bản"""
    with mlflow.start_run(run_name="deep_learning_models_training"):
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
        
        # Hyperparameters
        max_words = 10000
        max_len = 200
        embedding_dim = 128
        lstm_units = 64
        
        mlflow.log_params({
            "max_words": max_words,
            "max_len": max_len,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units
        })
        
        # Tokenize text
        print("Đang tokenize văn bản...")
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(train_data['text'])
        
        X_train = tokenizer.texts_to_sequences(train_data['text'])
        X_val = tokenizer.texts_to_sequences(val_data['text'])
        X_test = tokenizer.texts_to_sequences(test_data['text'])
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train, maxlen=max_len)
        X_val_pad = pad_sequences(X_val, maxlen=max_len)
        X_test_pad = pad_sequences(X_test, maxlen=max_len)
        
        # Build LSTM model
        print("Đang xây dựng mô hình LSTM...")
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Bidirectional(LSTM(lstm_units // 2)),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Đang huấn luyện mô hình LSTM...")
        history = model.fit(
            X_train_pad, train_data['label'],
            epochs=5,
            batch_size=64,
            validation_data=(X_val_pad, val_data['label']),
            verbose=1
        )
        
        # Log training metrics
        for i in range(len(history.history['loss'])):
            mlflow.log_metrics({
                "train_loss": history.history['loss'][i],
                "train_accuracy": history.history['accuracy'][i],
                "val_loss": history.history['val_loss'][i],
                "val_accuracy": history.history['val_accuracy'][i]
            }, step=i)
        
        # Evaluate model
        print("Đang đánh giá mô hình...")
        test_loss, test_accuracy = model.evaluate(X_test_pad, test_data['label'])
        
        # Log test metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
        # Generate predictions and calculate additional metrics
        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_data['label'], y_pred, average='binary'
        )
        
        # Log additional metrics
        mlflow.log_metrics({
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })
        
        # Generate confusion matrix
        cm = confusion_matrix(test_data['label'], y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - LSTM')
        
        # Save the confusion matrix
        os.makedirs("models", exist_ok=True)
        cm_path = "models/lstm_confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Save model and tokenizer
        model_path = "models/lstm_model.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)
        
        # Log model with MLflow
        mlflow.tensorflow.log_model(model, "lstm_model")
        
        # Save tokenizer
        import pickle
        tokenizer_path = "models/tokenizer.pkl"
        with open(tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        mlflow.log_artifact(tokenizer_path)
        
        print(f"Mô hình LSTM đã được huấn luyện thành công!")
        print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    
    train_lstm_model()