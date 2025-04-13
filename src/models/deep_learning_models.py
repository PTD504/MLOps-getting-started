import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

# Set memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

def train_lstm_model():
    # Start MLflow run
    with mlflow.start_run(run_name="deep_learning_models"):
        # Load processed data
        print("Loading processed data...")
        train_data = pd.read_csv("data/train.csv")
        val_data = pd.read_csv("data/val.csv")
        test_data = pd.read_csv("data/test.csv")
        
        # Hyperparameters
        max_words = 10000
        max_len = 200
        embedding_dim = 128
        lstm_units = 64
        batch_size = 64
        epochs = 10
        
        # Log hyperparameters
        mlflow.log_param("max_words", max_words)
        mlflow.log_param("max_len", max_len)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_param("lstm_units", lstm_units)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        
        # Tokenize text data
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(train_data['text'])
        
        # Convert text to sequences
        X_train_seq = tokenizer.texts_to_sequences(train_data['text'])
        X_val_seq = tokenizer.texts_to_sequences(val_data['text'])
        X_test_seq = tokenizer.texts_to_sequences(test_data['text'])
        
        # Pad sequences
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
        X_val_pad = pad_sequences(X_val_seq, maxlen=max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
        
        # Get labels
        y_train = train_data['label'].values
        y_val = val_data['label'].values
        y_test = test_data['label'].values
        
        # Build LSTM model
        print("Building LSTM model...")
        model = Sequential([
            Embedding(max_words, embedding_dim, input_length=max_len),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Bidirectional(LSTM(lstm_units)),
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
        
        # Model summary
        model.summary()
        
        # Create checkpoint directory
        checkpoint_dir = "models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, "lstm_model_best.h5"),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Train model
        print("Training LSTM model...")
        history = model.fit(
            X_train_pad, y_train,
            validation_data=(X_val_pad, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        plt.tight_layout()
        
        # Save plot
        history_plot_path = "models/lstm_training_history.png"
        plt.savefig(history_plot_path)
        mlflow.log_artifact(history_plot_path)
        
        # Log training metrics
        for epoch, acc in enumerate(history.history['accuracy']):
            mlflow.log_metric("train_accuracy", acc, step=epoch)
        
        for epoch, loss in enumerate(history.history['loss']):
            mlflow.log_metric("train_loss", loss, step=epoch)
        
        for epoch, val_acc in enumerate(history.history['val_accuracy']):
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        for epoch, val_loss in enumerate(history.history['val_loss']):
            mlflow.log_metric("val_loss", val_loss, step=epoch)
        
        # Evaluate on test set
        print("Evaluating model on test data...")
        test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_loss", test_loss)
        
        # Detailed metrics on test set
        y_pred_prob = model.predict(X_test_pad)
        y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )
        
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('LSTM Model Confusion Matrix')
        
        cm_plot_path = "models/lstm_confusion_matrix.png"
        plt.savefig(cm_plot_path)
        mlflow.log_artifact(cm_plot_path)
        
        # Save model and tokenizer
        model_path = "models/lstm_model.h5"
        model.save(model_path)
        mlflow.keras.log_model(model, "lstm_model")
        
        # Save tokenizer
        import pickle
        tokenizer_path = "models/tokenizer.pkl"
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        mlflow.log_artifact(tokenizer_path)
        
        print(f"LSTM model training completed. Test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    train_lstm_model()