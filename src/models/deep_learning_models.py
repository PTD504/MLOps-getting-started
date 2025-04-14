import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from transformers import BertTokenizer, TFBertForSequenceClassification, DistilBertTokenizer, TFDistilBertForSequenceClassification
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle, os

# Library for model building
from tensorflow.keras.regularizers import l2
import tensorflow
import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
#from attention import BahdanauAttention
from keras.layers import SimpleRNN,LSTM,GRU, Embedding, Dense, SpatialDropout1D, Dropout, BatchNormalization, Bidirectional, Attention
from sklearn.metrics import accuracy_score
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
import keras_tuner as kt
#Library to overcome Warnings
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
def load_data(data_dir, tokenizer_path):
    X_train = np.load(data_dir + 'X_train.npy', allow_pickle=True)
    X_test = np.load(data_dir + 'X_test.npy', allow_pickle=True)
    X_val = np.load(data_dir + 'X_val.npy', allow_pickle=True)

    y_train = np.load(data_dir + 'y_train.npy', allow_pickle=True)
    y_test = np.load(data_dir + 'y_test.npy', allow_pickle=True)
    y_val = np.load(data_dir + 'y_val.npy', allow_pickle=True)

    # convert label positive and negative to numeric label 0 and 1

    y_train = np.array([1 if label == "positive" else 0 for label in y_train])
    y_val = np.array([1 if label == "positive" else 0 for label in y_val])
    y_test = np.array([1 if label == "positive" else 0 for label in y_test])

    print(f"[INFO]: X train shape = {X_train.shape}")
    print(f"[INFO]: y train shape = {y_train.shape}")
    print(f"[INFO]: X test shape = {X_test.shape}")
    print(f"[INFO]: y test shape = {y_test.shape}")
    print(f"[INFO]: X val shape = {X_val.shape}")
    print(f"[INFO]: y val shape = {y_val.shape}")

    # load the tokenizer
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    print(f"[INFO]: Tokenizer loaded successfully!")
    return X_train, y_train, X_val,y_val, X_test, y_test, tokenizer

# Bidirectional RNN
def build_and_train_bi_rnn_model(X_train, y_train, X_val, y_val, input_dim, max_len=318):
    RNN_model = Sequential()
    RNN_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=100))
    RNN_model.add(SpatialDropout1D(0.2))
    RNN_model.add(Bidirectional(SimpleRNN(64, return_sequences=True)))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(BatchNormalization())
    RNN_model.add(Bidirectional(SimpleRNN(32, return_sequences=True)))
    RNN_model.add(Dropout(0.2))
    RNN_model.add(BatchNormalization())
    RNN_model.add(SimpleRNN(16, return_sequences=False))

    # Continue with other layers
    RNN_model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    RNN_model.add(Dense(1, activation='sigmoid'))

    # Summarize the model
    # RNN_model.summary()

    # compile the model
    RNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    epochs = 10
    batch_size = 32
    history = RNN_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    # log the model and params
    with mlflow.start_run(run_name="BiRNN_model_run", nested=True):
        # log the model
        mlflow.keras.log_model(RNN_model, 'RNN_model')
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'metrics': 'accuracy'
        })

    return RNN_model, history

# Build LSTM model
def build_and_train_lstm_model(X_train, y_train, X_val, y_val, input_dim, max_len=318):
    lstm_model = Sequential()
    lstm_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    lstm_model.add(SpatialDropout1D(0.5))
    lstm_model.add(LSTM(5, return_sequences=False))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(BatchNormalization())


    lstm_model.add(Dense(1, activation='sigmoid'))

    # Summarize the model
    # lstm_model.summary()

    # compile the model
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    epochs = 10
    batch_size = 128
    history = lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    # log the model and params
    with mlflow.start_run(run_name='LSTM_model_run', nested=True):
        # log the model
        mlflow.keras.log_model(lstm_model, 'LSTM_model')
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy']
        })
    
    return lstm_model, history

# Build GRU
def build_and_train_gru_model(X_train, y_train, X_val, y_val, input_dim, max_len=318):
    # Define the model
    GRU_model = Sequential()
    GRU_model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=max_len))
    GRU_model.add(SpatialDropout1D(0.5))
    GRU_model.add(GRU(5, return_sequences=False))
    GRU_model.add(Dropout(0.5))
    #GRU_model.add(BatchNormalization())


    GRU_model.add(Dense(1, activation='sigmoid'))

    # Summarize the model
    # GRU_model.summary()

    # compile the model
    GRU_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    epochs = 5
    batch_size = 256
    history = GRU_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    # log the model and params
    with mlflow.start_run(run_name='GRU_model_run',nested=True):
        # log the model
        mlflow.keras.log_model(GRU_model, 'GRU_model')
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'loss_function': 'binary_crossentropy',
            'optimizer': 'adam',
            'metrics': ['accuracy']
        })
    
    return GRU_model, history

def build_gru_model_fn(input_dim, input_length):
    def model_builder(hp):
        model = Sequential()
        model.add(Embedding(input_dim=input_dim,
                            output_dim=100,
                            input_length=input_length))

        units = hp.Int('units', min_value=32, max_value=128, step=32)
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(rate=hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(GRU(units, return_sequences=True))
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.1)))

        model.add(GRU(units))
        model.add(Dropout(rate=hp.Float('final_dropout_rate', 0.1, 0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=keras.optimizers.Adam(
                        learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    return model_builder


def tune_and_log_gru(x_train, y_train, x_val, y_val, input_dim, input_length, project_name="gru_sentiment_tuning"):
    tuner = RandomSearch(
        build_gru_model_fn(input_dim, input_length),
        objective="val_accuracy",
        max_trials=5,
        executions_per_trial=1,
        directory="tuner_logs",
        project_name=project_name,
        overwrite=True
    )

    tuner.search(x_train, y_train,
                 validation_data=(x_val, y_val),
                 epochs=10,
                 batch_size=32)

    best_model = tuner.get_best_models(num_models=1)[0] # list of models
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] # list of hyperparameters

    mlflow.set_experiment("GRU Hypertuning")

    with mlflow.start_run(run_name='Hypertuned_GRU__run',nested=True):
        mlflow.log_params(best_hps.values)
        history = best_model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=10,
            batch_size=32
        )

        val_acc = history.history["val_accuracy"][-1]
        mlflow.log_metric("val_accuracy", val_acc)

        mlflow.tensorflow.log_model(best_model, artifact_path="hyper_gru_model")
        best_model.save("best_gru_model.h5")
        
        import json
        with open("best_hyperparams.json", "w") as f:
            json.dump(best_hps.values, f)

    return best_model, best_hps

def predict(model_name, model, X_test, y_test):
    model_pred = model.predict(X_test)
    model_pred = (model_pred > 0.5).astype('int')
    model_accuracy = accuracy_score(model_pred, y_test) * 100
    print(f"{model_name}_Accuracy Score is: {model_accuracy}%")
    sentiment_labels = {0: 'negative', 1: 'positive'}

    report_dict = classification_report(y_test, model_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Save the classification report to a file
    report_path = f'classification_report_for_{model_name}.csv'
    report_df.to_csv(report_path, index=False)

    # Use MLflow to log the report
    with mlflow.start_run(f'Metrics_of_{model_name}_run',nested=True):
        # Log the model
        mlflow.keras.log_model(model, 'model')
        # Log the metrics
        mlflow.log_metrics({
            'accuracy': report_dict['accuracy'],
            'recall_class_0': report_dict['0']['recall'],
            'recall_class_1': report_dict['1']['recall'],
            'f1_score_macro': report_dict['macro avg']['f1-score']
        })
        # Log the classification report as an artifact
        mlflow.log_artifact(report_path, artifact_path='classification_report')

    # Generate sentiment predictions
    model_sentiments = [[sentiment_labels[val[0]]] for val in model_pred]

    # Confusion matrix
    cm = confusion_matrix(y_test, model_pred)
    cm_df = pl.DataFrame(cm)

    # Save the confusion matrix to a file
    cm_path = f'confusion_matrix_of_{model_name}.csv'
    cm_df.write_csv(cm_path)

    # Log the confusion matrix as an artifact
    mlflow.log_artifact(cm_path, artifact_path='confusion_matrix')

    return model_sentiments
def plot_model(model_name, history, directory):
    """
    Plots training and validation accuracy and loss for a model.

    Args:
        model_name (str): Name of the model (used for saving the plot).
        history (History): Keras History object containing training metrics.
        directory (str): Directory to save the plot.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Validation'], loc='upper left')

    # Save the plot
    plot_path = f'{directory}/{model_name}_accuracy_loss.png'
    plt.savefig(plot_path)
    print(f"[INFO]: Plot saved to {plot_path}")

    # Close the plot to free memory
    plt.close(fig)
# Main function to run the Optuna optimization and MLflow tracking
def main():
    # Load data
    data_dir = '/MLOps-getting-started/data/'
    tokenizer_path = '/MLOps-getting-started/src/data/tokenizer.pkl'
    X_train, y_train, X_val,y_val, X_test, y_test, tokenizer = load_data(data_dir, tokenizer_path)
    project_dir = '/MLOps-getting-started/'
    
    # Start MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("Sentiment Analysis")
    # RNN_model, RNN_history = build_and_train_bi_rnn_model(
    #     X_train, y_train, X_val, y_val, input_dim=len(tokenizer.word_index)+1
    # )
    LSTM_model, LSTM_history = build_and_train_lstm_model(
        X_train, y_train, X_val, y_val, input_dim=len(tokenizer.word_index)+1
    )
    plot_model('LSTM', LSTM_history, project_dir)
    # GRU_model, GRU_history = build_and_train_gru_model(
    #     X_train, y_train, X_val, y_val, input_dim=len(tokenizer.word_index)+1
    # )
    # Hyper_GRU_model, Hyper_GRU_history = build_and_train_Hyper_GRU_model(
    #      X_train, y_train, X_val, y_val, input_dim=len(tokenizer.word_index)+1
    # )
    models = {
        # 'BiRNN': RNN_model,
        'LSTM': LSTM_model,
        # 'GRU': GRU_model,
        # 'Hypertuned_GRU': Hyper_GRU_model
    }
    histories = {
        # 'BiRNN': RNN_history,
        'LSTM': LSTM_history,
        # 'GRU': GRU_history,
        # 'Hypertuned_GRU': Hyper_GRU_history
    }

    # Evaluate and predict
    for model_name, model in models.items():
        print(f"[INFO]: Evaluating {model_name} model ...")
        model_sentiments = predict(model_name, model, X_test, y_test)
        print(f"[INFO]: {model_name} predictions: {model_sentiments[0: 10]}") 
    # End the MLflow run

    # hyperparameter tuning
    
    best_gru_model, best_gru_hps = tune_and_log_gru(
        X_train, y_train, X_val, y_val, input_dim=len(tokenizer.word_index)+1, input_length=318)

    

if __name__ == '__main__':
    main()
