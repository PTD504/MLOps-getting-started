## Overview

This project implements a full MLOps workflow for sentiment analysis, comparing traditional machine learning approaches (Logistic Regression, SVM, Naive Bayes) with deep learning models (LSTM). It includes data preprocessing, model training, hyperparameter optimization, distributed execution, experiment tracking, and model serving via a web API.

## Features

- **Data Preprocessing**: Text cleaning, lemmatization, and vectorization with TF-IDF
- **Multiple Models**: 
  - Traditional ML: Logistic Regression, SVM, Naive Bayes
  - Deep Learning: Bidirectional LSTM with Embedding
- **MLOps Components**:
  - **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
  - **Hyperparameter Tuning**: Optuna for systematic optimization
  - **Distributed Training**: Ray for parallel execution of pipeline components
  - **Model Serving**: FastAPI for REST API endpoints with interactive documentation
- **Web Application**: Interactive UI for testing sentiment analysis models
- **Explainability**: Basic feature importance visualization

## Dataset

The project uses the IMDB Movie Reviews dataset containing 50,000 movie reviews labeled as positive or negative sentiment.

## Project Structure

```
MLOps-getting-started/
├── app.py                  # FastAPI application
├── pipeline.py             # Main orchestration pipeline using Ray
├── requirements.txt        # Project dependencies
├── Dockerfile              # Container definition
├── README.md               # Project documentation
├── data/                   # Dataset storage
│   ├── IMDB-Dataset.csv    # Original dataset
│   ├── train.csv           # Training split
│   ├── val.csv             # Validation split
│   └── test.csv            # Test split
├── src/                    # Source code
│   ├── data/
│   │   └── preprocessing.py # Data preprocessing module
│   └── models/
│       ├── traditional_models.py # Traditional ML models
│       └── deep_learning_models.py # LSTM and other DL models
├── models/                 # Saved models and artifacts
├── static/                 # Static files for web app
├── templates/              # HTML templates
└── mlruns/                 # MLflow experiment tracking data
```

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/MLOps-getting-started.git
cd MLOps-getting-started
```

2. **Set up a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```
**Note: This project is not compatible with Python 3.12 or higher due to certain dependencies not supporting those versions.**

1. **Download the IMDB dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Save as IMDB-Dataset.csv

## Usage

### Run the Full Pipeline

```bash
python pipeline.py
```

This will:
1. Preprocess the dataset
2. Train traditional ML models with Optuna hyperparameter tuning
3. Train deep learning models
4. Log all experiments to MLflow

### View Experiment Tracking Results

```bash
mlflow ui
```
Access the MLflow dashboard at http://localhost:5000

## Training Process Video
- [training_process](https://drive.google.com/file/d/1rPvdYF71s9emmPndpeG6CEJAPC7hnraU/view?usp=sharing)

## License

MIT License
