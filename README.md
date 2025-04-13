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

4. **Download the IMDB dataset**:
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

### Run the Web Application

```bash
uvicorn app:app --reload
```
Access the web interface at http://localhost:8000

### API Documentation

FastAPI generates automatic documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Docker Deployment

1. **Build the image**:
```bash
docker build -t sentiment-analysis .
```

2. **Run the container**:
```bash
docker run -p 8000:8000 sentiment-analysis
```


## Key Advantages

1. **Experiment Tracking**: All training runs are logged with MLflow, allowing easy comparison of model performance
2. **Hyperparameter Optimization**: Optuna systematically explores the parameter space to find optimal configurations
3. **Distributed Execution**: Ray enables parallel processing of pipeline components
4. **Interactive UI**: Easy-to-use web interface for testing models
5. **API Service**: FastAPI provides performant endpoints with auto-generated documentation

## Future Work

- Add transformer-based models (BERT, DistilBERT)
- Implement continuous training with data versioning
- Add more advanced explainability with LIME or SHAP
- Implement model monitoring for drift detection

## License

MIT License