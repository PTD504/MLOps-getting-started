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
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/                # Put your lstm_model.h5 and tokenizer.pkl here
│   ├── lstm_model.h5
│   └── tokenizer.pkl
├── templates/
│   └── index.html         # (created by app.py or you can make one)
├── static/                # (created by app.py or you can make one)
├── prometheus/
│   ├── prometheus.yml
│   ├── alertmanager.yml
│   └── alert.rules.yml
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasources.yml
│       └── dashboards/    # (Optional: for pre-configured dashboards)
│           └── dashboard.yml
├── loki/
│   └── loki-config.yml
└── promtail/
    └── promtail-config.yml
```

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/MLOps-getting-started.git
cd MLOps-getting-started
```

2. **Run the docker-compose file**:

```bash
docker-ccompose up --build -d
```

If needed, you need to delete all the containers before if they are using the same ports as this one of ours:

```bash
docker-compose down -v
```

3. **Check the list of running container**:

```bash
docker ps
```

_Note: This project is running with Python 3.9._

# Video demo
