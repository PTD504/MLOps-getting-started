# MLOps Getting Started - Complete Workflow

A comprehensive MLOps project implementing sentiment analysis with traditional machine learning and deep learning models, featuring complete containerization, deployment, and monitoring capabilities.

## 🎯 Overview

This project implements a full MLOps workflow for sentiment analysis, comparing traditional machine learning approaches (Logistic Regression, SVM, Naive Bayes) with deep learning models (LSTM). It includes data preprocessing, model training, hyperparameter optimization, distributed execution, experiment tracking, and model serving via a web API with comprehensive monitoring and observability.

## ✨ Features

### Core ML Capabilities
- **Data Preprocessing**: Text cleaning, lemmatization, and vectorization with TF-IDF
- **Multiple Models**: 
  - Traditional ML: Logistic Regression, SVM, Naive Bayes
  - Deep Learning: Bidirectional LSTM with Embedding
- **Explainability**: Basic feature importance visualization

### MLOps Components
- **Experiment Tracking**: MLflow for logging parameters, metrics, and artifacts
- **Hyperparameter Tuning**: Optuna for systematic optimization
- **Distributed Training**: Ray for parallel execution of pipeline components
- **Model Serving**: FastAPI for REST API endpoints with interactive documentation
- **Web Application**: Interactive UI for testing sentiment analysis models

### Production Features (Lab 3)
- **System Monitoring**: CPU, Memory, Disk usage tracking
- **API Monitoring**: Request rate, Error rate, Latency metrics
- **Model Monitoring**: Inference speed, Confidence scores
- **Structured Logging**: JSON format logging with multiple outputs
- **Alerting**: Alertmanager for anomaly detection and notifications
- **Containerization**: Full Docker deployment with monitoring stack

## 📊 Dataset

The project uses the IMDB Movie Reviews dataset containing 50,000 movie reviews labeled as positive or negative sentiment.

## 📁 Project Structure

```
MLOps-getting-started/
├── app.py                          # FastAPI application with monitoring endpoints
├── pipeline.py                     # Main orchestration pipeline using Ray
├── test_traffic.py                 # Traffic generation script for testing
├── requirements.txt                # Project dependencies (all labs)
├── Dockerfile                      # Multi-stage container definition
├── docker-compose.yml              # Basic API deployment
├── docker-compose-hub.yml          # Deploy from Docker Hub image
├── README.md                       # Project documentation
├── data/                           # Dataset storage
│   ├── IMDB-Dataset.csv           # Original IMDB dataset
│   ├── train.csv                  # Training split
│   ├── val.csv                    # Validation split
│   └── test.csv                   # Test split
├── src/                           # Source code modules
│   ├── data/
│   │   └── preprocessing.py       # Data preprocessing module
│   └── models/
│       ├── traditional_models.py  # Traditional ML models
│       └── deep_learning_models.py # LSTM and other DL models
├── models/                        # Saved models and artifacts
├── static/                        # Static files for web UI (CSS, JS, images)
├── templates/                     # HTML templates for web interface
├── logs/                          # Application logs directory
├── monitoring/                    # Monitoring stack configuration
│   ├── docker-compose-monitoring.yml # Complete monitoring stack
│   ├── prometheus/
│   │   └── prometheus.yml         # Prometheus configuration
│   ├── grafana/
│   │   ├── provisioning/          # Grafana auto-provisioning
│   │   └── dashboards/            # Pre-built dashboards
│   ├── alertmanager/
│   │   └── alertmanager.yml       # Alerting rules and receivers
│   └── fluentd/
│       └── fluent.conf            # Log aggregation configuration
└── mlruns/                        # MLflow experiment tracking data
```

## 🚀 Installation

### Prerequisites
- **Python**: 3.9+ (not compatible with Python 3.12 or higher due to certain dependencies)
- **Docker**: version 20.10.0 or later
- **Docker Compose**: version 2.0.0 or later
- **Memory**: 4GB+ RAM recommended for full monitoring stack

### Setup Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/MLOps-getting-started.git
cd MLOps-getting-started
```

2. **Set up a virtual environment**:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
# For all labs (recommended)
pip install -r requirements.txt

# Or install by specific lab
pip install -r requirements_lab1.txt  # Lab 1: ML Development
pip install -r requirements_lab2.txt  # Lab 2: Containerization
pip install -r requirements_lab3.txt  # Lab 3: Monitoring
```

4. **Download the IMDB dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - Save as `data/IMDB-Dataset.csv`

## 📈 Lab 1: ML Development & Experiment Tracking

### Run the Full Pipeline

```bash
python pipeline.py
```

This will execute:
1. **Data Preprocessing**: Text cleaning and feature extraction
2. **Traditional ML Training**: Multiple models with Optuna hyperparameter tuning
3. **Deep Learning Training**: LSTM model with optimization
4. **Experiment Logging**: All parameters, metrics, and artifacts logged to MLflow

### View Experiment Tracking Results

```bash
mlflow ui
```
Access the MLflow dashboard at http://localhost:5000

### Training Process Demo
- [Training Process Video](https://drive.google.com/file/d/1rPvdYF71s9emmPndpeG6CEJAPC7hnraU/view?usp=sharing)

## 🐳 Lab 2: Containerization & Deployment

### Local Development

Run API locally for development:
```bash
uvicorn app:app --host localhost --port 8000

# or 

python app.py
```

### Docker Deployment

#### Prerequisites for Docker Deployment
- Make sure you have trained models in the `models/` directory:
  - `logistic_regression.joblib`
  - `lstm_model.h5`
  - `tokenizer.pkl`
- If you don't have models, run the pipeline first: `python pipeline.py`
- Static files in `static/` directory (included in repository)
- HTML templates in `templates/` directory (included in repository)

#### Deploy with Docker Compose

1. **Build and start the container**:
```bash
docker-compose up -d
```

2. **Access the services**:
   - **Web interface**: http://localhost:8000
   - **API documentation**: http://localhost:8000/docs
   - **Health check**: http://localhost:8000/info

#### Docker Hub Deployment

1. **Build and publish to Docker Hub**:
```bash
# Log in to Docker Hub
docker login

# Tag and push the image
docker tag sentiment-analysis-api:latest <your-username>/sentiment-analysis-api:latest
docker push <your-username>/sentiment-analysis-api:latest

# Or use the automated script
./push-to-dockerhub.sh your-username
```

2. **Run from Docker Hub**:
```bash
docker-compose -f docker-compose-hub.yml up -d
```

#### Server Deployment

Deploy to production server:
```bash
# Give execute permission to the deployment script
chmod +x deploy.sh

# Run deployment script with parameters
./deploy.sh username server-ip server-path dockerhub-username
```

### Lab 2 Demo Videos
- **Docker Build & Deployment Tests**: [Video Collection](https://drive.google.com/drive/folders/1QZql71yOEhx4iyF9JAs8mpA3C-xdXVwe?usp=sharing)
  - `access_the_api_after_build`
  - `BuildAndStartContainer`
  - `publish_to_docker_hub`
  - `run_the_service_from_image_n_docker_hub`
- **Server Deployment Demo**: [Production Deploy Video](https://drive.google.com/file/d/1vAXwRElNjsoeqkng31pU12-9BI4JpP9t/view?usp=drive_link)

## 📊 Lab 3: Monitoring & Observability

### Overview
Extended monitoring and logging capabilities using Prometheus, Grafana, and Alertmanager for comprehensive observability.

### Monitoring Features
- **System Monitoring**: CPU, Memory, Disk, Network I/O (via node-exporter)
- **API Monitoring**: Request rate, Error rate, Response latency
- **Model Monitoring**: Inference speed, Confidence score distribution
- **Structured Logging**: JSON format logs with multiple outputs
- **Alerting**: Automated anomaly detection and notifications

### Installation and Setup

1. **Start the complete monitoring stack**:
```bash
cd monitoring
docker-compose -f docker-compose-monitoring.yml up -d
```

2. **Verify all services are running**:
```bash
# Check container status
docker-compose -f docker-compose-monitoring.yml ps

# Test API endpoints
curl http://localhost:8000/info
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Test monitoring services
curl http://localhost:9090/targets  # Prometheus targets
```

### Service Access URLs

| Service | URL | Credentials | Description |
|---------|-----|-------------|-------------|
| **API** | http://localhost:8000 | - | Main application interface |
| **API Docs** | http://localhost:8000/docs | - | Interactive API documentation |
| **Prometheus** | http://localhost:9090 | - | Metrics collection and querying |
| **Grafana** | http://localhost:3000 | admin/admin | Dashboards and visualization |
| **Alertmanager** | http://localhost:9093 | - | Alert management and routing |
| **Node Exporter** | http://localhost:9100 | - | System metrics collection |

### Testing and Traffic Generation

1. **Generate test traffic for monitoring**:
```bash
python test_traffic.py
```

This script generates:
- **Phase 1**: Normal traffic (120 seconds at 3 RPS)
- **Phase 2**: Error generation (30 seconds) 
- **Phase 3**: High traffic load (60 seconds at 5 RPS)

2. **Monitor in Grafana**:
   - Login to http://localhost:3000 with admin/admin
   - Import pre-built dashboards or create custom ones
   - Observe real-time metrics during traffic generation

3. **Test alerting system**:
   - Script automatically generates errors to trigger alerts
   - Check Alertmanager UI at http://localhost:9093 for active alerts
   - Verify alert notifications (if configured)

### Monitoring Metrics

#### System Metrics (Node Exporter)
- **CPU**: Usage percentage, load average
- **Memory**: Total, used, available, swap usage
- **Disk**: Usage percentage, I/O operations, read/write rates
- **Network**: Traffic in/out, packet rates, errors

#### API Metrics (Prometheus + FastAPI Instrumentator)
- **Request Rate**: Requests per second by endpoint and method
- **Error Rate**: HTTP error percentage by status code
- **Response Latency**: P50, P95, P99 percentiles
- **Active Connections**: Concurrent request count
- **Request Duration**: Histogram of response times

#### Model-Specific Metrics
- **Prediction Latency**: Inference time by model type (traditional vs LSTM)
- **Confidence Scores**: Distribution of prediction confidence
- **Model Usage**: Request count per model type
- **Prediction Results**: Sentiment distribution (positive/negative)

### Logging Architecture

**Multi-layer logging system**:
- **Console Output**: Development debugging and real-time monitoring
- **File Logging**: Persistent storage in `logs/` directory with rotation
- **Fluentd Integration**: Centralized log aggregation and forwarding
- **Structured Format**: JSON logs for machine processing
- **Log Levels**: Configurable verbosity (DEBUG, INFO, WARNING, ERROR)

### Alerting Rules

Alerts are automatically triggered when:
- **High Error Rate**: >50% errors sustained for 5 minutes
- **Slow Response Time**: API response time >5 seconds consistently
- **Low Model Confidence**: Confidence scores <0.6 for multiple predictions
- **Resource Exhaustion**: System resource usage >80% (CPU/Memory)
- **Service Health**: Container restarts or failures
- **High Request Volume**: Unusual traffic spikes

### Troubleshooting

#### Common Issues and Solutions

1. **Check container logs**:
```bash
# View logs for specific service
docker-compose -f monitoring/docker-compose-monitoring.yml logs <service-name>
# Available services: sentiment-api, prometheus, grafana, alertmanager, node-exporter

# Follow logs in real-time
docker-compose -f monitoring/docker-compose-monitoring.yml logs -f <service-name>
```

2. **Restart services**:
```bash
# Restart entire monitoring stack
docker-compose -f monitoring/docker-compose-monitoring.yml restart

# Restart specific service
docker-compose -f monitoring/docker-compose-monitoring.yml restart alertmanager
```

3. **Verify endpoints and connectivity**:
```bash
# Check API health and metrics
curl http://localhost:8000/health
curl http://localhost:8000/metrics

# Verify Prometheus targets
curl http://localhost:9090/api/v1/targets

# Test model prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great movie!", "model_type": "traditional"}'
```

4. **Debug container issues**:
```bash
# Check container status
docker ps | findstr monitoring  # Windows
docker ps | grep monitoring    # Linux/Mac

# Inspect container configuration
docker inspect monitoring-sentiment-api-1

# Access container shell for debugging
docker exec -it monitoring-sentiment-api-1 /bin/bash
```

### Performance Optimization

- **Monitoring Overhead**: Metrics collection adds ~5-10ms latency
- **Resource Usage**: Monitor stack requires ~2GB additional RAM
- **Data Retention**: Configure Prometheus retention based on storage capacity
- **Scrape Intervals**: Adjust collection frequency for performance vs accuracy

## 🔧 Requirements

### Dependencies by Lab

```txt
# Core Dependencies (All Labs)
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.6.1
tensorflow==2.16.1
nltk==3.8.1
joblib==1.3.2

# MLOps & Experiment Tracking (Lab 1)
mlflow==2.21.3
optuna==3.5.0
ray==2.7.0

# Web API & Deployment (Lab 2)
fastapi==0.110.0
uvicorn==0.27.1
jinja2==3.1.3
python-multipart==0.0.9

# Monitoring & Observability (Lab 3)
prometheus-fastapi-instrumentator==6.1.0
prometheus-client==0.19.0
psutil==5.9.8
structlog==23.2.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Additional Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Prometheus Documentation**: https://prometheus.io/docs/
- **Grafana Documentation**: https://grafana.com/docs/
- **Docker Documentation**: https://docs.docker.com/

---

**That is all for 3 Lab**