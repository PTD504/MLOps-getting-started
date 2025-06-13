## Overview

This project implements a full MLOps workflow for sentiment analysis, comparing traditional machine learning approaches (Logistic Regression, SVM, Naive Bayes) with deep learning models (LSTM). It includes data preprocessing, model training, hyperparameter optimization, distributed execution, experiment tracking, and model serving via a web API.

## Assignment Requirement of Lab3:
Với API đã xây dựng ở Lab 2, hãy thêm Monitoring và Logging service, trong đó, yêu cầu tối thiểu như sau:

1. **Monitoring service phải monitor được các tài nguyên cơ bản của server:**
+ CPU usage
+ GPU usage (optional -> cộng điểm nếu có thể monitor)
+ RAM usage
+ Disk space, disk IO
+ Network IO (total transmitted, total receieved)

2. **Monitoring API đã xây dựng ở Lab 2 bằng các thông số như:**
+ Request per second
+ Error rate
+ Latency

3. **Monitoring model:**
+ Inference speed (CPU time và GPU time)
+ Confidence score

4. **Logging service cẩn capture được log từ:**
+ syslog: đây là log của server, giúp xác định lỗi không phải từ ứng dụng (ví dụ temperature cao -> tự shutdown đột ngột -> không phải do API gây ra)
+ stdout: đây là log stream hiển thị trên console
+ stderror: đây là log stream sẽ in ra traceback khi có lỗi xảy ra
+ logfile: tùy thuộc vào đường dẫn file log của ứng dụng các bạn đã xây dựng ở Lab 2, capture log từ file này
Khi có bất thường trong quá trình monitoring (ví dụ error rate cao > 50% hoặc confidence score < 0.6), sử dụng Alertmanager để thông báo hoặc thực hiện action được define trước. Lưu ý thế nào là bất thường do các bạn tự định nghĩa, action có thể là thông báo qua mail, telegram, slack hoặc trigger action train lại mô hình.

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
docker-compose up --build -d
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

## Video demo
Link: [click here](https://drive.google.com/file/d/1kz0grRHgfGDE0eng2kFirgOmrQ4-Fk5S/view?usp=sharing)

## Service Ports

This project utilizes several services, each accessible on a specific port on your `localhost` after running `docker-compose up -d`.

| Service                      | Host Port | Access URL                        | Notes                                               |
| :--------------------------- | :-------- | :-------------------------------- | :-------------------------------------------------- |
| FastAPI Application          | `8000`    | `http://localhost:8000`           | API and `/metrics` endpoint                         |
| Prometheus                   | `9090`    | `http://localhost:9090`           | Metrics, Targets, Alerts UI, Graph/Query            |
| Grafana                      | `3000`    | `http://localhost:3000`           | Dashboards (Default Login: `admin` / `admin`)       |
| Alertmanager                 | `9093`    | `http://localhost:9093`           | View active/silenced alerts                         |
| Loki                         | `3100`    | (Accessed via Grafana data source)  | Log storage; queried by Grafana                   |
| Node Exporter                | `9100`    | `http://localhost:9100/metrics`   | Server metrics; primarily scraped by Prometheus     |
| Promtail HTTP (Internal)     | `9080`    | (Not for direct user interaction) | Promtail's internal HTTP server for its operations  |


**Note:**
*   The "Access URL" for Loki is indirect; you query Loki logs through Grafana's "Explore" view by selecting the Loki data source.
*   The Node Exporter metrics endpoint is typically consumed by Prometheus, not directly accessed by users routinely.
*   Promtail's HTTP port is for its own operational purposes and not for end-user log viewing.


## Requirements:

All packages and libraries needed for this code is stored in the requirements.txt file. They are declared with their specific version to avoid the conflict
