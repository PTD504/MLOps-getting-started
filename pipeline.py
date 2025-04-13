import subprocess
import os
import mlflow
from mlflow.tracking import MlflowClient
import ray  # Thêm Ray cho distributed computing

# Khởi tạo Ray
ray.init(ignore_reinit_error=True)

def setup_mlflow():
    """Setup MLflow tracking server"""
    print("Setting up MLflow...")
    # Create mlruns directory if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Set tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create experiment if it doesn't exist
    experiment_name = "sentiment-analysis"
    client = MlflowClient()
    
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking has been set up with experiment '{experiment_name}', ID: {experiment_id}")

@ray.remote
def run_data_preprocessing(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"  # Quan trọng: thiết lập encoding UTF-8
    
    print("Running data preprocessing...")
    try:
        # Sử dụng env để truyền biến môi trường
        result = subprocess.run(
            ["python", "src/data/data-preprocessing.py"],
            env=env_vars,
            check=True  # Hiển thị lỗi nếu có
        )
        return "Data preprocessing completed"
    except subprocess.CalledProcessError as e:
        print(f"Error running preprocessing: {e}")
        return "Data preprocessing failed"

@ray.remote
def run_traditional_models(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"  # Quan trọng: thiết lập encoding UTF-8
    
    print("Training traditional models...")
    try:
        result = subprocess.run(
            ["python", "src/models/traditional_models.py"],
            env=env_vars,
            check=True
        )
        return "Traditional model training completed"
    except subprocess.CalledProcessError as e:
        print(f"Error running traditional models: {e}")
        return "Traditional model training failed"

@ray.remote
def run_deep_learning_models(tracking_uri, experiment_id):
    # Thiết lập biến môi trường
    env_vars = dict(os.environ)
    env_vars["MLFLOW_TRACKING_URI"] = tracking_uri
    env_vars["MLFLOW_EXPERIMENT_ID"] = experiment_id
    env_vars["PYTHONIOENCODING"] = "utf-8"  # Quan trọng: thiết lập encoding UTF-8
    
    print("Training deep learning models...")
    try:
        result = subprocess.run(
            ["python", "src/models/deep_learning_models.py"],
            env=env_vars,
            check=True
        )
        return "Deep learning model training completed"
    except subprocess.CalledProcessError as e:
        print(f"Error running deep learning models: {e}")
        return "Deep learning model training failed"

def main():
    # Setup MLflow tracking
    setup_mlflow()
    
    # Lấy thông tin để truyền cho các task
    tracking_uri = mlflow.get_tracking_uri()
    experiment_id = mlflow.get_experiment_by_name("sentiment-analysis").experiment_id
    
    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment ID: {experiment_id}")
    
    # Chạy pipeline components
    print("Starting ML pipeline...")
    
    try:
        # Truyền thông tin MLflow vào các task
        preprocessing_result = ray.get(run_data_preprocessing.remote(tracking_uri, experiment_id))
        print(preprocessing_result)
        
        # Tương tự cho các task khác
        traditional_future = run_traditional_models.remote(tracking_uri, experiment_id)
        deep_learning_future = run_deep_learning_models.remote(tracking_uri, experiment_id)
        
        # Đợi các tác vụ hoàn thành
        traditional_result, deep_learning_result = ray.get([traditional_future, deep_learning_future])
        
        print(traditional_result)
        print(deep_learning_result)
        
        print("Pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in pipeline: {e}")

if __name__ == '__main__':
    main()