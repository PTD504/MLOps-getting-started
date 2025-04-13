import subprocess
import os
import mlflow
from mlflow.tracking import MlflowClient

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
    print(f"MLflow tracking set to experiment '{experiment_name}' with ID: {experiment_id}")

def run_data_preprocessing():
    print("Running data preprocessing...")
    # Run data preprocessing script
    subprocess.run(["python", "src/data/data-preprocessing.py"])

def run_training_and_fine_tuning():
    print("Training models...")
    # Run training scripts for traditional and deep learning models
    subprocess.run(["python", "src/models/traditional_models.py"])
    subprocess.run(["python", "src/models/deep_learning_models.py"])

def main():
    # Setup MLflow tracking
    setup_mlflow()
    
    # Run pipeline components
    print("Starting ML pipeline...")
    run_data_preprocessing()
    run_training_and_fine_tuning()
    
    print("Pipeline execution completed successfully!")

if __name__ == '__main__':
    main()