import subprocess

def run_data_preprocessing():
    print("Running data preprocessing...")
    # Chạy script tiền xử lý dữ liệu
    subprocess.run(["python", "src/data/data_preprocessing.py"])

def run_training_and_fine_tuning():
    print("Training models...")
    # Chạy script huấn luyện mô hình học máy và học sâu
    subprocess.run(["python", "src/models/traditional_model.py"])
    subprocess.run(["python", "src/models/deep_learning_models.py"])

def main():
    # Chạy toàn bộ pipeline
    run_data_preprocessing()
    run_training_and_fine_tuning()

if __name__ == '__main__':
    main()
