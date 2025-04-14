import polars as pl
import pandas as pd  # giữ để tương thích ngược
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
import os
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def preprocess_data():
    """Preprocess IMDB dataset and save train/test splits"""
    # Start MLflow run for tracking preprocessing
    with mlflow.start_run(run_name="data_preprocessing"):
        print("Bắt đầu tiền xử lý dữ liệu...")
        
        # Tạo thư mục data nếu chưa có
        os.makedirs("data", exist_ok=True)
        
        # Load dataset
        print("Đang tải dataset...")
        # Sử dụng Polars thay vì Pandas
        try:
            data = pl.read_csv("data/IMDB-Dataset.csv")
            # Log dataset details
            mlflow.log_param("dataset_size", data.height)
            mlflow.log_param("dataset_source", "IMDB-Dataset.csv")
            mlflow.log_param("dataset_version", "1.0")
            
            # Check class distribution
            sentiment_counts = data.group_by("sentiment").count()
            sentiment_dict = {row['sentiment']: row['count'] for row in sentiment_counts.to_dicts()}
            mlflow.log_params({f"class_{k}": v for k, v in sentiment_dict.items()})
            
            # Clean text data
            print("Đang xử lý text...")
            data = data.with_columns(
                pl.col("review").apply(clean_text).alias("cleaned_review")
            )
            
            # Convert sentiment to binary labels
            data = data.with_columns(
                pl.when(pl.col("sentiment") == "positive").then(1).otherwise(0).alias("label")
            )
            
            # Convert back to Pandas for sklearn compatibility
            pandas_data = data.to_pandas()
            
            # Split data into train, validation, and test sets
            print("Chia dataset...")
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                pandas_data['cleaned_review'], pandas_data['label'], 
                test_size=0.2, random_state=42, stratify=pandas_data['label']
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=0.25, random_state=42, stratify=y_train_val
            )
            
            # Log split sizes
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("test_size", len(X_test))
            
            # Save processed data
            print("Lưu dữ liệu đã xử lý...")
            train_data = pd.DataFrame({'text': X_train, 'label': y_train})
            val_data = pd.DataFrame({'text': X_val, 'label': y_val})
            test_data = pd.DataFrame({'text': X_test, 'label': y_test})
            
            train_data.to_csv("data/train.csv", index=False)
            val_data.to_csv("data/val.csv", index=False)
            test_data.to_csv("data/test.csv", index=False)
            
            # Log artifacts
            mlflow.log_artifact("data/train.csv")
            mlflow.log_artifact("data/val.csv")
            mlflow.log_artifact("data/test.csv")
            
            print("Tiền xử lý dữ liệu hoàn thành!")
        except Exception as e:
            print(f"Lỗi trong quá trình xử lý dữ liệu: {e}")

if __name__ == "__main__":
    preprocess_data()