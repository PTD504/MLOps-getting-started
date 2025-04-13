import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import mlflow

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords (optional)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def preprocess_data():
    """Preprocess IMDB dataset and save train/test splits"""
    # Start MLflow run for tracking preprocessing
    with mlflow.start_run(run_name="data_preprocessing"):
        print("Loading dataset...")
        # Load dataset
        data = pd.read_csv("data/IMDB-Dataset.csv")
        
        # Log dataset details
        mlflow.log_param("dataset_size", len(data))
        mlflow.log_param("dataset_source", "IMDB-Dataset.csv")
        
        # Check class distribution
        sentiment_counts = data['sentiment'].value_counts().to_dict()
        mlflow.log_params({f"class_{k}": v for k, v in sentiment_counts.items()})
        
        # Clean text data
        print("Cleaning text data...")
        data['cleaned_review'] = data['review'].apply(clean_text)
        
        # Convert sentiment to binary labels
        data['label'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        
        # Split data into train, validation, and test sets
        print("Splitting dataset...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            data['cleaned_review'], data['label'], 
            test_size=0.2, random_state=42, stratify=data['label']
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
        print("Saving processed data...")
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
        
        print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_data()