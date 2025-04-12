import pandas as pd
import numpy as np
import re  # Regular expressions for text cleaning
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # For stemming
import joblib  # To save/load models and vectorizer (optional)
import sys

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Global variables for stopwords and stemmer
STOP_WORDS = set(stopwords.words('english'))  # Includes words like 'a', 'the', etc.
STEMMER = PorterStemmer()

def remove_html_tags(text):
    """
    Remove HTML tags from the given text using a regular expression.
    """
    pattern = re.compile(r'<.*?>')
    return re.sub(pattern, '', text)

def remove_non_alpha(text):
    """
    Remove punctuation and numbers, keeping only letters and spaces.
    """
    return re.sub(r'[^a-zA-Z\s]', '', text, flags=re.I)

def tokenize_text(text):
    """
    Tokenize the text into a list of words.
    """
    return text.split()

def remove_stopwords(words):
    """
    Remove stopwords from a list of words.
    """
    return [word for word in words if word not in STOP_WORDS]

def apply_stemming(words):
    """
    Apply stemming to a list of words using PorterStemmer.
    """
    return [STEMMER.stem(word) for word in words]

def preprocess_text(text):
    """
    Preprocess the given text by:
      - Removing HTML tags
      - Removing non-alphabet characters
      - Converting to lowercase
      - Tokenizing, removing stopwords and applying stemming
    Returns the cleaned text as a single string.
    """
    # Remove HTML tags
    text = remove_html_tags(text)
    # Remove punctuation and numbers
    text = remove_non_alpha(text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text into words
    words = tokenize_text(text)
    # Remove stopwords
    words = remove_stopwords(words)
    # Apply stemming
    words = apply_stemming(words)
    # Rejoin words into a cleaned string
    return ' '.join(words)

def load_data(file_path):
    """
    Load dataset from a CSV file given its file path.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        sys.exit(1)

def preprocess_dataset(df, review_column='review', output_column='cleaned_review'):
    """
    Apply preprocessing to a dataset. It creates a new column with cleaned reviews.
    """
    df[output_column] = df[review_column].apply(preprocess_text)
    return df

def save_data(df, output_file):
    """
    Save the preprocessed DataFrame to a CSV file.
    """
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to: {output_file}")

def main():
    # Define file paths
    input_file = 'data/IMDB-Dataset.csv'
    output_file = 'data/IMDB-Dataset_preprocessed.csv'
    
    # Load the dataset
    df = load_data(input_file)
    
    # Preprocess the review texts
    df = preprocess_dataset(df)
    
    # Save the preprocessed dataset
    save_data(df, output_file)

if __name__ == '__main__':
    main()
