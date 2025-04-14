# library for data manipulation
import numpy as np
import pandas as pd
import polars as pl
import re, sys
# preprocessing
import contractions
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import spacy
import pickle
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")
print(f"[INFO]: Spacy model loaded successfully!")
chat_words = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laughter",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "LOL": "Laughing out loud",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "IDC": "I don’t care",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "LMAO": "Laughing my a** off",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing",
}
# Global variables for stopwords and stemmer
STOP_WORDS = set(stopwords.words('english'))  # Includes words like 'a', 'the', etc.


def dataset_info(df):
    print(f"dataset shape = {df.shape}\n")
    # step 2
    print(f"Overview of dataset: \n {df.schema}\nNull Count: \n{df.null_count()}\n")
    # step 3
    print(f"Checking duplicated records: {df.is_duplicated().sum()}")
    print(f"Peform Drop Duplicates: \n")
    df = df.unique()
    print(f"After gdropping duplicates: {df.shape}")
    print(f"Finished Dropping Duplicates")
    return df

def lowercase(text):
    """
    Convert the text to lowercase.
    """
    return text.lower()

def remove_html_tags(text):
    """
    Remove HTML tags from the given text using a regular expression.
    """
    pattern = re.compile(r'<.*?>')
    return re.sub(pattern, '', text)

def remove_urls(text):
    return re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)', '', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('','', string.punctuation))

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words.get(w.upper()))
        else:
            new_text.append(w)
    return " ".join(new_text)

def stopwords_removal(text):
    new_text = []
    for word in text.split():
        if word in STOP_WORDS:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:] # get all items
    new_text.clear()
    return " ".join(x)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def lemmatize_text(text):
    doc = nlp(text)
    lemmatize_text = ' '.join([token.lemma_ for token in doc])
    return lemmatize_text

def tokenize(text):
    token_words = word_tokenize(text)
    return token_words

def preprocess_text(text):
    """
    Preprocess the given text by:
      - Converting to lowercase
      - Removing HTML tags
      - Removing URLs
      - Removing punctuation
      - Chat words conversion
      - Stopwords removal
      - Removing emojis
      - Expanding contractions
      - Lemmatizing
      - Tokenizing
    Returns the cleaned text as a single string.
    """
    # convert to lowercase
    text = lowercase(text)
    # Remove HTML tags
    text = remove_html_tags(text)
    # Remove punctuation and numbers
    text = remove_urls(text)
    # remove punctuation
    text = remove_punctuation(text)
    # convert chat words
    text = chat_conversion(text)
    # reomve stopwords
    text = stopwords_removal(text)
    # remove emojis
    text = remove_emoji(text)
    # expand contractions
    text = expand_contractions(text)
    # lemmatize
    text = lemmatize_text(text)
    # tokenize text
    text = tokenize(text)
    # join the tokens back 
    return text

def load_data(file_path):
    """
    Load dataset from a CSV file given its file path.
    """
    try:
        df = pl.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        # display dataset info
        df = dataset_info(df)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please check the file path.")
        sys.exit(1)

def preprocess_dataset(df, review_column='review', output_column='cleaned_review'):
    """
    Apply preprocessing to a dataset. It creates a new column with cleaned reviews.
    """

    df = df.with_columns([
        pl.col(review_column).map_elements(preprocess_text).alias(output_column)
    ])
    print(f"[INFO]: Preprocessing completed!")
    return df

def save_data_after_preprocessing(df, output_file):
    df.write_parquet(output_file)
    print(f"[INFO]: Data savved to {output_file}")
def main():
    # Define file paths
    input_file = '/MLOps-getting-started/data/IMDB-Dataset.csv'
    output_dir = '/MLOps-getting-started/data/'
    
    # Load the dataset
    df = load_data(input_file)
    
    # Preprocess the review texts
    df = preprocess_dataset(df)
    
if __name__ == '__main__':
    # Define file paths
    input_file = '/MLOps-getting-started/data/IMDB-Dataset.csv'
    output_dir = '/MLOps-getting-started/data/'
    
    # Load the dataset
    df = load_data(input_file)
    
    # Preprocess the review texts
    df = preprocess_dataset(df)
    # save the preprocessed data
    output_file = output_dir + 'preprocessed_data.parquet'
    save_data_after_preprocessing(df, output_file)
