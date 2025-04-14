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

def read_data(file_path):
    """ read the data that is already preprocessed in preprocessing_data.py"""
    df = pl.read_parquet(file_path)
    return df

def splitting_dataset(df):
    X = df['cleaned_review']
    y = df['sentiment']
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=42)
    print(f"[INFO]: Splitting dataset done!")
    return X_train, y_train, X_val, y_val, X_test, y_test

def tokenize_and_padding(X_train, X_val, X_test, max_words=5000, max_len=318):
    # convert seires Polars to list
    X_train_lst = X_train.to_list()
    X_test_lst = X_test.to_list()
    X_val_lst = X_val.to_list()
    # cr eate a tokenizer
    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(X_train_lst)

    print(f"[INFO]: vocab size = {len(tokenizer.word_index)}")
    print(f"[INFO]: number of document (equivalent to X_train.shape) = {tokenizer.document_count}")
    
    print("[INFO]: Begin the process of tokenizing")
    tokenized_X_train = tokenizer.texts_to_sequences(X_train_lst)
    tokenized_X_test = tokenizer.texts_to_sequences(X_test_lst)
    tokenized_X_val = tokenizer.texts_to_sequences(X_val_lst)
    print("[INFO]: Tokenizing done!")

    print(f"[INFO]: Dimension of first sequences (train):  {len(tokenized_X_train[0])}")
    print(f"[INFO]: Dimension of first sequences (test):  {len(tokenized_X_test[0])}")
    print(f"[INFO]: Dimension of first sequences (val):  {len(tokenized_X_val[0])}")
    max_length = 0
    for sequence in tokenized_X_train:
        max_length = len(sequence) if max_length < len(sequence) else max_length
    print(f"[INFO]: Maximum sequence length for train set: {max_length}")

    max_length = 0
    for sequence in tokenized_X_test:
        max_length = len(sequence) if max_length < len(sequence) else max_length
    print(f"[INFO]: Maximum sequence length for test set: {max_length}")

    max_length = 0
    for sequence in tokenized_X_val:
        max_length = len(sequence) if max_length < len(sequence) else max_length
    print(f"[INFO]: Maximum sequence length for val set: {max_length}")


    # add padding
    preprocessed_X_train = pad_sequences(tokenized_X_train, maxlen=max_len, padding='post')
    preprocessed_X_test = pad_sequences(tokenized_X_test, maxlen=max_len, padding='post')
    preprocessed_X_val = pad_sequences(tokenized_X_val, maxlen=max_len, padding='post')

    print(f"[INFO]: Padding done!")
    print(f"length of first entry of preprocessed_X_train: = {len(preprocessed_X_train[0])}")
    print(f"length of first entry of preprocessed_X_test: = {len(preprocessed_X_test[0])}")
    print(f"length of first entry of preprocessed_X_val: = {len(preprocessed_X_val[0])}")
    print(f"preprocessed_X_train[0]: \n {preprocessed_X_train[0]}")

    return preprocessed_X_train, preprocessed_X_val, preprocessed_X_test, tokenizer

def save_data(df, output_dir, tokenizer_file='tokenizer.pkl'):
    """
    Save the preprocessed DataFrame to a CSV file.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = splitting_dataset(df)
    preprocessed_X_train, preprocessed_X_val, preprocessed_X_test,  tokenizer = tokenize_and_padding(X_train, X_val, X_test)
    # save the tokenizer
    with open(tokenizer_file, 'wb') as file:
        pickle.dump(tokenizer, file)
    # print(f"Tokenizer saved to: {tokenizer_file}")
    print(f"[INFO]: Tokenizer saved to: {tokenizer_file}")

    # 418 samples are duplicates
    np.save(output_dir + 'X_train.npy', preprocessed_X_train)
    np.save(output_dir + 'X_test.npy', preprocessed_X_test)
    np.save(output_dir + 'X_val.npy', preprocessed_X_val)
    np.save(output_dir + 'y_train.npy', y_train)
    np.save(output_dir + 'y_test.npy', y_test)
    np.save(output_dir + 'y_val.npy', y_val)
    print(f"[INFO]: Preprocessed data saved to: {output_dir}")

def main():
    # define the path
    input_file = '/MLOps-getting-started/data/IMDB-Dataset-Processed.parquet'
    df = read_data(input_file)
    print(df.head())
    # save the data
    save_data(df, '/MLOps-getting-started/data/')  

if __name__ == '__main__':
    # main()
    data_dir ='/MLOps-getting-started/data/'
    X_train = np.load(data_dir + 'X_train.npy', allow_pickle=True)
    X_test = np.load(data_dir + 'X_test.npy', allow_pickle=True)
    X_val = np.load(data_dir + 'X_val.npy', allow_pickle=True)

    y_train = np.load(data_dir + 'y_train.npy', allow_pickle=True)
    y_test = np.load(data_dir + 'y_test.npy', allow_pickle=True)
    y_val = np.load(data_dir + 'y_val.npy', allow_pickle=True)
    print(f"X train shape = {X_train.shape}")
    print(f"y train shape = {y_train.shape}")
    print(f"X test shape = {X_test.shape}")
    print(f"y test shape = {y_test.shape}")
    print(f"X val shape = {X_val.shape}")
    print(f"y val shape = {y_val.shape}")