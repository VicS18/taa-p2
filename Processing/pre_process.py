# Text pre-processing, cleaning and feature engineering

# NLTK - Streamline text processing
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy
import string

# Read and clean

def clean_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df = df.dropna()

    df['text'] = df['text'].astype(str)

    # Tokenize using punkt
    df['text'] = df['text'].apply(word_tokenize)

    # To lowercase
    df['text'] = df['text'].apply(lambda tokens: [word.lower() for word in tokens])

    # Remove punctuation
    tr_table = str.maketrans('', '', string.punctuation)
    df['text'] = df['text'].apply(lambda tokens: [ word.translate(tr_table) for word in tokens])

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

    # Remove non-alphabetic tokens
    df['text'] = df['text'].apply(lambda tokens: [word for word in tokens if word.isalpha()])

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])

    return df

if __name__ == '__main__':
    clean_dataset('../Datasets/combined_data.csv').head()