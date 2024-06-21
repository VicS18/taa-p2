#
# Model definition, training and testing for Naive Bayes
#

import numpy as np
import pandas as pd
import time 
import joblib
import nltk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"

from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, TextVectorization, GlobalAveragePooling1D

from tensorflow.data.experimental import make_csv_dataset
from tensorflow.keras.utils import split_dataset
from tensorflow.keras.losses import BinaryCrossentropy

if tf.test.gpu_device_name():
   print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, auc, classification_report, roc_curve
from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

LOAD_MODEL = True

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

# For stemming
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc) if w not in STOP_WORDS)

# Text dataset processing
def load_data(filename, test_size=0.3):
    spam_mail = make_csv_dataset(
        filename, batch_size=4,
        label_name=SPAM_LABEL)
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds, test_ds = keras.utils.split_dataset(spam_mail, right_size=test_size)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, test_ds


def create_model(train_ds): 
    embedding_dim=16

    vectorize_layer = TextVectorization()

    text_ds = train_ds.map(lambda x, y: x)

    vectorize_layer.adapt(text_ds)

    model = Sequential([
        vectorize_layer,
        Embedding(vectorize_layer.get_vocabulary(), embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model
    

if __name__ == "__main__":
    # Preprocess and split dataset
    train_ds, test_ds = load_data(DATASET_PATH, 0.3)

    model = create_model(train_ds)

    start = time.time()

    model.fit(
        train_ds,
        validation_split=0.3,
        epochs=15,
    )

    end = time.time()
    print(f"Training time: {end - start}")
