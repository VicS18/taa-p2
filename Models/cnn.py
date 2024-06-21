import re
import string
import numpy as np
import pandas as pd
import time
import nltk
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, TextVectorization, Conv1D, Dropout, GlobalMaxPooling1D, Input

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Ensure stopwords are downloaded
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'
BATCH_SIZE = 16  # Reduce batch size to reduce memory usage
TUNE = True

# Data loading function
def load_data(filename, test_size=0.3):
    print("Loading dataset...")
    emails = pd.read_csv(filename)

    # Reduce size for debugging
    # emails = emails.sample(n=10_000, random_state=1)  # Sample 10,000 emails for debugging

    labels = emails[SPAM_LABEL].values
    texts = emails[TEXT_LABEL].values

    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=test_size, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    print("Loading finished...")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model creation function
def create_model(hp, vectorizer):
    embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=128, step=32)
    hp_conv_units = hp.Int('conv_units', min_value=16, max_value=32, step=8) 
    hp_kernel_size = hp.Int('kernel_size', min_value=1, max_value=3, step=1) 
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=256, step=32) 

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)
    model.add(Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=embedding_dim, name="embedding"))
    model.add(Conv1D(filters=hp_conv_units, kernel_size=hp_kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    if hp.Boolean('denser'):
        model.add(Dropout(0.5))
        model.add(Dense(units=hp_dense_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    print("Model compiled...")

    return model

class CNNSpamModel(kt.HyperModel):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        
    def build(self, hp):
        return create_model(hp, self.vectorizer)
    
    def fit(self, hp, model, dataset, **kwargs):
        return model.fit(dataset, **kwargs)

# Main script
if __name__ == "__main__":
    # Configure TensorFlow to only allocate memory as needed
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Preprocess and split dataset
    print("=== LOADING + PREPROCESSING ===")
    start = time.time()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(DATASET_PATH, 0.3)

    print("Fitting vectorizer...")
    vectorize_layer = TextVectorization(output_mode='int', standardize='lower_and_strip_punctuation', split='whitespace')
    
    # Add verbose logging
    start_vectorize = time.time()
    vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(X_train).batch(BATCH_SIZE))
    print(f"Vectorization adaptation time: {time.time() - start_vectorize}")

    print(f"Loading + preprocessing time: {time.time() - start}")

    print("=== HYPERPARAMETER TUNING ===")
    start = time.time()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    tuner = kt.Hyperband(CNNSpamModel(vectorize_layer),
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='taa_p2_cnn')
    
    if TUNE:
        tuner.search(train_dataset, validation_data=val_dataset, epochs=50)

        print(f"HT time: {time.time() - start}")

        tuner.search_space_summary()

    hypermodel = CNNSpamModel(vectorize_layer)
    best_hp = tuner.get_best_hyperparameters()[0]

    # print(f"""
    # The hyperparameter search is complete. The optimal number of units in the first densely-connected
    # layer is {best_hp.get('units')} and the optimal learning rate for the optimizer
    # is {best_hp.get('learning_rate')}.
    # """)

    model = hypermodel.build(best_hp)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    model.evaluate(test_dataset)

    model.summary()
