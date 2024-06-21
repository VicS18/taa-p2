import re
import string
import numpy as np
import pandas as pd
import time
import nltk
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, TextVectorization, Conv1D, Dropout, GlobalMaxPooling1D, Input
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer

# Ensure stopwords are downloaded
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'
BATCH_SIZE = 16

# Data loading function
def load_data(filename, test_size=0.3):
    print("Loading dataset...")
    emails = pd.read_csv(filename)

    labels = emails[SPAM_LABEL].values
    texts = emails[TEXT_LABEL].values

    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=test_size, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    print("Loading finished...")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model creation function
def create_model(vectorizer):
    embedding_dim = 64
    conv_units = 16
    kernel_size = 3
    dense_units = 128

    model = Sequential()
    model.add(Input(shape=(1,), dtype=tf.string))
    model.add(vectorizer)
    model.add(Embedding(input_dim=len(vectorizer.get_vocabulary()), output_dim=embedding_dim, name="embedding"))
    model.add(Conv1D(filters=conv_units, kernel_size=kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(units=dense_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    print("Model compiled...")

    return model

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
    
    start_vectorize = time.time()
    vectorize_layer.adapt(X_train)
    print(f"Vectorization adaptation time: {time.time() - start_vectorize}")

    print(f"Loading + preprocessing time: {time.time() - start}")

    print("=== TRAINING ===")
    model = create_model(vectorize_layer)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)

    model.fit(train_dataset, validation_data=val_dataset, epochs=50, batch_size=BATCH_SIZE)

    print("=== EVALUATION ===")
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    model.evaluate(test_dataset)

    model.summary()
