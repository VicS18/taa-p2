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
import keras_tuner as kt
from tensorflow import keras

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3,4"

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, TextVectorization, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Embedding, GlobalMaxPooling1D, Input


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
from sklearn.model_selection import train_test_split

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

LOAD_MODEL = False

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'
BATCH_SIZE = 32

nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

# For stemming
stemmer = EnglishStemmer()
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc) if w not in STOP_WORDS)

# Text dataset processing
def load_data(filename, test_size=0.3):
    print("Loading dataset...")
    emails = pd.read_csv(filename)

    emails = emails.loc[:10_000,:]

    labels = emails[SPAM_LABEL].values

    X_train, X_test, y_train, y_test = train_test_split(emails[TEXT_LABEL].values, labels, test_size = test_size)

    # # Convert to TensorFlow datasets
    # train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    # test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    # # Cache and prefetch
    # print("Caching and prefetching configuration...")
    # AUTOTUNE = tf.data.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print("Loading finished...")
    return X_train, X_test, y_train, y_test


def create_model(hp, vectorizer): 
    embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=300, step=32)

    hp_conv_units = hp.Int('conv_units', min_value=32, max_value=512, step=32)
    hp_kernel_size = hp.Int('kernel_size', min_value=1, max_value=5, step=1)
    hp_dense_units = hp.Int('dense_units', min_value=32, max_value=512, step=32)

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
    model.add(Dense(units=1, activation = 'sigmoid'))
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
    
    def fit(self, hp, model, X_train, y_train, **kwargs):
        return model.fit(X_train, y_train, **kwargs)
    

if __name__ == "__main__":
    # Preprocess and split dataset
    print("=== LOADING + PREPROCESSING ===")
    start = time.time()
    X_train, X_test, y_train, y_test = load_data(DATASET_PATH, 0.3)

    print("Fitting vectorizer...")
    vectorize_layer = TextVectorization(output_mode='int', standardize='lower_and_strip_punctuation', split='whitespace')
    vectorize_layer.adapt(X_train)

    print(f"Loading + preprocessing time: {time.time() - start}")

    print("=== HYPERPARAMETER TUNING ===")
    start = time.time()

    tuner = kt.Hyperband(CNNSpamModel(vectorize_layer),
                        objective='val_accuracy',
                        max_epochs=10,
                        factor=3,
                        directory='my_dir',
                        project_name='taa_p2_cnn')
    
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2)

    print(f"HT time: {time.time() - start}")

    tuner.search_space_summary()

    hypermodel = CNNSpamModel(vectorize_layer)
    best_hp = tuner.get_best_hyperparameters()[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hp.get('units')} and the optimal learning rate for the optimizer
    is {best_hp.get('learning_rate')}.
    """)

    model = hypermodel.build(best_hp)

    model.evaluate(X_test, y_test)

    model.summary()

    # model.fit(
    #     X_train,
    #     y_train,
    #     batch_size=32,
    #     validation_split=0.3,
    #     epochs=32,
    # )


    # model.summary()

    # print(f"Training time: {time.time() - start}")

    # print("=== TESTING ===")
    # start = time.time()
    # model.evaluate(X_test, y_test)
    # print(f"Testing time: {time.time() - start}")

