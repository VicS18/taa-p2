import re
import string
from matplotlib import pyplot as plt
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

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


# Ensure stopwords are downloaded
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english')).union({"escapenumber"})

DATASET_PATH = '../Datasets/combined_data.csv'
TEXT_LABEL = 'text'
SPAM_LABEL = 'label'
BATCH_SIZE = 16

LOAD_MODEL = False

# Data loading function
def load_data(filename, test_size=0.3):
    print("Loading dataset...")
    emails = pd.read_csv(filename)

    emails = emails.sample(n=30_000, random_state=1)

    labels = emails[SPAM_LABEL].values
    texts = emails[TEXT_LABEL].values

    X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=test_size, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    print("Loading finished...")
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model creation function
def create_model(vectorizer):
    embedding_dim = 128
    conv_units = 32
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

# Plotting functions
def plot_roc_auc(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_true)))
    plt.xticks(tick_marks, ['Not Spam', 'Spam'], rotation=45)
    plt.yticks(tick_marks, ['Not Spam', 'Spam'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
    
    if not LOAD_MODEL:
        print("=== TRAINING ===")
        model = create_model(vectorize_layer)
        model.fit(train_dataset, validation_data=val_dataset, epochs=50, batch_size=BATCH_SIZE)
        model.save('cnn_no_ht.h5')
    else:
        model = tf.keras.models.load_model('cnn_no_ht.h5')

    print("=== EVALUATION ===")
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)
    loss, accuracy = model.evaluate(test_dataset)
    
    model.summary()

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    y_pred_proba = model.predict(test_dataset)
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)  # Assuming threshold of 0.5 for binary classification

    # Plot ROC curve
    plot_roc_auc(y_test, y_pred_proba)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
