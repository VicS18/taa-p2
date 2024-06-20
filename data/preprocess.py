import pandas as pd
from langdetect import detect
import re
data = pd.read_csv('../Datasets/combined_data.csv')

X = data.iloc[:,1]
y = data.iloc[:,0]
num_examples = X.shape[0]
file = open("../Datasets/cleaned_combined_data.csv", "w+")

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown' 

multiline = re.compile('.*\n.*')
for index in range(num_examples):
    lang = detect_language(X[index])
    if lang == 'en':
        file.write(str(y[index]))
        file.write(",")
        if multiline.match(X[index]): #Multiline text
            file.write("\"")
            file.write(X[index])
            file.write("\"")
        else:
            file.write(X[index])
        file.write("\n")

file.close()