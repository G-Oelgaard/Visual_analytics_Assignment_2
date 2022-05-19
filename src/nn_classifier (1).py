import sys,os
sys.path.append(os.path.join(".."))

import numpy as np
from utils.neuralnetwork import NeuralNetwork

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Importing Data and normalizing
def import_data():
    X, y = fetch_openml('mnist_784', return_X_y=True)
    
    X_norm = X/255 # normalizing
    
    return X_norm, y

# Train-test split
def split(X_norm, y):
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state = 42, test_size=0.25)
    
    return X_train, X_test, y_train, y_test

# label_binarizer
def binarizer(y_train, y_test):
    lb = LabelBinarizer()
    
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    
    return y_train, y_test

# Neutral network
def neural_network(X_train, y_train):
    print("[INFO] training network...")
    
    input_shape = X_train.shape[1]
    
    nn = NeuralNetwork([input_shape, 64, 10])
    
    print(f"[INFO] {nn}")
    
    nn.fit(X_train, y_train, epochs=15, displayUpdate=1) # epoches set to 15 in the interest of time
    
    return nn

# Classification report
def class_report(nn, X_test, y_test):
    predictions = nn.predict(X_test)
    
    y_pred = predictions.argmax(axis=1)
    
    report = classification_report(y_test.argmax(axis=1), y_pred)
    
    print(report)
    
    return report
    
# Save report
def report_output(report):
    outpath = os.path.join("..","out", "nn_report.txt")
    
    with open(outpath,"w") as file:
        file.write(str(report))
    
# Defining main
def main():
    X_norm, y = import_data()
    X_train, X_test, y_train, y_test = split(X_norm, y)
    y_train, y_test = binarizer(y_train, y_test)
    nn = neural_network(X_train, y_train)
    report = class_report(nn, X_test, y_test)
    report_output(report)
             
# Running main
if __name__ == "__main__":
    main()
