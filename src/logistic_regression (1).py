import sys,os
sys.path.append(os.path.join(".."))

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# Importing Data and normalizing
def import_data():
    X, y = fetch_openml('mnist_784', return_X_y=True)
    
    X_norm = X/255 # normalizing
    
    return X_norm, y

# Train-test split + print report
def split(X_norm, y):
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, random_state = 42, test_size=0.25)
    
    return X_train, X_test, y_train, y_test
    
def LogReg(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(penalty='none', tol=0.1, solver='saga', multi_class="multinomial").fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    report = classification_report(y_test, y_pred)
    
    print(report)
    
    return report
    
# Save report
def report_output(report):
    outpath = os.path.join("..","out", "lr_report.txt")
    
    with open(outpath,"w") as file:
        file.write(str(report))

# Defining main
def main():
    #args = parse_args()
    X_norm, y = import_data()
    X_train, X_test, y_train, y_test = split(X_norm, y)
    report = LogReg(X_train, X_test, y_train, y_test)
    report_output(report)
             
# Running main
if __name__ == "__main__":
    main()