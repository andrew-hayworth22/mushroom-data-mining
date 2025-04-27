import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("Reading preprocessed dataset...")

if not os.path.isfile('mushrooms_processed.csv'):
    print("Preprocessed dataset not found! Run the preprocess command first...")
    exit(1)

dataset = pd.read_csv("mushrooms_processed.csv")

print("Preparing dataset for model training...")

features = dataset.loc[:, dataset.columns != 'class']
target = np.ravel(dataset.loc[:, dataset.columns == 'class'])

target_binary = (target > np.median(target)).astype(int)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

print("Training logistic regression model...")

lr = LogisticRegression()
lr.fit(features_train, target_train)

print("Evaluating logistic regression model...")

target_pred = lr.predict(features_test)
lr_accuracy = accuracy_score(target_test, target_pred)

print("Logistic regression accuracy: {:.2f}%".format(lr_accuracy * 100))

print("Training random forest classifier...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features_train, target_train)

print("Evaluating random forest classifier...")

target_pred = rf.predict(features_test)
rf_accuracy = accuracy_score(target_test, target_pred)

print("Random forest classifier accuracy: {:.2f}%".format(rf_accuracy * 100))

print("Training SVM model...")

svm = SVC(kernel='linear')
svm.fit(features_train, target_train)

print("Evaluating SVM model...")

target_pred = svm.predict(features_test)
svm_accuracy = accuracy_score(target_test, target_pred)

print("SVM classifier accuracy: {:.2f}%".format(svm_accuracy * 100))

print("Rendering graphs...")

plt.bar(
    ['Logistic Regression', 'Random Forest Classifier', 'Support Vector Machine'],
    [lr_accuracy, rf_accuracy, svm_accuracy])
plt.xlabel('Model Accuracy')
plt.title('Accuracy of each ML model')

plt.show()