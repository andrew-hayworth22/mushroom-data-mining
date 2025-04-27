import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


print("Reading preprocessed dataset...")

if not os.path.isfile('mushrooms_processed.csv'):
    print("Preprocessed dataset not found! Run the preprocess command first...")
    exit(1)

dataset = pd.read_csv("mushrooms_processed.csv")

print("Preparing dataset for model training before pruning...")

features = dataset.loc[:, dataset.columns != 'class']
target = np.ravel(dataset.loc[:, dataset.columns == 'class'])

target_binary = (target > np.median(target)).astype(int)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

print("Evaluating random forest classifier before pruning...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features_train, target_train)

target_pred = rf.predict(features_test)
rf_accuracy = accuracy_score(target_test, target_pred)

print("Random forest classifier accuracy before pruning: {:.2f}%".format(rf_accuracy * 100))

print("Feature importances: ")

feature_importances = rf.feature_importances_
feature_importance_threshold = np.float64(0.02)

plt.barh(features.columns, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')

for i in range(len(feature_importances)):
    print(f"{features.columns[i]}: {str(feature_importances[i])} {" PRUNED" if feature_importances[i] < feature_importance_threshold else ""}")
    if feature_importances[i] < feature_importance_threshold:
        dataset = dataset.loc[:, dataset.columns != features.columns[i]]

print("Preparing dataset for model training after pruning...")

features = dataset.loc[:, dataset.columns != 'class']
target = np.ravel(dataset.loc[:, dataset.columns == 'class'])

target_binary = (target > np.median(target)).astype(int)

features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)

print("Evaluating random forest classifier after pruning...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(features_train, target_train)

target_pred = rf.predict(features_test)
rf_accuracy = accuracy_score(target_test, target_pred)

print("Random forest classifier accuracy after pruning: {:.2f}%".format(rf_accuracy * 100))

print("Writing pruned dataset...")

dataset.to_csv("mushrooms_pruned.csv")

print("Persisting ML model...")

pickle.dump(rf, open('rf.sav', 'wb'))

print("Successfully generated final ML model!")

plt.show()