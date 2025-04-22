import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

train_data = scipy.io.loadmat('data/training_data.mat')
train_labels = scipy.io.loadmat('data/training_label.mat')
X_train = train_data['X_train3']
y_train = train_labels['Ytrain']

# Fix the shape of y_train
if len(y_train.shape) > 1:
    y_train = np.squeeze(y_train)  # Remove singleton dimensions
    # If still multi-dimensional, flatten it
    if len(y_train.shape) > 1:
        y_train = y_train.ravel()

# Load test data from .mat file
test_data = scipy.io.loadmat('data/test_data.mat')
test_labels = scipy.io.loadmat('data/test_label.mat')
X_test = test_data['X_test']
y_test = test_labels['Ytest']

# Fix the shape of y_test
if len(y_test.shape) > 1:
    y_test = np.squeeze(y_test)  # Remove singleton dimensions
    # If still multi-dimensional, flatten it
    if len(y_test.shape) > 1:
        y_test = y_test.ravel()

#--This was for choosing the right keys---
# print("Training data keys:", train_data.keys())
# print("Training labels keys:", train_labels.keys())
# print("Test data keys:", test_data.keys())
# print("Test labels keys:", test_labels.keys())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'kNN Accuracy: {accuracy_knn:.4f}')

# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f'Decision Tree Accuracy: {accuracy_dt:.4f}')

# 3. Support Vector Machine
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM Accuracy: {accuracy_svm:.4f}')

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.4f}')

# 5. Multi-layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
print(f'MLP Accuracy: {accuracy_mlp:.4f}')