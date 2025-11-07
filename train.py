# train.py
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import os

# Cargar y preprocesar MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0

# ----------------------------
# 1. Regresión Logística
# ----------------------------
print("Entrenando regresión logística...")
lr = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='multinomial')
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Regresión logística - Precisión: {lr_acc:.4f}")

# Guardar modelo y predicciones
joblib.dump(lr, "models/logistic_regression.pkl")
np.save("results/lr_predictions.npy", lr_pred)
np.save("results/y_test.npy", y_test)

# ----------------------------
# 2. Red Neuronal (MLP simple)
# ----------------------------
print("Entrenando red neuronal...")
nn = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
nn.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1, validation_split=0.1)

nn_pred = nn.predict(x_test).argmax(axis=1)
nn_acc = accuracy_score(y_test, nn_pred)
print(f"Red neuronal - Precisión: {nn_acc:.4f}")

# Guardar modelo y predicciones
nn.save("models/neural_network.h5")
np.save("results/nn_predictions.npy", nn_pred)

# ----------------------------
# Guardar métricas
# ----------------------------
with open("results/metrics.txt", "w") as f:
    f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
    f.write(f"Neural Network Accuracy: {nn_acc:.4f}\n")

print("Entrenamiento completado. Modelos y resultados guardados.")