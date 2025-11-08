# app.py
import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from streamlit_drawable_canvas import st_canvas

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

results_dir = os.path.join(project_root, "results")
models_dir = os.path.join(project_root, "models")

y_test = np.load(os.path.join(results_dir, "y_test.npy"))
lr_pred = np.load(os.path.join(results_dir, "lr_predictions.npy"))
nn_pred = np.load(os.path.join(results_dir, "nn_predictions.npy"))

# Cargar métricas
with open(os.path.join(results_dir, "metrics.txt")) as f:
    lines = f.readlines()
    lr_acc = float(lines[0].split()[-1])
    nn_acc = float(lines[1].split()[-1])

# Cargar modelos (opcional, solo si quieres hacer predicciones nuevas)
# lr_model = joblib.load("models/logistic_regression.pkl")
# nn_model = load_model("models/neural_network.h5")

# Título
st.title("🔍 Comparación: Regresión Logística vs Red Neuronal (MNIST)")
st.markdown("Proyecto de comparación en el dataset MNIST de dígitos escritos a mano.")

# Mostrar métricas
col1, col2 = st.columns(2)
col1.metric("Regresión Logística", f"{lr_acc:.4f}")
col2.metric("Red Neuronal", f"{nn_acc:.4f}")

# Mostrar ejemplos del test set
st.subheader("Ejemplos de clasificación")
indices = np.random.choice(len(y_test), 5, replace=False)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
(x_train, _), (x_test_orig, _) = tf.keras.datasets.mnist.load_data()
x_test_images = x_test_orig  # Solo para visualizar

for i, idx in enumerate(indices):
    img = x_test_images[idx]
    true_label = y_test[idx]
    lr_label = lr_pred[idx]
    nn_label = nn_pred[idx]

    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Verdadero: {true_label}\nLR: {lr_label} | NN: {nn_label}")
    axes[i].axis("off")

st.pyplot(fig)

# Matrices de confusión
st.subheader("Matrices de confusión")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, lr_pred), annot=False, fmt="d", ax=ax1, cmap="Blues")
ax1.set_title("Regresión Logística")

sns.heatmap(confusion_matrix(y_test, nn_pred), annot=False, fmt="d", ax=ax2, cmap="Greens")
ax2.set_title("Red Neuronal")

st.pyplot(fig)

@st.cache_resource
def load_models():
    lr = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    nn = load_model(os.path.join(models_dir, "neural_network.h5"))
    return lr, nn

lr_model, nn_model = load_models()
# Configuración del canvas
st.title("✍️ Dibuja un dígito y compara clasificadores")
st.markdown("Dibuja un número del 0 al 9 en el lienzo. Ambos modelos intentarán reconocerlo.")

# Canvas para dibujar (280x280 px, escala fácil a 28x28)
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Fondo transparente
    stroke_width=20,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Procesar el dibujo cuando se detecte un cambio
if canvas_result.image_data is not None:
    # Convertir a PIL y escalar a 28x28 en escala de grises
    input_image = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    input_image = input_image.convert("L")  # escala de grises
    input_image = input_image.resize((28, 28))  # MNIST size

    # Convertir a array y normalizar como en el entrenamiento
    img_array = np.array(input_image)
    img_array = 255 - img_array  # invertir: fondo blanco -> negro, trazo negro -> blanco
    img_array = img_array.astype("float32") / 255.0
    img_flat = img_array.reshape(1, -1)  # (1, 784)

    # Predicciones
    lr_pred_proba = lr_model.predict_proba(img_flat)[0]
    lr_pred = lr_model.predict(img_flat)[0]

    nn_pred_proba = nn_model.predict(img_flat)[0]  # img_flat ya es (1, 784)
    nn_pred = np.argmax(nn_pred_proba)

    # Mostrar imagen dibujada
    st.image(img_array, caption="Tu dígito (28x28)", width=150)

    # Mostrar predicciones
    col1, col2 = st.columns(2)
    col1.metric("Regresión Logística", f"{lr_pred}")
    col2.metric("Red Neuronal", f"{nn_pred}")

    # Gráfico de probabilidades
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))

    digits = np.arange(10)
    ax[0].bar(digits, lr_pred_proba, color="steelblue")
    ax[0].set_title("Regresión Logística - Probabilidades")
    ax[0].set_ylim(0, 1)

    ax[1].bar(digits, nn_pred_proba, color="seagreen")
    ax[1].set_title("Red Neuronal - Probabilidades")
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    st.pyplot(fig)

# Sección de comparación del dataset (tu código anterior)
st.markdown("---")
st.subheader("📊 Resultados en el dataset de prueba (MNIST)")

# Cargar datos de prueba y predicciones guardadas
try:
    y_test = np.load(os.path.join(results_dir, "y_test.npy"))
    lr_pred_test = np.load(os.path.join(results_dir, "lr_predictions.npy"))
    nn_pred_test = np.load(os.path.join(results_dir, "nn_predictions.npy"))

    lr_acc = np.mean(lr_pred_test == y_test)
    nn_acc = np.mean(nn_pred_test == y_test)

    col1, col2 = st.columns(2)
    col1.metric("Precisión LR (test)", f"{lr_acc:.4f}")
    col2.metric("Precisión NN (test)", f"{nn_acc:.4f}")

except FileNotFoundError:
    st.warning("Ejecuta 'python train.py' primero para ver los resultados del test.")