# download_mnist_images.py
import tensorflow as tf
import os
from PIL import Image

# 1. Cargar MNIST
print("Cargando MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Crear carpetas
base_dir = "raw"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Crear subcarpetas para cada dígito (0-9)
for label in range(10):
    os.makedirs(os.path.join(train_dir, str(label)), exist_ok=True)
    os.makedirs(os.path.join(test_dir, str(label)), exist_ok=True)

# 3. Guardar imágenes de entrenamiento
print("Guardando imágenes de entrenamiento...")
for i in range(len(x_train)):
    label = y_train[i]
    img = Image.fromarray(x_train[i], mode='L')
    img.save(os.path.join(train_dir, str(label), f"train_{i}.png"))

# 4. Guardar imágenes de prueba
print("Guardando imágenes de prueba...")
for i in range(len(x_test)):
    label = y_test[i]
    img = Image.fromarray(x_test[i], mode='L')
    img.save(os.path.join(test_dir, str(label), f"test_{i}.png"))

print("✅ ¡Listo! Todas las imágenes se guardaron en la carpeta 'mnist_dataset/'.")
print(f"   - Entrenamiento: {len(x_train)} imágenes")
print(f"   - Prueba: {len(x_test)} imágenes")
print(f"   - Total: {len(x_train) + len(x_test)} imágenes")