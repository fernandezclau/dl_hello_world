# backend/utils.py
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def get_evaluation_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1-Score": float(f1)
    }


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return plot_to_base64(fig)


def plot_comparison_chart(metrics_dl, metrics_lr):
    labels = list(metrics_dl.keys())
    dl_vals = [metrics_dl[k] for k in labels]
    lr_vals = [metrics_lr[k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, dl_vals, width, label='Deep Learning', color='#3498db')
    ax.bar(x + width / 2, lr_vals, width, label='Logistic Regression', color='#2ecc71')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1)
    return plot_to_base64(fig)


def plot_per_class_metrics(report_dict, title="Per-Class Metrics"):
    classes = list(report_dict.keys())[:-1]  # omitir 'accuracy', 'macro avg', etc.
    if 'accuracy' in classes:
        classes = [str(i) for i in range(10)]

    precisions = [report_dict[c]['precision'] for c in classes]
    recalls = [report_dict[c]['recall'] for c in classes]
    f1s = [report_dict[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label='Precision', color='#3498db')
    ax.bar(x, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, f1s, width, label='F1-Score', color='#2ecc71')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    return plot_to_base64(fig)


def process_drawing(image_data_url):
    import re
    from PIL import Image
    import io
    import base64

    # Extraer base64
    header, encoded = image_data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Convertir a escala de grises y fondo blanco
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
    gray = background.convert("L")

    # Redimensionar a 28x28
    resized = gray.resize((28, 28), Image.LANCZOS)

    # Invertir y normalizar
    img_array = np.array(resized)
    img_array = 255 - img_array  # blanco -> negro, negro -> blanco
    img_array = img_array.astype("float32") / 255.0
    img_flat = img_array.reshape(1, -1)  # (1, 784)

    # Para mostrar la imagen procesada
    processed_img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L')
    buf = BytesIO()
    processed_img.save(buf, format='PNG')
    processed_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return img_flat, processed_b64