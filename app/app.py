# backend/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import (
    get_evaluation_metrics,
    plot_confusion_matrix,
    plot_comparison_chart,
    plot_per_class_metrics,
    process_drawing
)
from sklearn.metrics import classification_report

app = Flask(__name__, static_folder='../app', static_url_path='')
CORS(app)

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")

# Cargar datos y modelos una vez al iniciar
print("Cargando modelos y datos...")
y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))
lr_pred = np.load(os.path.join(RESULTS_DIR, "lr_predictions.npy"))
nn_pred = np.load(os.path.join(RESULTS_DIR, "nn_predictions.npy"))

lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
nn_model = load_model(os.path.join(MODELS_DIR, "neural_network.h5"))

# Reportes (solo texto, no gráficos)
report_dl = classification_report(y_test, nn_pred, output_dict=True)
report_lr = classification_report(y_test, lr_pred, output_dict=True)

@app.route('/')
def index():
    return send_from_directory('../app', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../app', path)

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        # Métricas
        metrics_dl = get_evaluation_metrics(y_test, nn_pred)
        metrics_lr = get_evaluation_metrics(y_test, lr_pred)

        # Gráficos
        cm_dl_b64 = plot_confusion_matrix(y_test, nn_pred, "Deep Learning")
        cm_lr_b64 = plot_confusion_matrix(y_test, lr_pred, "Logistic Regression")
        comparison_b64 = plot_comparison_chart(metrics_dl, metrics_lr)
        per_class_dl_b64 = plot_per_class_metrics(report_dl, "Deep Learning")
        per_class_lr_b64 = plot_per_class_metrics(report_lr, "Logistic Regression")

        return jsonify({
            "success": True,
            "metrics_dl": metrics_dl,
            "metrics_lr": metrics_lr,
            "confusion_matrix_dl": cm_dl_b64,
            "confusion_matrix_lr": cm_lr_b64,
            "comparison_chart": comparison_b64,
            "per_class_dl": per_class_dl_b64,
            "per_class_lr": per_class_lr_b64,
            "report_dl": report_dl,
            "report_lr": report_lr
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data_url = data['image']

        # Procesar dibujo
        img_flat, processed_b64 = process_drawing(image_data_url)

        # Predicciones
        lr_proba = lr_model.predict_proba(img_flat)[0]
        lr_pred_class = int(lr_model.predict(img_flat)[0])
        lr_conf = float(lr_proba[lr_pred_class])

        nn_proba = nn_model.predict(img_flat)[0]
        nn_pred_class = int(np.argmax(nn_proba))
        nn_conf = float(nn_proba[nn_pred_class])

        return jsonify({
            "success": True,
            "processed_image": processed_b64,
            "prediction_dl": {
                "class": nn_pred_class,
                "confidence": nn_conf
            },
            "prediction_lr": {
                "class": lr_pred_class,
                "confidence": lr_conf
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)