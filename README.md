
# Project structure

    mnist-comparison/
    │
    ├── data/                          # Datos (opcional si usas datasets integrados)
    │   └── raw/                       # MNIST crudo (si lo descargas manualmente)
    │
    ├── models/                        # Código de los modelos
    │   ├── __init__.py
    │   ├── logistic_regression.py     # Clasificador clásico
    │   └── neural_network.py          # Red neuronal (puede ser con TensorFlow/PyTorch)
    │
    ├── notebooks/                     # Para experimentación rápida (opcional)
    │   └── exploratory_analysis.ipynb
    │
    ├── results/                       # Métricas, gráficos, logs
    │   ├── metrics_logistic.csv
    │   ├── metrics_nn.csv
    │   └── confusion_matrices/
    │
    ├── app/                           # Aplicación web (front + back ligero)
    │   ├── static/
    │   │   └── style.css
    │   ├── templates/
    │   │   └── index.html
    │   └── app.py                     # Backend ligero (Flask/FastAPI)
    │
    ├── utils/                         # Funciones auxiliares
    │   ├── data_loader.py
    │   └── evaluation.py
    │
    ├── train.py                       # Script principal para entrenar ambos modelos
    ├── requirements.txt               # Dependencias
    └── README.md                      # Instrucciones para ejecutar el proyecto