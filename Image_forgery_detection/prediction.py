import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Forçage CPU-only

import gc
import tensorflow as tf
import numpy as np
import sys
import psutil
from keras.models import load_model
from keras import backend as K
from Image_forgery_detection.ela import convert_to_ela_image
from scipy import stats

# Charge une seule fois le modèle
MODEL_PATH = "Image_forgery_detection/trained_model.h5"
model = load_model(MODEL_PATH)
print("✅ Modèle chargé une fois :", MODEL_PATH)

# Taille image optimisée
n_pix_h, n_pix_v = 64, 64
class_names = ["Falsified", "Authentic"]

def prepare_image(fname):
    image_size = (n_pix_h, n_pix_v)
    scale, ela_img = convert_to_ela_image(fname, 95)
    ela_image = np.array(ela_img.resize(image_size)).flatten() / scale
    return scale, ela_image

def predict_result(fname):
    print("📊 RAM avant prédiction:", psutil.virtual_memory().used / 1e6, "Mo")

    scale, test_image = prepare_image(fname)

    test_image_reshaped = test_image.reshape(-1, n_pix_h, n_pix_v, 3)
    y_pred = model.predict(test_image_reshaped)
    y_pred_class = round(y_pred[0][0])

    prediction = "Authentic" if class_names[y_pred_class] == "Authentic" else "Forged"
    confidence = f"{(1 - y_pred[0][0]) * 100:.2f}" if y_pred[0][0] <= 0.5 else f"{y_pred[0][0] * 100:.2f}"

    print("🔍 Résultat prédiction :", prediction, "| Confiance :", confidence, "%")

    # Nettoyage mémoire
    K.clear_session()
    gc.collect()
    print("🧹 RAM après nettoyage:", psutil.virtual_memory().used / 1e6, "Mo")

    return test_image_reshaped, test_image, scale, prediction, confidence
