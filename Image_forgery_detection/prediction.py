import os
import gc
import time
import sys
import numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf
from scipy import stats
from Image_forgery_detection.ela import convert_to_ela_image
from dwave.system import LeapHybridSampler
from dotenv import load_dotenv

# Désactiver GPU (important pour Render)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Charger les variables d'environnement (dont PRIVATE_TOKEN)
load_dotenv()
private_token = os.getenv("PRIVATE_TOKEN")

# Chargement du modèle une seule fois
global_model = None
def load_global_model():
    global global_model
    if global_model is None:
        print("Chargement du modèle Keras...")
        global_model = load_model("Image_forgery_detection/trained_model.h5")
    return global_model

# Préparation de l’image (taille réduite à 64×64)
def prepare_image(fname, n_pix_h=64, n_pix_v=64):
    image_size = (n_pix_h, n_pix_v)
    scale, ela_img = convert_to_ela_image(fname, 95)
    ela_image = np.array(ela_img.resize(image_size)).flatten() / scale
    return scale, ela_image

# Prédiction de l’authenticité d’une image
def predict_result(fname, n_pix_h=64, n_pix_v=64):
    model = load_global_model()
    class_names = ["Falsified", "Authentic"]
    scale, test_image = prepare_image(fname, n_pix_h, n_pix_v)

    test_image_input = test_image.reshape(-1, n_pix_h, n_pix_v, 3)
    print(f"Taille test_image (RAM): {sys.getsizeof(test_image_input)/1024:.2f} KB")

    y_pred = model.predict(test_image_input)
    y_pred_class = round(y_pred[0][0])
    prediction = "Authentic" if class_names[y_pred_class] == "Authentic" else "Forged"
    confidence = f"{(1 - y_pred[0][0]) * 100:.2f}" if y_pred <= 0.5 else f"{y_pred[0][0] * 100:.2f}"

    return test_image_input, test_image, scale, prediction, confidence

# Traitement de la segmentation sur le canal rouge uniquement
def find_forged_region(fname, test_image, n_pix_h=64, n_pix_v=64):
    n_data = len(test_image)
    n_pixs = int(n_data / 3)

    Red_pixs = np.array([test_image[i] for i in range(0, n_data, 3)], dtype=np.float64)
    Delta_e = np.unique(test_image)[1]
    e_mod_R = stats.mode(Red_pixs, keepdims=True)[0][0]
    Red_pixs = (-Red_pixs + e_mod_R) / Delta_e
    Red_pixs_mat = Red_pixs.reshape(n_pix_h, n_pix_v, order='C')

    np_bh, np_bv = 32, 32
    nbh, nbv = n_pix_h // np_bh, n_pix_v // np_bv
    segm_image_R = np.zeros((n_pix_h, n_pix_v), dtype=np.int64)
    segm_image_R = fill_segm_image("Rouge", Red_pixs_mat, segm_image_R, np_bh, np_bv, nbh, nbv)

    segm_image = np.zeros((n_pixs), dtype=np.int64)
    for i in range(n_pix_h):
        for j in range(n_pix_v):
            if segm_image_R[i][j] == 1:
                segm_image[i * n_pix_v + j] = 1

    print("Segmentation terminée.")
    return segm_image

# Construction de la QUBO
def calc_QUBO_matrix(pixs_mat, x_opt, i_b, j_b, np_bh, np_bv):
    nq = np_bh * np_bv
    Q = np.zeros((nq, nq), dtype=np.float64)
    for i in range(np_bh):
        for j in range(np_bv):
            Q[i*np_bv + j][i*np_bv + j] = np.sign(pixs_mat[i_b*np_bh+i][j_b*np_bv+j]) * (pixs_mat[i_b*np_bh+i][j_b*np_bv+j])**2
    Q_dict = {(i, j): Q[i][j] for i in range(nq) for j in range(i+1)}
    return list(Q_dict.keys()), Q_dict

# Résolution du QUBO avec D-Wave LeapHybrid
def dwave_solve(Q_dict, Q_inds, n_pixs_b):
    print("Résolution du QUBO via D-Wave LeapHybridSampler...")
    sampler = LeapHybridSampler(token=private_token)
    sampleset = sampler.sample_qubo(Q_dict, label="ai_anomalie")
    solution = np.array([sampleset.first.sample[i] for i in range(n_pixs_b)])
    return solution

# Remplissage de l’image segmentée (canal Rouge uniquement ici)
def fill_segm_image(Color, col_pixs_mat, segm_image_col, np_bh, np_bv, nbh, nbv):
    x_opt = np.zeros((np_bh * np_bv), dtype=np.int64)
    for i_b in range(nbh):
        for j_b in range(nbv):
            Q_inds, Q_dict = calc_QUBO_matrix(col_pixs_mat, x_opt, i_b, j_b, np_bh, np_bv)
            x_opt = dwave_solve(Q_dict, Q_inds, np_bh * np_bv)
            for i in range(i_b * np_bh, (i_b + 1) * np_bh):
                for j in range(j_b * np_bv, (j_b + 1) * np_bv):
                    segm_image_col[i][j] = x_opt[(i - i_b*np_bh)*np_bv + (j - j_b*np_bv)]
    return segm_image_col

# Nettoyage mémoire manuel (à appeler à la fin)
def clean_memory():
    global global_model
    del global_model
    global_model = None
    K.clear_session()
    gc.collect()
    print(" Mémoire libérée.")

