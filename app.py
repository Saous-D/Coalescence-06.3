from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import io
import base64
import Back.quantum_segmentation as quantum_segmentation
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os
import sys

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def qseg():
    return render_template('Front/qseg-choose-image.html')

@app.route('/extract_rgb', methods=['POST'])
def extract_rgb():
    image_path = request.form['image_path']
    if not image_path:
        return jsonify({'error': 'No image path provided'})

    try:
        # Appliquer la fonction extract_rgb_bands() sur l'image
        processed_matrix = quantum_segmentation.extract_rgb_bands(image_path)

        # Convertir la matrice en image base64
        buf = io.BytesIO()
        plt.imsave(buf, processed_matrix, format='png', cmap='viridis')
        buf.seek(0)
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({'image_data': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        original_matrix = quantum_segmentation.extract_rgb_bands(filepath)
        buf = io.BytesIO()
        plt.imsave(buf, original_matrix, format='png', cmap='viridis')
        buf.seek(0)
        img_bytes = buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({'image_path': filepath, 'image_data': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

def run_quantum_segmentation(image_path, image_height, image_width, water_penalty, no_water_penalty, sigma, mu, k):
    # Redirect stdout to a buffer
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        matrix = quantum_segmentation.super_segmentation(
            image_path, image_height, image_width, water_penalty,
            no_water_penalty, sigma, mu,k,
            callback=lambda msg: socketio.emit('messages', {'data': msg}))

        messages = buffer.getvalue()
        return matrix
    finally:
        sys.stdout = old_stdout

def convert(string, form):
    if request.form[string] != 'null':
        if form=='float':
            return float(request.form[string])
        else:
            return int(request.form[string])
    else:
        return None

@app.route('/process', methods=['POST'])
def process_image():
    image_path = request.form['image_path']
    if not image_path:
        return jsonify({'error': 'No image path provided'})

    image_height = int(request.form['image_height'])
    image_width = int(request.form['image_width'])
    water_penalty = float(request.form['water_penalty'])
    no_water_penalty = float(request.form['no_water_penalty'])
    sigma = float(request.form['sigma'])
    mu = int(request.form['mu'])
    k = int(request.form['k'])
    print("app.py : water_penalty = " +str(water_penalty))
    matrix = run_quantum_segmentation(
        image_path, image_height, image_width, water_penalty,
        no_water_penalty, sigma, mu, k)

    buf = io.BytesIO()
    plt.imsave(buf, matrix, format='png', cmap='viridis')
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return jsonify({'image_data': img_base64})

if __name__ == "__main__":
    socketio.run(app, debug=True)