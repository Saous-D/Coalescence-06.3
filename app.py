from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import io
import base64
import quantum_segmentation
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import os
import sys

app = Flask(__name__, template_folder='Front')
socketio = SocketIO(app, async_mode='threading')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/enter-code')
def enter_code():
    return render_template('enter-code.html')

@app.route('/insurance-detail.html')
def insurance_detail():
    return render_template('insurance-detail.html')

@app.route('/insurance')
def insurance():
    return render_template('insurance.html')

@app.route('/qapps')
def qapps():
    return render_template('qapps.html')

@app.route('/qmeds-details')
def qmeds_details():
    return render_template('qmeds-details.html')

@app.route('/qmeds-results')
def qmeds_results():
    return render_template('qmeds-results.html')

@app.route('/qmeds-run-screen')
def qmeds_run_screen():
    return render_template('qmeds-run-screen.html')

@app.route('/qmeds-upload')
def qmeds_upload():
    return render_template('qmeds-upload.html')

@app.route('/qseg-choose-image-selection')
def qseg_choose_image_selection():
    return render_template('qseg-choose-image-selection.html')

@app.route('/qseg-choose-image')
def qseg_choose_image():
    return render_template('qseg-choose-image.html')

@app.route('/qsegfire-choose-image')
def qsegfire_choose_image():
    return render_template('qsegfire-choose-image.html')

@app.route('/qsegwet-choose-image')
def qsegwet_choose_image():
    return render_template('qsegwet-choose-image.html')

@app.route('/qseg-details')
def qseg_details():
    return render_template('qseg-details.html')

@app.route('/qseg-runing-screen')
def qseg_runing_screen():
    return render_template('qseg-runing-screen.html')

@app.route('/qseg-settings')
def qseg_settings():
    return render_template('qseg-settings.html')

@app.route('/qsegfire-settings')
def qsegfire_settings():
    return render_template('qsegfire-settings.html')

@app.route('/qsegwet-settings')
def qsegwet_settings():
    return render_template('qsegwet-settings.html')

@app.route('/sign-in')
def sign_in():
    return render_template('sign-in.html')

@app.route('/sign-up')
def sign_up():
    return render_template('sign-up.html')

@app.route('/qseg-result')
def qseg_result():
    return render_template('qseg-result.html')

@app.route('/qsegfire-result')
def qsegfire_result():
    return render_template('qsegfire-result.html')

@app.route('/qsegwet-result')
def qsegwet_result():
    return render_template('qsegwet-result.html')

@app.route('/qsegchoose')
def qsegchoose():
    return render_template('qsegchoose.html')

@app.route('/qsegchooseseg')
def qsegchooseseg():
    return render_template('qsegchooseseg.html')

@app.route('/qsegAI-choose-image')
def qsegAI_choose_image():
    return render_template('qsegAI-choose-image.html')

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
    
@app.route('/heatmap', methods=['POST'])
def generate_heatmap():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        color = request.form.get('color', 'winter_r')

        heatmap_buf = quantum_segmentation.return_heatmap(filepath, color=color)

        img_bytes = heatmap_buf.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return jsonify({'image_data': img_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



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
    matrix = quantum_segmentation.quantum_scan(image_path, "water")

    print("app.py : water done ")

    matrix2 = quantum_segmentation.overlay_masks(matrix, quantum_segmentation.quantum_scan(image_path, "vegetation"))

    print("app.py : seg done ")


    buf = io.BytesIO()
    plt.imsave(buf, matrix, format='png', cmap='winter_r')
    buf.seek(0)
    img_bytes = buf.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')


    img_base642 = base64.b64encode(matrix2.read()).decode('utf-8')

    return jsonify({'image_water': img_base64, 'image_data': img_base642})

if __name__ == "__main__":
    socketio.run(app, debug=True)