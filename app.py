from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import io
import base64
import quantum_segmentation
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance
import os
import sys
from PIL import Image
from Image_forgery_detection.ela import convert_to_ela_image, convert_to_bn_image
from Image_forgery_detection.prediction import predict_result, find_forged_region

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


@app.route('/qsegai-result')
def qsegai_result():
    return render_template('qsegai-result.html')



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




#*************AI_Detection*************
@app.route('/convert_to_ela_image', methods=['POST'])
def conv_to_ela():
    try:
        data = request.get_json()

        if not data or 'image_base64' not in data:
            return jsonify({'error': 'Aucune image reçue.'}), 400
        
        print("test1")

        image_base64 = data['image_base64']
        # # Si la donnée est préfixée par data:..., on la découpe
        image_data = image_base64.split(',')[1] if ',' in image_base64 else image_base64

        # # Décoder l'image depuis base64
        image_bytes = io.BytesIO(base64.b64decode(image_data))
        image = Image.open(image_bytes).convert("RGB")

        # # Sauvegarder temporairement pour traitement
        temp_path = 'uploaded_image.jpg'
        image.save(temp_path, "JPEG")
        print("on envoie l'image à ela")

        # Appliquer ELA (supposons que cette fonction est déjà définie quelque part)
        scale, ela_image = convert_to_ela_image(temp_path, quality=95)


        show_image = ImageEnhance.Brightness(ela_image).enhance(40)
        show_image = ImageEnhance.Contrast(show_image).enhance(1.5)
        show_image = ImageEnhance.Sharpness(show_image).enhance(2.0)
        # Convertir l'image ELA en base64
        buffered = io.BytesIO()
        ela_image.save(buffered, format="PNG")
        encoded_ela = base64.b64encode(buffered.getvalue()).decode()
        buffered = io.BytesIO()
        show_image.save(buffered, format="PNG")
        encoded_show = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({'image_data': encoded_ela, 'show_image': encoded_show})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/analyze_ela', methods=['POST'])
def analyze_ela_image():
    try:
        # 1. Récupération de l'image envoyée en base64 (par exemple)
        data = request.get_json()
        image_base64 = data.get('image_base64')
        if not image_base64:
            return jsonify({'error': 'Aucune image transmise.'}), 400

        # 2. Décodage base64 vers image
        header, encoded = image_base64.split(',', 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 3. Sauvegarde temporaire de l’image
        temp_image_path = 'temp_uploaded_image.png'
        image.save(temp_image_path)

        # 4. Prédiction avec ELA
        n_pix_h, n_pix_v = 128,128
        test_image, test_image_d, scale, prediction, confidence = predict_result(temp_image_path, n_pix_h, n_pix_v)

        if prediction=="Falsifiée":
            segm_image=find_forged_region(temp_image_path, test_image_d, n_pix_h, n_pix_v)
            # 5. Conversion en N&B
            bn_image = convert_to_bn_image(segm_image, n_pix_h, n_pix_v)

        # 6. Encodage du résultat en base64 pour l’affichage dans le frontend
        buffered = io.BytesIO()
        bn_image.save(buffered, format="PNG")
        bn_image_base64 = base64.b64encode(buffered.getvalue()).decode()

        return jsonify({
            'success': True,
            'pixel_count': len(test_image_d.flatten()),
            'prediction':prediction,
            'confidence':confidence,
            'details': f"Prédiction : {prediction} (confiance : {float(confidence):.2f})",
            'bw_image': f"data:image/png;base64,{bn_image_base64}"
        })
    except Exception as e:
        
        import traceback
        traceback.print_exc()  # <-- Ceci affichera l’erreur dans la console
        return jsonify({'error': str(e)}), 500
        return jsonify({'error': str(e)}), 500



#******************************************************************
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Port assigné par Render ou 5000 par défaut
    socketio.run(app, host="0.0.0.0", port=port)