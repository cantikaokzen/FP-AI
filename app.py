from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('model_penyakit_daun_mangga.h5')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Preprocess image
    image = load_img(image_path, target_size=(256, 256))  # Sesuaikan ukuran input model
    image = img_to_array(image)
    image = image / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)

    # Flatten input if required by model
    if model.input_shape[-1] == 25088:  # Jika model mengharapkan input 1D
        image = image.reshape((1, -1))  # Ratakan tensor menjadi 1D

    # Predict
    yhat = model.predict(image)
    label_index = np.argmax(yhat, axis=1)[0]
    
    # Daftar kelas sesuai model
    class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
                    'Die Back', 'Gall Midge', 'Healthy', 
                    'Powdery Mildew', 'Sooty Mould']
    classification = f'{class_labels[label_index]} ({np.max(yhat) * 100:.2f}%)'

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=5500, debug=True)
