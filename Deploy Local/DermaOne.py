from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model_path = 'DermaOne_Model.h5'
model = load_model(model_path)

# Class map
class_map = {
    0: 'BBC (Sel Basal)',
    1: 'Dermatitis Atopik',
    2: 'Eczema (Eksim)',
    3: 'Melanoma',
    4: 'Tahi Lalat (Nevus Melanositik)',
    5: 'Kutil Tua (Keratosis Seboroik)',
    6: 'Kutil Molluscum',
    7: 'Lesi Keratosis Jinak', 
    8: 'Psoriasis',
    9: 'Ringworm (Kurap infeksi Jamur)'
}

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')
        
        file = request.files['file']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', prediction='No selected file')
        
        # Save the uploaded file
        upload_folder = 'static/uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)
        
        # Preprocess the image
        img = preprocess_image(file_path)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        predicted_label = class_map[predicted_class]
        
        return render_template('index.html', prediction=predicted_label, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
