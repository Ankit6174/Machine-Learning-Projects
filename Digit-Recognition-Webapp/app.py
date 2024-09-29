from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import os
from ml_model.random_forest_classifier import prediction

app = Flask(__name__)

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        # Save the image to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Open the image using PIL
        img = Image.open(file_path)

        # Ensure the image is 28x28 pixels
        img = img.resize((28, 28))

        # Convert the image to grayscale
        img = img.convert('L')

        # Convert image to a NumPy array of pixel values
        img_array = np.array(img)

        # Print the pixel values in the terminal
        result = prediction(img_array.reshape(1, -1))

        # Optional: Return a success message
        return render_template('output.html', output=result)

if __name__ == '__main__':
    app.run(debug=True)

