# Digit Recognition Web App

This project is a simple web application that uses machine learning to recognize handwritten digits from uploaded images.

## Description

The application allows users to upload 28x28 pixel images of handwritten digits. It then uses a trained machine learning model to predict which digit (0-9) is represented in the image.

## Features

- Web interface for image upload
- Support for 28x28 pixel images
- Machine learning model for digit recognition
- Instant prediction results

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/digit-recognition-webapp.git
   ```
2. Navigate to the project directory:
   ```
   cd digit-recognition-webapp
   ```
3. Install the required dependencies:
   ```
   pip install -r flask numpy pandas PIL sklearn
   ```

## Usage

1. Start the web server:
   ```
   python app.py
   ```
2. Open a web browser and go to `http://localhost:5000`
3. Upload a 28x28 pixel image of a handwritten digit
4. Click "Predict" to see the model's prediction

## Technologies Used

- Python
- Flask (for web server)
- Sklearn (for machine learning model)
- HTML/CSS (for frontend)

## Future Improvements

- Add support for drawing digits directly in the browser
- Improve model accuracy with more training data
- Add explanation of the prediction process