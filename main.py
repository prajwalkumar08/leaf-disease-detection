from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Initialize Flask app
app = Flask(__name__)

# Load the models
CNN_MODEL_PATH = "models/model_lenet.h5"
INCEPTION_MODEL_PATH = "models/model_inception.h5"
DENSENET_MODEL_PATH = "models/densenet121_fine_tuned.h5"

cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
inception_model = tf.keras.models.load_model(INCEPTION_MODEL_PATH)
densenet_model = tf.keras.models.load_model(DENSENET_MODEL_PATH)

# Define class labels
class_labels = [
    "Algal_Leaf_Spot_of_Jackfruit", "Anthracnose mango", "Bacterial Canker Mango",
    "Black_Spot_of_Jackfruit", "CCI_Caterpillars Coconut", "CCI_Leaflets Coconut",
    "Cordana Banana", "Cutting Weevil Mango", "Die Back Mango", "Gall Midge Mango",
    "Healthy banana", "Healthy mango", "Healthy_Leaf_of_Jackfruit",
    "Panama Disease Banana", "Powdery Mildew Mango", "Sooty Mould Mango",
    "WCLWD_DryingofLeaflets", "WCLWD_Flaccidity Coconut", "WCLWD_Yellowing coconut",
    "Yellow and Black Sigatoka Banana", "Anthracnose Cashew ", "Bacterial Blight cassava",
    "Brown Spot cassava", "Green Mite cassava", "Gumosis Cashew", "Healthy Cashew",
    "Healthy Cassava", "Leaf Miner Cashew", "Mosaic cassava", "Red Rust cashew"
]

# Prediction route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Validate input
        if 'file' not in request.files:
            return jsonify({'error': 'File not provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        try:
            # Preprocess the image for each model
            cnn_img = load_img(file_path, target_size=(224, 224))
            inception_img = load_img(file_path, target_size=(128, 128))
            densenet_img = load_img(file_path, target_size=(224, 224))

            cnn_img_array = np.expand_dims(img_to_array(cnn_img) / 255.0, axis=0)
            inception_img_array = np.expand_dims(img_to_array(inception_img) / 255.0, axis=0)
            densenet_img_array = np.expand_dims(img_to_array(densenet_img) / 255.0, axis=0)

            # Get predictions from each model
            cnn_predictions = cnn_model.predict(cnn_img_array)
            inception_predictions = inception_model.predict(inception_img_array)
            densenet_predictions = densenet_model.predict(densenet_img_array)

            # Extract class and confidence for each model
            cnn_pred_class = class_labels[np.argmax(cnn_predictions)]
            cnn_confidence = np.max(cnn_predictions)

            inception_pred_class = class_labels[np.argmax(inception_predictions)]
            inception_confidence = np.max(inception_predictions)

            densenet_pred_class = class_labels[np.argmax(densenet_predictions)]
            densenet_confidence = np.max(densenet_predictions)

            # Cleanup and return results
            os.remove(file_path)

            return render_template('results.html',
                                   cnn_pred_class=cnn_pred_class, cnn_confidence=round(float(cnn_confidence), 2),
                                   inception_pred_class=inception_pred_class, inception_confidence=round(float(inception_confidence), 2),
                                   densenet_pred_class=densenet_pred_class, densenet_confidence=round(float(densenet_confidence), 2))

        except Exception as e:
            os.remove(file_path)
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')


if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
