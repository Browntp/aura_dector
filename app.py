from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import numpy as np
import pickle

app = Flask(__name__)

# Load the vectorizer configuration and weights
with open('vectorizer.pkl', 'rb') as f:
    vectorizer_config, vectorizer_vocab, vectorizer_weights = pickle.load(f)

# Recreate the TextVectorization layer from the config
vectorizer = TextVectorization.from_config(vectorizer_config)
vectorizer.set_vocabulary(vectorizer_vocab)
vectorizer.set_weights(vectorizer_weights)

# Load the pre-trained model
model = load_model('emotions_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Preprocess the text input as needed
    # Assuming input text is preprocessed properly
    input_text = vectorizer(np.array([text]))
    category = model.predict(input_text)
    return render_template('result.html', category=category)

if __name__ == '__main__':
    app.run(debug=True)
