from flask import Flask, request, render_template, redirect, url_for
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

# List to store all predictions
predictions = []

@app.route('/')
def home():
    total = sum(predictions)
    return render_template('index.html', predictions=predictions, total=total)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Preprocess the text input as needed
    # Assuming input text is preprocessed properly
    input_text = vectorizer(np.array([text]))
    array = model.predict(input_text)
    aura = converter(array)
    predictions.append(aura)
    return redirect(url_for('home'))


def converter(array):
    """
    # array(['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame'],
      dtype=object)
    """
   
    anger = array[0][0]
    disgust = array[0][1]
    fear = array[0][2]
    guilt = array[0][3]
    joy = array[0][4]
    sadness = array[0][5]
    shame = array[0][6]

    return (joy + sadness + disgust - (anger + fear + guilt + shame)) * 1000 
    #return f"{anger} {disgust} {fear} {guilt} {joy} {sadness} {shame}"

if __name__ == '__main__':
    app.run(debug=True)
