import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warnings (level 3 suppresses errors too)
from flask import Flask, render_template, request, redirect, url_for,session
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import joblib
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the tokenizer and LSTM model
tokenizer = pickle.load(open("tokenizer1.pkl", "rb"))
lstm_model = load_model("lstm_model.h5")
vectorizer = joblib.load('vectorizer.pkl')
rf_model = joblib.load('random_forest_model.pkl')
logistic_model = joblib.load('logistic_regression_model.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')

# Define maximum sequence length (same as used during training)
MAX_SEQUENCE_LENGTH = 100

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = None
    
    # Handle form submission (POST request)
    if request.method == 'POST':
        user_input = request.form['text']
        input_text = user_input  # Store the input text for re-rendering
        
        if user_input:
            # Preprocess the input text
            sequences = tokenizer.texts_to_sequences([user_input])
            padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
            
            # Predict using the LSTM model
            prediction_prob = lstm_model.predict(padded_sequences)[0][0]
            
            # Apply threshold for classification
            prediction = "Cyberbullying" if prediction_prob > 0.3 else "Not Cyberbullying"
            session['input_text'] = user_input
        else:
            prediction = None  # If no input is provided, no prediction

    return render_template('index.html', prediction=prediction, input_text=input_text)

@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('index'))

@app.route('/models', methods=['GET', 'POST'])
def models():
    # Retrieve input from session
    user_input = session.get('input_text', None)
    logistic_prediction = None
    nb_prediction = None
    rf_prediction = None

    if user_input:
        # Preprocess the input text for other models
        transformed_input = vectorizer.transform([user_input])

        # Logistic Regression prediction
        logistic_prediction = "Cyberbullying" if logistic_model.predict(transformed_input)[0] == 1 else "Not Cyberbullying"

        # Naive Bayes prediction
        nb_prediction = "Cyberbullying" if nb_model.predict(transformed_input)[0] == 1 else "Not Cyberbullying"

        # Random Forest prediction
        rf_prediction = "Cyberbullying" if rf_model.predict(transformed_input)[0] == 1 else "Not Cyberbullying"

    return render_template(
        'models.html',
        input_text=user_input,
        logistic_prediction=logistic_prediction,
        nb_prediction=nb_prediction,
        rf_prediction=rf_prediction
    )

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

if __name__ == '__main__':
    app.run(debug=True)
