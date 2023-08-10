from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Create a numpy array from the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=f'Predicted class: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
