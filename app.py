from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
app.static_folder = 'static'

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = request.form.get('sepal_length', default=None, type=float)
    sepal_width = request.form.get('sepal_width', default=None, type=float)
    petal_length = request.form.get('petal_length', default=None, type=float)
    petal_width = request.form.get('petal_width', default=None, type=float)

    theme = request.form.get('theme', default='light')  # Default to light theme if not provided

    if any(value is None or np.isnan(value) for value in [sepal_length, sepal_width, petal_length, petal_width]):
        return render_template('index.html', prediction='Please provide valid values for all input fields.', theme=theme)

    # Create a numpy array from the input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=f'Predicted class: {prediction}', theme=theme)

if __name__ == '__main__':
    app.run(debug=True)