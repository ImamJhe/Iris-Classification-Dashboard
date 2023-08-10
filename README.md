# Iris Classification Dashboard

This is a simple Flask web application that allows users to input sepal and petal measurements for an iris flower and get a prediction of its class using a pre-trained SVM model.

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone https://github.com/ImamJhe/Iris-Classification-Dashboard.git
```

2. Install the required packages:

```bash
pip install flask numpy scikit-learn
```

3. Run the Flask app:
```bash
cd Iris-Classification-Dashboard
```
```bash
python app.py
```

The app will be accessible at `http://127.0.0.1:5000/` in your web browser.

## Features

- User-friendly interface to input iris flower measurements.
- Prediction of iris flower class using a pre-trained Support Vector Machine (SVM) model.
- Theme toggle switch to switch between light and dark modes for the dashboard.

## Project Structure

```
├── app.py            # Flask app code
├── model.pkl         # Pre-trained SVM model
├── static            # Static assets (CSS)
│   └── styles.css
└── templates         # HTML templates
    └── index.html
```
