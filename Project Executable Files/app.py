from flask import Flask, request, render_template, url_for, redirect
import numpy as np
import pickle
import os

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'fetal_health_model.pkl')
model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        prolongued_decelerations = float(request.form['prolongued_decelerations'])
        abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
        percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
        histogram_variance = float(request.form['histogram_variance'])
        histogram_median = float(request.form['histogram_median'])
        mean_value_of_long_term_variability = float(request.form['mean_value_of_long_term_variability'])
        histogram_mode = float(request.form['histogram_mode'])
        accelerations = float(request.form['accelerations'])

        # Prepare the input for prediction
        X = [[prolongued_decelerations, abnormal_short_term_variability, percentage_of_time_with_abnormal_long_term_variability,
              histogram_variance, histogram_median, mean_value_of_long_term_variability, histogram_mode, accelerations]]

        # Make a prediction
        output = model.predict(X)[0]

        # Map the output to a human-readable format
        if int(output) == 0:
            prediction = 'Normal'
        elif int(output) == 1:
            prediction = 'Pathological'
        else:
            prediction = 'Suspect'

        return render_template('output.html', output=prediction)
    return render_template('predict.html')

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    name = request.form['name']
    email = request.form['email']
    subject = request.form['subject']
    message = request.form['message']
    
    # Here you can add code to send the message, save it to a database, etc.
    print(f"Message from {name} ({email}): {subject} - {message}")
    
    return redirect(url_for('contact'))

if __name__ == "__main__":
    app.run(debug=True)

