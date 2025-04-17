from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # For loading the model

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')  # Save your model as a .pkl file
scaler = joblib.load('scaler.pkl')  # Save your scaler as a .pkl file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = {
        'Station_Names': request.form['station_names'],
        'Year': int(request.form['year']),
        'Month': int(request.form['month']),
        'Max_Temp': float(request.form['max_temp']),
        'Min_Temp': float(request.form['min_temp']),
        'Rainfall': float(request.form['rainfall']),
        'Relative_Humidity': float(request.form['relative_humidity']),
        'Wind_Speed': float(request.form['wind_speed']),
        'Cloud_Coverage': float(request.form['cloud_coverage']),
        'Bright_Sunshine': float(request.form['bright_sunshine']),
        'Station_Number': int(request.form['station_number']),
        'X_COR': float(request.form['x_cor']),
        'Y_COR': float(request.form['y_cor']),
        'LATITUDE': float(request.form['latitude']),
        'LONGITUDE': float(request.form['longitude']),
        'ALT': float(request.form['alt']),
        'Period': request.form['period']
    }

    # Convert to DataFrame
    new_df = pd.DataFrame([data])

    # Preprocess the input data (similar to your notebook)
    new_df['Station_Names'] = pd.Categorical(new_df['Station_Names']).codes
    new_df['Period'] = pd.Categorical(new_df['Period']).codes

    # Scale features
    X_new = new_df.values
    X_new = scaler.transform(X_new)

    # Make prediction
    prediction = model.predict(X_new)
    probability = model.predict_proba(X_new)[:, 1]

    result = {
        'prediction': 'Flood' if prediction[0] == 1 else 'No Flood',
        'probability': f"{probability[0]:.1%}"
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)