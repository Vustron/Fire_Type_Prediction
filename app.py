# type: ignore
# Importing dependencies
from flask import Flask, request, jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import joblib

# init app
app = Flask(__name__)

# Load the trained model and other necessary objects
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le_location = joblib.load('le_location.pkl')
le_type = joblib.load('le_type.pkl')
le_occupancy = joblib.load('le_occupancy.pkl')

# Helper function to convert month name to integer
def month_to_int(month):
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    return month_mapping.get(month, 1) 

# Helper function to check if location is valid
def is_valid_location(location):
    return location in VALID_LOCATIONS

# List of valid locations
VALID_LOCATIONS = [
    'Anibongan', 'Anislagan', 'Binuangan', 'Bucana', 'Calabcab', 'Concepcion', 
    'Dumlan', 'Elizalde (Somil)', 'Pangi (Gaudencio Antonio)', 'Gubatan', 'Hijo', 
    'Kinuban', 'Langgam', 'Lapu-lapu', 'Libay-libay', 'Limbo', 'Lumatab', 'Magangit', 
    'Malamodao', 'Manipongol', 'Mapaang', 'Masara', 'New Asturias', 'Panibasan', 
    'Panoraon', 'Poblacion', 'San Juan', 'San Roque', 'Sangab', 'Taglawig', 
    'Mainit', 'New Barili', 'New Leyte', 'New Visayas', 'Panangan', 'Tagbaros', 
    'Teresa'
]

# Helper function to check if occupancy is valid
def is_valid_occupancy(occupancy):
    return occupancy in VALID_OCCUPANCIES

VALID_OCCUPANCIES = ['Residential', 'Commercial', 'Industrial', 'Mixed']

# Helper function to convert Day and Hour from string to integer
def convert_to_int(data, key, default=0):
    try:
        return int(data.get(key, default))
    except ValueError:
        return default

# ---------------------- Index API route ----------------------
@app.route('/')
@cross_origin()
def index():
      return jsonify({"success": "Index"}), 200

# ---------------------- Predict API route ----------------------
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
   try:
      # Get the JSON data from the request
      data = request.json
      
      # Extract location and preprocess it
      location = data.get('Location', '')
      if not is_valid_location(location):
          return jsonify({"error": "Invalid location"}), 400
    
      location_encoded = le_location.transform([location])[0]
      
       # Extract month and preprocess it
      month = data.get('Month', 'January')
      month_int = month_to_int(month)
      
       # Extract occupancy and preprocess it
      occupancy = data.get('Occupancy', '')
      if not is_valid_occupancy(occupancy):
          return jsonify({"error": "Invalid occupancy"}), 400
      
      # Convert Day and Hour to integers
      day_int = convert_to_int(data, 'Day', 1)
      hour_int = convert_to_int(data, 'Hour', 0)

      # Create a dataframe for the input data
      input_data = pd.DataFrame({
            'Location': [location_encoded],
            'Month': [month_int],  
            'Day': [day_int],      
            'Hour': [hour_int], 
            'Occupancy': [le_occupancy.transform([occupancy])[0]]
      })
      
      # Normalize the input data
      input_data = scaler.transform(input_data)
      
       # Make prediction probabilities
      probabilities = model.predict_proba(input_data)[0]
      fire_types = le_type.inverse_transform(np.arange(len(probabilities)))
        
      # Combine fire types with their probabilities
      result = {fire_type: prob for fire_type, prob in zip(fire_types, probabilities)}
    
      # Return the prediction probabilities as JSON
      return jsonify(result)

   except Exception as e:
      app.logger.error(f"An error occurred: {str(e)}")
      return jsonify({"error": "Internal Server Error"}), 500

# ---------------------- End of API routes ----------------------
if __name__ == '__main__':
   app.run(debug = True)
