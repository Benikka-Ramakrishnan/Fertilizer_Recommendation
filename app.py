from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scalers
model = pickle.load(open('model.pkl', 'rb'))
standscaler = pickle.load(open('standscaler.pkl', 'rb'))
minmaxscaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Moisture = float(request.form['Moisture'])
        Soil_Type = int(request.form['Soil_Type'])
        Crop_Type = int(request.form['Crop_Type'])
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorous = float(request.form['Phosphorous'])
        Potassium = float(request.form['Potassium'])

        feature_list = [Temperature, Humidity, Moisture, Soil_Type, Crop_Type,
                        Nitrogen, Phosphorous, Potassium]
        single_pred = np.array(feature_list).reshape(1, -1)

        scaled_features = minmaxscaler.transform(single_pred)
        final_features = standscaler.transform(scaled_features)
        prediction = model.predict(final_features)

        fertilizer_dict = {
            1: "Urea",
            2: "DAP",
            3: "14-35-14",
            4: "28-28",
            5: "17-17-17",
            6: "20-20",
            7: "10-26-26"
        }

        fert_name = fertilizer_dict.get(int(prediction[0]), "Unknown")

        result = f"The Recommended Fertilizer is: {fert_name}"
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
