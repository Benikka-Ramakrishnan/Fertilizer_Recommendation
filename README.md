# Fertilizer_Recommendation

A machine learning web application built with Flask that recommends the most suitable fertilizer based on soil conditions, crop type, and nutrient levels.

## 📁 Project Structure

```
fertilizer_recommendation/
│
├── static/               # CSS, JS assets
├── templates/
│   └── index.html        # Frontend UI
│
├── Fertilizer_Recommendation.csv       # Dataset
├── Fertilizer_Recommendation_Model.py  # Model training script
├── app.py                # Flask web application
├── model.pkl             # Trained ML model
├── minmaxscaler.pkl      # MinMaxScaler
├── standscaler.pkl       # StandardScaler
└── README.md
```

## 🌱 Features

- Predicts one of 7 fertilizers based on input parameters
- Inputs: Temperature, Humidity, Moisture, Soil Type, Crop Type, N, P, K levels
- Random Forest classifier with MinMax + Standard scaling pipeline

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy
```

### 2. Train the model (first time only)
```bash
python Fertilizer_Recommendation_Model.py
```
This generates `model.pkl`, `minmaxscaler.pkl`, and `standscaler.pkl`.

### 3. Run the Flask app
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## 📊 Dataset

The dataset (`Fertilizer_Recommendation.csv`) contains the following columns:
- `Temparature` — Soil/ambient temperature (°C)
- `Humidity ` — Air humidity (%)
- `Moisture` — Soil moisture (%)
- `Soil Type` — Sandy, Loamy, Black, Red, Clayey
- `Crop Type` — Maize, Sugarcane, Cotton, Tobacco, Paddy, Barley, Wheat, Millets, Oil seeds, Pulses, Ground Nuts
- `Nitrogen`, `Phosphorous`, `Potassium` — Nutrient levels (kg/ha)
- `Fertilizer Name` — Target label

## 🌿 Fertilizer Classes

| Code | Fertilizer   |
|------|-------------|
| 1    | Urea        |
| 2    | DAP         |
| 3    | 14-35-14    |
| 4    | 28-28       |
| 5    | 17-17-17    |
| 6    | 20-20       |
| 7    | 10-26-26    |

## 🛠 Tech Stack
- **Backend**: Python, Flask
- **ML**: scikit-learn (Random Forest)
- **Frontend**: HTML, CSS
- **Data**: pandas, numpy
