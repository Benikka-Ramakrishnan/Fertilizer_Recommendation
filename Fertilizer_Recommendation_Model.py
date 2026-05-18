import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('Fertilizer_Recommendation.csv')

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# --- Encode categorical columns ---
soil_mapping = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}
df['Soil Type'] = df['Soil Type'].map(soil_mapping)

crop_mapping = {
    'Maize': 0, 'Sugarcane': 1, 'Cotton': 2, 'Tobacco': 3, 'Paddy': 4,
    'Barley': 5, 'Wheat': 6, 'Millets': 7, 'Oil seeds': 8, 'Pulses': 9, 'Ground Nuts': 10
}
df['Crop Type'] = df['Crop Type'].map(crop_mapping)

fertilizer_mapping = {
    'Urea': 1, 'DAP': 2, '14-35-14': 3, '28-28': 4,
    '17-17-17': 5, '20-20': 6, '10-26-26': 7
}
df['Fertilizer Name'] = df['Fertilizer Name'].map(fertilizer_mapping)

df.dropna(inplace=True)
print("After cleaning:", df.shape)

# --- Features and Target ---
X = df[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type',
        'Nitrogen', 'Phosphorous', 'Potassium']]
y = df['Fertilizer Name']

# --- Scaling ---
minmax = MinMaxScaler()
stand = StandardScaler()

X_minmax = minmax.fit_transform(X)
X_scaled = stand.fit_transform(X_minmax)

# --- Train on ALL data (small dataset) ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)
print(f"Model Accuracy: {accuracy_score(y, y_pred) * 100:.2f}%")

# --- Save artifacts ---
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(stand, open('standscaler.pkl', 'wb'))
pickle.dump(minmax, open('minmaxscaler.pkl', 'wb'))

print("Model and scalers saved successfully!")