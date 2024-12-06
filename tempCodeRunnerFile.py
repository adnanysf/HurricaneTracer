import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("storms.csv")

# Map status to numerical values
status_mapping = {
    'tropical depression': 0,
    'tropical storm': 1,
    'hurricane': 2,
    'extratropical': 3
}
df['status_num'] = df['status'].map(status_mapping)

# Select relevant features for modeling
features = ['lat', 'long', 'wind', 'pressure']
target = ['lat', 'long', 'wind', 'pressure', 'status_num']

# Check for outliers
print("Checking for outliers:")
print(df[features].describe())

# Remove outliers (optional step, depending on the data)
# Here, we remove rows where any feature is more than 3 standard deviations from the mean
df = df[(np.abs(df[features] - df[features].mean()) <= (3 * df[features].std())).all(axis=1)]

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :len(features)])
        y.append(data[i+seq_length, :len(features)])
    return np.array(X), np.array(y)

seq_length = 5
X, y = create_sequences(df[features + target].values, seq_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Debugging: Print ranges of input features before and after scaling
print("Original Data Ranges:")
print("Lat:", df['lat'].min(), "-", df['lat'].max())
print("Long:", df['long'].min(), "-", df['long'].max())
print("Wind:", df['wind'].min(), "-", df['wind'].max())
print("Pressure:", df['pressure'].min(), "-", df['pressure'].max())

print("Scaled Data Ranges:")
print("Lat:", X_train_scaled[:, :, 0].min(), "-", X_train_scaled[:, :, 0].max())
print("Long:", X_train_scaled[:, :, 1].min(), "-", X_train_scaled[:, :, 1].max())
print("Wind:", X_train_scaled[:, :, 2].min(), "-", X_train_scaled[:, :, 2].max())
print("Pressure:", X_train_scaled[:, :, 3].min(), "-", X_train_scaled[:, :, 3].max())

# Build the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    LSTM(32),
    Dense(y.shape[1])
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)

def predict_hurricane_path(start_lat, start_long, start_wind, start_pressure, steps=24):
    initial_sequence = np.array([[start_lat, start_long, start_wind, start_pressure]] * 5)
    initial_sequence_scaled = scaler.transform(initial_sequence)
    
    predictions = []
    current_sequence = initial_sequence_scaled
    
    for _ in range(steps):
        next_step = model.predict(current_sequence.reshape(1, 5, 4))
        predictions.append(next_step[0])
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_step[0][:4]
    
    predictions = np.array(predictions)
    features_predictions = predictions[:, :4]
    
    # Debugging: Print scaled predictions
    print("Scaled Predictions:\n", features_predictions)
    
    features_predictions = scaler.inverse_transform(features_predictions)
    
    # Debugging: Print inverse transformed predictions
    print("Inverse Transformed Predictions:\n", features_predictions)
    
    return pd.DataFrame(features_predictions, columns=['lat', 'long', 'wind', 'pressure'])

# Example usage
path = predict_hurricane_path(27.5, -79.0, 25, 1013)
print(path)