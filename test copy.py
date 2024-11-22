import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('storms.csv')

columns_to_drop = ["Unnamed: 0", "name", "status", "category", "tropicalstorm_force_diameter", "hurricane_force_diameter"]
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

data = data.fillna(method='ffill')

data['month'] = data['month'].astype(int)
data['day'] = data['day'].astype(int)
data['hour'] = data['hour'].astype(int)

data['lat_lag1'] = data.groupby(['year', 'month', 'day', 'hour'])['lat'].shift(1)
data['long_lag1'] = data.groupby(['year', 'month', 'day', 'hour'])['long'].shift(1)
data['wind_lag1'] = data.groupby(['year', 'month', 'day', 'hour'])['wind'].shift(1)
data['pressure_lag1'] = data.groupby(['year', 'month', 'day', 'hour'])['pressure'].shift(1)

data = data.dropna()

features = ['lat_lag1', 'long_lag1', 'wind_lag1', 'pressure_lag1']
target_lat = 'lat'
target_long = 'long'

X = data[features]
y_lat = data[target_lat]
y_long = data[target_long]

X_train, X_test, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.2, random_state=42)
_, _, y_long_train, y_long_test = train_test_split(X, y_long, test_size=0.2, random_state=42)

lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
lat_model.fit(X_train, y_lat_train)

long_model = RandomForestRegressor(n_estimators=100, random_state=42)
long_model.fit(X_train, y_long_train)

y_lat_pred = lat_model.predict(X_test)
y_long_pred = long_model.predict(X_test)
lat_mae = mean_absolute_error(y_lat_test, y_lat_pred)
long_mae = mean_absolute_error(y_long_test, y_long_pred)

print(f"Latitude MAE: {lat_mae}")
print(f"Longitude MAE: {long_mae}")

def predict_next_position(lat, long, wind, pressure):
    input_features = np.array([[lat, long, wind, pressure]])
    next_lat = lat_model.predict(input_features)[0]
    next_long = long_model.predict(input_features)[0]
    return next_lat, next_long

def predict_hurricane_path(start_lat, start_long, start_wind, start_pressure, max_steps=100, wind_threshold=20, lat_long_change_limit=1.0):
    lat, long, wind, pressure = start_lat, start_long, start_wind, start_pressure
    path = [(lat, long, wind, pressure)]
    
    for _ in range(max_steps):
        next_lat, next_long = predict_next_position(lat, long, wind, pressure)
        
        lat_change = np.clip(next_lat - lat, -lat_long_change_limit, lat_long_change_limit)
        long_change = np.clip(next_long - long, -lat_long_change_limit, lat_long_change_limit)
        
        lat += lat_change
        long += long_change
        path.append((lat, long, wind, pressure))
        
        wind = max(wind - 2, 0) 
        pressure += 1          
        
        if wind < wind_threshold:
            break
            
    return path

starting_lat = 27.5
starting_long = -79.0
starting_wind = 120
starting_pressure = 1013

path = predict_hurricane_path(starting_lat, starting_long, starting_wind, starting_pressure)

for i, (lat, long, wind, pressure) in enumerate(path):
    print(f"Step {i}: Latitude {lat}, Longitude {long}, Wind {wind}, Pressure {pressure}")
