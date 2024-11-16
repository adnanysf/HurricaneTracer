import arcpy
from datetime import datetime, timedelta
import os
import arcpy.management
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Hurricane files
hurricane_lists = ["HarveyData.csv", "IdaData.csv", "IrmaData.csv", "KatrinaData.csv", "LauraData.csv", 
                   "MichaelData.csv", "RitaData.csv"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------------------------------>

def process_hurricane_data(input_folder, output_folder):
    # Checks if output folder exists, if not create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # ArcGIS workspace environment
    arcpy.env.workspace = output_folder

    # Loop through all CSV files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            
            # Load hurricane data
            hurricane_data = pd.read_csv(file_path)
            hurricane_data['Date'] = pd.to_datetime(hurricane_data['Date'])
            hurricane_data = hurricane_data.dropna(subset = ['LAT', 'LON'])

            print(f"Processing file: {file_name}")
            print(hurricane_data.head())

            # ------------------------------------------------------------------------------------------>
            # Prepare data for modeling
            start_date = hurricane_data['Date'].min()
            hurricane_data['DaysSinceStart'] = (hurricane_data['Date'] - start_date).dt.days

            # Features and targets
            X = hurricane_data[['DaysSinceStart']]
            y_latitude = hurricane_data['LAT']
            y_longitude = hurricane_data['LON']

            # Polynomial Features (2nd degree for simplicity)
            poly = PolynomialFeatures(degree = 2)
            X_poly = poly.fit_transform(X)

            # Linear Regression models for Latitude and Longitude
            lat_model = LinearRegression()
            lat_model.fit(X_poly, y_latitude)

            lon_model = LinearRegression()
            lon_model.fit(X_poly, y_longitude)

            # ------------------------------------------------------------------------------------------>
            # Predict future points (next 5 days)
            future_days = np.array([hurricane_data['DaysSinceStart'].max() + i for i in range(1, 6)]).reshape(-1, 1)
            future_days_poly = poly.transform(future_days)

            predicted_latitude = lat_model.predict(future_days_poly)
            predicted_longitude = lon_model.predict(future_days_poly)

            predictions = pd.DataFrame({
                'DaysAfterStart': future_days.flatten(),
                'PredictedLatitude': predicted_latitude,
                'PredictedLongitude': predicted_longitude
            })

            print(predictions)

            # ------------------------------------------------------------------------------------------>

            # Create feature class name from the CSV file name (excluding the file extension)
            feature_class_name = os.path.splitext(file_name)[0] + "Predicted"

            # Name validation for the file
            name = arcpy.ValidateFieldName(feature_class_name, r"C:\Users\Cesar\HurricaneTracer\HurricaneTracer\Hurricane_Tracer_Project\Hurricane_Tracer_Project.gdb")
            print("New validated name is: " + name)

            out_fc = os.path.join(output_folder, feature_class_name)

            # Delete the feature class if it already exists
            if arcpy.Exists(out_fc):
                arcpy.Delete_management(out_fc)
                print(f"Feature class {feature_class_name} deleted.")

            # Define spatial reference (WGS84)
            spatial_reference = arcpy.SpatialReference(4326)  # WGS84

            # Create a new feature class with Point geometry
            arcpy.management.CreateFeatureclass(output_folder, name, "POINT", spatial_reference = spatial_reference)

            # Insert predicted points into feature class
            with arcpy.da.InsertCursor(out_fc, ['SHAPE@XY']) as cursor:
                for i in range(len(predictions)):
                    point = (predictions.iloc[i]['PredictedLongitude'], predictions.iloc[i]['PredictedLatitude'])
                    cursor.insertRow([point])

            print(f"Prediction path added to feature class: {feature_class_name}")
            
    print("All files processed.")

# usage:
input_folder = r"C:\Users\Cesar\HurricaneTracer\HurricaneTracer\Hurricane_data"  # Folder containing CSV files
output_folder = r"C:\Users\Cesar\HurricaneTracer\HurricaneTracer\Hurricane_Tracer_Project\Hurricane_Tracer_Project.gdb"  # ArcGIS feature class output folder
process_hurricane_data(input_folder, output_folder)
