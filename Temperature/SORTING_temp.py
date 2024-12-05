import pandas as pd

# Step 1: Load the large temperature dataset (water_temperature.xlsx)
df = pd.read_excel('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/water_temperature.xlsx')

# Convert the 'Date' column in the temperature data to datetime format (in case it's not already)
df['Date'] = pd.to_datetime(df['Date']).dt.date  # Remove time information for matching

# Step 2: Load the hurricane dataset (filtered_hurricane_temperature_data.csv)
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_hurricane_data.csv')

# Convert the 'Date' column in the hurricane data to datetime format (in case it's not already)
hurricane_df['Date'] = pd.to_datetime(hurricane_df['Date']).dt.date  # Remove time information

# Step 3: Initialize an empty list to collect valid dates
valid_dates = []

# Step 4: Loop through each hurricane and create the 15-day window before, during, and after the hurricane
for _, hurricane in hurricane_df.iterrows():
    hurricane_date = hurricane['Date']
    
    # Define the 15-day window before, during, and after the hurricane
    start_date = hurricane_date - pd.Timedelta(days=15)
    end_date = hurricane_date + pd.Timedelta(days=15)
    
    # Add dates within the range of this hurricane to the valid_dates list
    valid_dates.extend(pd.date_range(start=start_date, end=end_date).date)  # Get the range of dates as a list of dates

# Step 5: Filter the temperature data to only include rows with dates within the valid date range
filtered_temperature_data = df[df['Date'].isin(valid_dates)]

# Step 6: Save the filtered temperature data to a new CSV file
filtered_temperature_data.to_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_water_temperature_data.csv', index=False)

print("Filtered temperature data saved to 'filtered_water_temperature_data.csv'.")