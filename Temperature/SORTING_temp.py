import pandas as pd

# Step 1: Load the large temperature dataset (water_temperature.xlsx)
df = pd.read_excel('/Users/aubreyanderson/Desktop/geog392/PROJECT/water_temperature.xlsx')

# Convert the 'Date' column in the temperature data to datetime format (in case it's not already)
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Load the filtered hurricane dataset (filtered_hurricane_temperature_data.csv)
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/filtered_hurricane_data.csv')

# Convert the 'Date' column in the hurricane data to datetime format
hurricane_df['Date'] = pd.to_datetime(hurricane_df['Date'])

# Step 3: Extract relevant dates from the hurricane dataset
relevant_dates = hurricane_df['Date'].unique()  # Get the unique dates of hurricanes

# Step 4: Filter the temperature data to only include rows that have relevant dates
filtered_temperature_data = df[df['Date'].isin(relevant_dates)]

# Step 5: Save the filtered temperature data to a new CSV file
filtered_temperature_data.to_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/filtered_water_temperature_data.csv', index=False)

print("Filtered temperature data saved to 'filtered_temperature_data.csv'.")
