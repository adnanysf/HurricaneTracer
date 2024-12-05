import pandas as pd
import matplotlib.pyplot as plt

# Load the temperature and hurricane data
sst_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_water_temperature_data.csv')
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_hurricane_data.csv')

# Merge the hurricane data with SST data on the Date column
merged_df = pd.merge(hurricane_df, sst_df, on='Date', how='inner')

# Define the SST categories
def categorize_sst(temp):
    if temp < 26:
        return '24-26°C'
    elif 26 <= temp < 28:
        return '26-28°C'
    elif 28 <= temp < 30:
        return '28-30°C'
    else:
        return 'Above 30°C'

# Apply the categorization function to the SST data
merged_df['SST_Category'] = merged_df['sea_water_temperature'].apply(categorize_sst)

# Count the number of hurricanes in each SST category
hurricane_counts_by_sst = merged_df['SST_Category'].value_counts()

# Plot the results
plt.figure(figsize=(10,6))
hurricane_counts_by_sst.plot(kind='bar', color='skyblue')
plt.title('Frequency of Hurricanes by SST Category')
plt.xlabel('Sea Surface Temperature Category')
plt.ylabel('Number of Hurricanes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Optionally, save the results to a CSV file
hurricane_counts_by_sst.to_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/ANALYSIS_threshold_for_formation.csv', header=True)

print("SST categories and hurricane frequencies analysis complete.")
