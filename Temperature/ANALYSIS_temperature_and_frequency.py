import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the hurricane and temperature data
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_hurricane_data.csv')
temperature_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_water_temperature_data.csv')

# Convert 'Date' in hurricane_df to datetime
hurricane_df['Date'] = pd.to_datetime(hurricane_df['Year'].astype(str) + '-' + hurricane_df['Day'].astype(str))

# Extract the year from the Date column
hurricane_df['Year'] = hurricane_df['Date'].dt.year

# Calculate hurricane frequency per year
hurricane_frequency = hurricane_df.groupby('Year').size().reset_index(name='Hurricane_Frequency')

# Convert 'Date' in temperature_df to datetime
temperature_df['Date'] = pd.to_datetime(temperature_df['Date'])

# Extract the year from the Date column in temperature_df
temperature_df['Year'] = temperature_df['Date'].dt.year

# Calculate the average water temperature per year
average_temperature = temperature_df.groupby('Year')['sea_water_temperature'].mean().reset_index(name='Average_Temperature')

# Merge the two dataframes on 'Year'
merged_df = pd.merge(hurricane_frequency, average_temperature, on='Year', how='inner')

# Inspect the merged dataframe
print(merged_df.head())

# Calculate the correlation coefficient
correlation = merged_df['Hurricane_Frequency'].corr(merged_df['Average_Temperature'])
print(f'Correlation Coefficient: {correlation}')

# Set seaborn style for better plots
sns.set(style="whitegrid")

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x='Hurricane_Frequency', y='Average_Temperature', color='blue')

# Add labels and title
plt.title('Hurricane Frequency vs. Average Water Temperature')
plt.xlabel('Hurricane Frequency (Number of Hurricanes per Year)')
plt.ylabel('Average Water Temperature (°C)')

# Show the plot
plt.tight_layout()
plt.show()
