import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load filtered temperature data
filtered_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_water_temperature_data.csv')

# Load hurricane data
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/filtered_hurricane_data.csv')

# Convert 'Date' columns to datetime
filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
hurricane_df['Date'] = pd.to_datetime(hurricane_df['Year'].astype(str) + '-' + hurricane_df['Day'].astype(str))

# Ensure that the hurricane data contains the 'Category' column (adjust if needed)
if 'Category' not in hurricane_df.columns:
    print("No 'Category' column found in hurricane data. Ensure the data includes hurricane intensity categories.")
else:
    # Create a column to store the period classification ('Before', 'During', 'After')
    filtered_df['Period'] = 'Non-Hurricane'  # Default value

    # Loop through each hurricane and categorize periods
    for _, hurricane in hurricane_df.iterrows():
        hurricane_date = hurricane['Date']
        start_date = hurricane_date - pd.Timedelta(days=15)
        end_date = hurricane_date + pd.Timedelta(days=15)

        # Classify before, during, or after the hurricane
        before_condition = (filtered_df['Date'] >= start_date) & (filtered_df['Date'] < hurricane_date)
        after_condition = (filtered_df['Date'] > hurricane_date) & (filtered_df['Date'] <= end_date)
        
        # Assign periods for each category of hurricane
        filtered_df.loc[before_condition, 'Period'] = 'Before'
        filtered_df.loc[after_condition, 'Period'] = 'After'
        filtered_df.loc[filtered_df['Date'] == hurricane_date, 'Period'] = 'During'

        # Ensure we have the hurricane category
        category = hurricane['Category']
        
        # Add the category to the temperature data
        filtered_df.loc[before_condition | after_condition | (filtered_df['Date'] == hurricane_date), 'Hurricane_Category'] = category

    # Calculate the average temperature before, during, and after hurricanes of each category
    result = filtered_df.groupby(['Hurricane_Category', 'Period'])['sea_water_temperature'].mean().reset_index()

    # Save the results to a CSV file
    result.to_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/hurricane_category_temperature_analysis.csv', index=False)

    print("Analysis complete. Results saved to 'hurricane_category_temperature_analysis.csv'.")


# Load the results CSV file
result = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/hurricane_category_temperature_analysis.csv')

# Set seaborn style for better plots
sns.set(style="whitegrid")

# Create a bar plot to compare temperature by category and period
plt.figure(figsize=(10, 6))
sns.barplot(data=result, x='Hurricane_Category', y='sea_water_temperature', hue='Period', palette="coolwarm")

# Add labels and title
plt.title('Average Water Temperature Before, During, and After Hurricanes by Category')
plt.xlabel('Hurricane Category')
plt.ylabel('Average Water Temperature (°C)')

# Show the plot
plt.tight_layout()
plt.show()