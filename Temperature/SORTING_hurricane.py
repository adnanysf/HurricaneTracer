import pandas as pd

# Load the hurricane data CSV
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/Hurricane_data.csv')

# Check the data to see if the 'Day' column contains full month names or dates
print(hurricane_df.head())  # To inspect the first few rows

# Function to handle non-numeric day formats (like "June 28")
def parse_date(row):
    try:
        # Try to parse the day and year when Day is in a full date format like "June 28"
        return pd.to_datetime(f"{row['Year']} {row['Day']}", format="%Y %B %d")
    except Exception as e:
        # If it doesn't match the expected format, return NaT (Not a Time)
        return pd.NaT

# Apply the parsing function to the 'Day' column
hurricane_df['Date'] = hurricane_df.apply(parse_date, axis=1)

# Drop rows where the 'Date' column is NaT (which means it couldn't be parsed)
hurricane_df = hurricane_df.dropna(subset=['Date'])

# Remove duplicate dates (keep only the first instance of each unique date)
hurricane_df = hurricane_df.drop_duplicates(subset=['Date'])

# Sort the data by the 'Date' column
hurricane_df = hurricane_df.sort_values(by='Date')

# Optional: Reset index after sorting
hurricane_df = hurricane_df.reset_index(drop=True)

# Save the cleaned data to a new CSV file
hurricane_df.to_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/filtered_hurricane_data.csv', index=False)

print("Hurricane data sorted and cleaned. Saved to 'filtered_hurricane_data.csv'.")
