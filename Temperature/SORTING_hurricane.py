import pandas as pd

# Load the hurricane data
hurricane_df = pd.read_csv('/Users/aubreyanderson/Desktop/geog392/PROJECT/HurricaneTracer/Temperature/Hurricane_data.csv')

# Inspect the raw data
print("Sample Data:")
print(hurricane_df.head())

# Clean up the 'Day' column and remove notes from 'Category'
hurricane_df['Day'] = hurricane_df['Day'].str.strip()  # Remove leading/trailing spaces
hurricane_df['Category'] = hurricane_df['Category'].str.replace(r'\[.*?\]', '', regex=True).str.strip()

# Function to parse the date
def parse_date(row):
    try:
        return pd.to_datetime(f"{row['Year']} {row['Day']}", format="%Y %B %d", errors="coerce")
    except Exception as e:
        print(f"Error parsing date for row: {row}")
        return pd.NaT

# Apply the parsing function
hurricane_df['Date'] = hurricane_df.apply(parse_date, axis=1)

# Identify rows with parsing issues
failed_parses = hurricane_df[hurricane_df['Date'].isna()]
if not failed_parses.empty:
    print("Rows with parsing issues:")
    print(failed_parses)

# Drop rows with invalid dates
hurricane_df = hurricane_df.dropna(subset=['Date'])

# Remove duplicate dates and sort
hurricane_df = hurricane_df.drop_duplicates(subset=['Date']).sort_values(by='Date').reset_index(drop=True)

# Save the cleaned data
output_path = '/Users/aubreyanderson/Desktop/geog392/PROJECT/filtered_hurricane_data.csv'
hurricane_df.to_csv(output_path, index=False)

print(f"Hurricane data sorted and cleaned. Saved to '{output_path}'.")