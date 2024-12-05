import os
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np




file_path = r"D:\OneDrive - Texas A&M University\GEOG 362\Group Projects\Data\SocioEconomic\nhgis0003_ds262_20225_blck_grp.csv"
shapefile_path = r"D:\OneDrive - Texas A&M University\GEOG 362\Group Projects\Data\Shapefiles\Boundary\Tx_BlockGroup_2022.shp"
snap_raster_path = "D:/OneDrive - Texas A&M University/GEOG 362/Group Projects/Data/Rasterfiles/LULC/nlcd_2021_reclassify_prj.tif"


data = pd.read_csv(file_path)
shapefile = gpd.read_file(shapefile_path, engine="pyogrio")



shapefile.plot()



print(data)



columns_to_keep = [
    "GISJOIN", "COUNTY", "GEO_ID",
    "AQM4E001", "AQM4E002", "AQM4E003", "AQM4E004", "AQM4E005", "AQM4E006",
    "AQM4E026", "AQM4E027", "AQM4E028", "AQM4E029", "AQM4E030", "AQM4E031",
    "AQM5E001", "AQM5E002", "AQM5E003", "AQNGE002", "AQNGE003", "AQNGE004",
    "AQNGE005", "AQNGE006", "AQNGE007", "AQNWE006", "AQNWE011", "AQN3E001",
    "AQN7E002", "AQN7E003", "AQN7E004", "AQN7E005", "AQN7E006", "AQN7E007",
    "AQN7E008", "AQN7E009", "AQN7E010", "AQN7E011", "AQN7E012", "AQN7E013",
    "AQN9E001", "AQOCE002", "AQOCE022", "AQO6E018", "AQO6E019", "AQO7E025",
    "AQO7E049", "AQP0E002", "AQP1E001", "AQP2E002", "AQQKE003", "AQR1E005",
    "AQR8E005", "AQR8E007", "AQTXE010", "AQTXE011", "AQT0E006", "AQT0E007",
    "AQT0E008", "AQT0E009", "AQT0E010", "AQT0E011", "AQT1E001", "AQT9M007",
    "AQT9M016", "AQUAM003", "AQUAM010", "AQUDM003", "AQUGM003"
]

data = data[columns_to_keep]
print(data)



column_mapping = {
    "GISJOIN": "GISJOIN",
    "COUNTY": "COUNTY",
    "GEO_ID": "GEO_ID",
    "AQM4E001": "Total population",
    "AQM4E002": "Total Male",
    "AQM4E003": "Male: Under 5 years",
    "AQM4E004": "Male: 5 to 9 years",
    "AQM4E005": "Male: 10 to 14 years",
    "AQM4E006": "Male: 15 to 17 years",
    "AQM4E026": "Total Female",
    "AQM4E027": "Female: Under 5 years",
    "AQM4E028": "Female: 5 to 9 years",
    "AQM4E029": "Female: 10 to 14 years",
    "AQM4E030": "Female: 15 to 17 years",
    "AQM4E031": "Female: 18 and 19 years",
    "AQM5E001": "Median age: Total",
    "AQM5E002": "Median age: Male",
    "AQM5E003": "Median age: Female",
    "AQNGE002": "White alone",
    "AQNGE003": "Black or African American alone",
    "AQNGE004": "American Indian and Alaska Native alone",
    "AQNGE005": "Asian alone",
    "AQNGE006": "Native Hawaiian and Other Pacific Islander alone",
    "AQNGE007": "Some Other Race alone",
    "AQNWE006": "Male Workers 16 years and over",
    "AQNWE011": "Female Workers 16 years and over",
    "AQN3E001": "Aggregate travel time to work (in minutes)",
    "AQN7E002": "Less than 5 minutes",
    "AQN7E003": "5 to 9 minutes",
    "AQN7E004": "10 to 14 minutes",
    "AQN7E005": "15 to 19 minutes",
    "AQN7E006": "20 to 24 minutes",
    "AQN7E007": "25 to 29 minutes",
    "AQN7E008": "30 to 34 minutes",
    "AQN7E009": "35 to 39 minutes",
    "AQN7E010": "40 to 44 minutes",
    "AQN7E011": "45 to 59 minutes",
    "AQN7E012": "60 to 89 minutes",
    "AQN7E013": "90 or more minutes",
    "AQN9E001": "Total Population under 18 years in HH",
    "AQOCE002": "Lives alone",
    "AQOCE022": "65 years and over",
    "AQO6E018": "Female: Widowed",
    "AQO6E019": "Female: Divorced",
    "AQO7E025": "Male: Not enrolled in school",
    "AQO7E049": "Female: Not enrolled in school",
    "AQP0E002": "Families_Income in the past 12 months below poverty level",
    "AQP1E001": "Aggregate income deficit in the past 12 months",
    "AQP2E002": "HH_Income in the past 12 months below poverty level",
    "AQQKE003": "No earnings",
    "AQR1E005": "Household did not receive Food Stamps/SNAP in the past 12 months",
    "AQR8E005": "In labor force: Civilian labor force: Unemployed",
    "AQR8E007": "Not in labor force",
    "AQTXE010": "Mobile home",
    "AQTXE011": "Boat, RV, van, etc",
    "AQT0E006": "Built 1980 to 1989",
    "AQT0E007": "Built 1970 to 1979",
    "AQT0E008": "Built 1960 to 1969",
    "AQT0E009": "Built 1950 to 1959",
    "AQT0E010": "Built 1940 to 1949",
    "AQT0E011": "Built 1939 or earlier",
    "AQT1E001": "Median year structure built",
    "AQT9M007": "Owner occupied: No telephone service available",
    "AQT9M016": "Renter occupied: No telephone service available",
    "AQUAM003": "Owner occupied: No vehicle available",
    "AQUAM010": "Renter occupied: No vehicle available",
    "AQUDM003": "Lacking complete plumbing facilities",
    "AQUGM003": "Lacking complete kitchen facilities"
}



data.rename(columns=column_mapping, inplace=True)


print(data)


data['MalePop_under18'] = data['Male: Under 5 years'] + data['Male: 5 to 9 years'] + data['Male: 10 to 14 years'] + data['Male: 15 to 17 years']
data['FemalePop_under18'] = data['Female: Under 5 years'] + data['Female: 5 to 9 years'] + data['Female: 10 to 14 years'] + data['Female: 15 to 17 years']
data["TravelTimetoWork_morethan30min"] = data["30 to 34 minutes"]+data["35 to 39 minutes"]+data["40 to 44 minutes"]+data["45 to 59 minutes"]+data["60 to 89 minutes"]+ data["90 or more minutes"]

data["HouseMorethan40yrsOld"] = data["Built 1980 to 1989"]+data["Built 1970 to 1979"]+data["Built 1960 to 1969"]+data["Built 1950 to 1959"]+data["Built 1940 to 1949"]+ data["Built 1939 or earlier"]
data["No telephone service in HH"] = data ["Owner occupied: No telephone service available"]+ data["Renter occupied: No telephone service available"]
data["No vehicle in HH"] = data["Owner occupied: No vehicle available"]+data["Renter occupied: No vehicle available"]



print(data.columns)

missing_values = data.isna().sum()
missing_values

len(data["GISJOIN"].unique())
gisjoin_values = shapefile['GISJOIN'].unique()
len(gisjoin_values)

filtered_data = data[data['GISJOIN'].isin(gisjoin_values)]
filtered_data

missing_values1 = filtered_data.isna().sum()
missing_values1

filtered_data.info()
numeric_cols = filtered_data.select_dtypes(include = ["float64" , "int64"])
print(numeric_cols)

missing_cols = [col for col in numeric_cols if col not in filtered_data.columns]
print("Missing columns:", missing_cols)

print(filtered_data.dtypes)


numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(numeric_cols)


# Define the output Excel file path
excel_output_path = r"D:\OneDrive - Texas A&M University\GEOG 361 Group Projects\Data\SocioEconomic\scaled_df_with_ids.xlsx"

# Save the DataFrame to Excel
scaled_df_with_ids.to_excel(excel_output_path, index=False)

print(f"Excel file saved at: {excel_output_path}")


plt.figure(figsize=(12, 10))
sns.heatmap(scaled_df_with_ids.drop(columns=['GISJOIN', 'COUNTY', 'GEO_ID']).corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()