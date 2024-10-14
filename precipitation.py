import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
import requests
import xarray as xr
from matplotlib.colors import LogNorm
from datetime import datetime
import time

hurricane_lists = ["katrina", "rita", "harvey", "irma", "michael", "laura","ida"]

# Define the directory for each of the hurricanes
txt_dir = r"C:\Users\Natalia\OneDrive - Texas A&M University\GEOG 676\HurricaneTracer\Hurricane_txt"
data_dir = r"C:\Users\Natalia\OneDrive - Texas A&M University\GEOG 676\HurricaneTracer\Data_Folder"
map_dir = r"C:\Users\Natalia\OneDrive - Texas A&M University\GEOG 676\HurricaneTracer\Maps"

hurricane_data ={
    "katrina": {
        "extent": [-90, -85, 28, 32],
        "cities": {
            "New Orleans": (29.9511, -90.0715),
            "Biloxi": (30.3960, -88.8853)
        }
    },
    "rita":{
        "extent": [-95, -90, 28, 32],
        "cities": {
            "Lake Charles": (30.2266, -93.2174),
            "Beaumont": (30.0802, -94.1266)
        }
    },
    "harvey": {
        "extent": [-98, -94, 27, 31],
        "cities": {
            "Houston": (29.7604, -95.3698),
            "Rockport": (28.0206, -97.0544),
            "Corpus Christi": (27.8006, -97.3964)
        }
    },
    "ida": {
        "extent": [-95, -88, 28, 32],
        "cities" :{
            "New Orleans": (29.9511, -90.0715),
            "Baton Rouge": (30.4515, -91.1871),
            "Houma": (29.5958, -90.7195),
            "Thibodaux": (29.7958, -90.8188)
        }
    },
    "irma": {
        "extent": [-88, -79, 24, 32],
        "cities" : {
            'Miami': (25.7617, -80.1918),
            'Naples': (26.1420, -81.7948),
            'Tampa': (27.9506, -82.4572),
            'Jacksonville': (30.3322, -81.6557),
            'Orlando': (28.5383, -81.3792)
        }
    },
    "laura": {
        "extent": [-95, -89, 28, 33],
        "cities" : {
            'Lake Charles': (30.2266, -93.2174),
            'Lafayette': (30.2241, -92.0198),
            'Baton Rouge': (30.4515, -91.1871)
        }
    },
    "michael": {
        "extent": [-88, -82, 29, 32],
        "cities" : {
            'Tallahassee': (30.4383, -84.2807),
            'Pensacola': (30.4213, -87.2169),
            'Panama City': (30.1588, -85.6602)
        }
    },
}
    
def create_maps_folder(name):
    # Format the folder name dynamically based on the hurricane name
    maps_folder = f"{name.capitalize()}_Maps"  # Capitalize the first letter of the hurricane name

    # Combine them to form the full path
    maps_folder_full_path = os.path.join(map_dir, maps_folder)

    # Check if the folder exists
    if os.path.exists(maps_folder_full_path):
        print(f"Map folder for {name} already exists at {maps_folder_full_path}")
    else:
        os.makedirs(maps_folder_full_path)
        print(f"Map folder for {name} created at {maps_folder_full_path}")
    return maps_folder_full_path

def create_data_folder(name):
    # Format the folder name dynamically based on the hurricane name
    data_folder = f"{name.capitalize()}"  # Capitalize the first letter of the hurricane name
    
    # Combine them to form the full path
    data_folder_full_path = os.path.join(data_dir, data_folder)

    # Check if the folder exists
    if os.path.exists(data_folder_full_path):
        print(f"Data folder for {name} already exists at {data_folder_full_path}")
    else:
        os.makedirs(data_folder_full_path)
        print(f"Data folder for {name} created at {data_folder_full_path}")
    return data_folder_full_path

def download_links(name, data_folder_full_path):
    # Update this with the path to your .txt file
    txt_path = os.path.join(txt_dir, f"{name}.txt")
    with open(txt_path, "r") as f:
        urls = f.readlines()

    # Loop through each URL and download the corresponding NetCDF file
    for url in urls:
        url = url.strip()  # Remove any leading/trailing whitespace or newline characters
        file_name = url.split("/")[-1]  # Extract the file name from the URL

        # print(f"Downloading {url} from {url}")

        # Send the HTTP request to download the file
        response = requests.get(url)

        # Create the full path for saving the file in the specified directory
        file_path = os.path.join(data_folder_full_path, file_name)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(response.content)
    print(f"{name}.txt urls downloaded")
   
def create_maps(maps_folder_full_path, data_folder_full_path, file_name,name,extent,cities):
    # Only process files with the .nc4 or .nc extension
    if file_name.endswith(".nc4") or file_name.endswith(".nc"):
        # Extract the date from the file name (assuming the format like '20050823')
        date_str = "".join(file_name.split(".")[1:3])
        # Extract the part that contains date and time
        date_time_part = date_str[1:]  # Ignore the 'A' and extract the rest
        # Parse the string into a datetime object
        parsed_datetime = datetime.strptime(date_time_part, '%Y%m%d%H%M')

        # Format the datetime object to a readable string
        formatted_datetime = parsed_datetime.strftime('%Y-%m-%d %H:%M')
        
        # Create the full path for saving the file in the specified directory
        file_path = os.path.join(data_folder_full_path, file_name)
        
        # Load the dataset
        dataset = Dataset(file_path)
        # print(dataset)
        rainfall = dataset.variables["Rainf"][0, :, :]  # Extract data for this file

        # Calculate the min and max for scaling the colormap (ignoring non-positive values for LogNorm)
        min_rainfall = np.nanmin(rainfall[rainfall > 0])  # Min > 0 for log scale
        max_rainfall = np.nanmax(rainfall)

        lon = dataset.variables["lon"][:]  # Assuming lon is in the dataset
        lat = dataset.variables["lat"][:]  # Assuming lat is in the dataset

        # Create a larger figure for better visualization
        fig, ax = plt.subplots(
            figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
        )  # Change figsize to (10, 10) for a larger, square plot

        # Set extent to focus on Florida
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Use pcolormesh to display the rainfall data with the Blues colormap and a log scale
        img = ax.pcolormesh(
            lon,
            lat,
            rainfall,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            shading="auto",
            norm=LogNorm(vmin=min_rainfall, vmax=max_rainfall),  # Apply normalization
        )

        # Add contour lines for better visibility of rainfall levels
        contour = ax.contour(
            lon,
            lat,
            rainfall,
            levels=np.linspace(min_rainfall, max_rainfall, 10),
            colors="black",
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.clabel(contour, inline=True, fontsize=8)

        # Add state boundaries for Texas, Louisiana, and other states
        states = cfeature.NaturalEarthFeature(
            category="cultural",
            scale="50m",
            facecolor="none",
            name="admin_1_states_provinces_lines",
        )

        # Add features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="black")
        ax.add_feature(states, edgecolor="black", linewidth=1)  # Add state boundaries

        # Plot the cities with markers and labels
        for city, (lat, lon) in cities.items():
            ax.plot(lon, lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree())
            # Offset the text to avoid covering the marker
            ax.text(lon + 0.2, lat + 0.2, city, fontsize=12, backgroundcolor="white", transform=ccrs.PlateCarree())

        # Add a colorbar with more granularity
        plt.colorbar(
            img,
            ax=ax,
            label="Rainfall (kg/m^2)",
            ticks=np.linspace(min_rainfall, max_rainfall, 10),
        )

        plt.title(f"Rainfall for Hurricane {name.capitalize()} {formatted_datetime}")

        # Save the plot to the output directory with a unique name
        output_file_path = os.path.join(maps_folder_full_path, f"rainfall_map_{formatted_datetime}.png")
        plt.tight_layout()
        plt.gcf().canvas.draw()  # Force a canvas draw before saving
        plt.savefig(output_file_path, dpi=300)
        print(f"Map saved to {output_file_path}")
        time.sleep(1)  # Add a delay to ensure file is written

        plt.close(fig)  # Close the figure after saving
        dataset.close()

def smooth_data(data, window_size=3):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')
        
def plot_smoothed_time_series_for_cities(hurricane_data, name):
    fig, ax = plt.subplots(figsize=(10, 6))

    for city, precip_data in hurricane_data.items():
        smoothed_data = smooth_data(precip_data)  # Apply smoothing
        ax.plot(smoothed_data, label=city)
    
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Precipitation (kg/m^2)')
    ax.set_title(f'Total Daily Precipitation for Hurricane {name}')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # Close the figure to ensure no overlap with the next iteration
    plt.close(fig)

def gather_precipitation_data(cities, data_folder, name):
    # Dictionary to accumulate daily precipitation data for each city
    accumulated_precip = {city: [] for city in cities.keys()}
    
    # List to track days for time series plot
    days = []

    # Loop through each file in the data folder
    for file_name in sorted(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        dataset = Dataset(file_path)

        # Assuming the shape is (time, lat, lon)
        rainfall = np.sum(dataset.variables["Rainf"][:, :, :], axis=0)  # Sum over time axis
        print(f"Processing file: {file_name}, Shape of rainfall data: {rainfall.shape}")

        # Extract the date from the file name or dataset to track days
        date_str = file_name.split('.')[1]  # Adjust based on your file naming pattern
        days.append(date_str)  # Track the date

        for city, (city_lat, city_lon) in cities.items():
            # Find nearest indices for lat/lon in the dataset's grid
            lat_idx = np.abs(dataset.variables["lat"][:] - city_lat).argmin()
            lon_idx = np.abs(dataset.variables["lon"][:] - city_lon).argmin()

            # Append the precipitation for this city on this day
            accumulated_precip[city].append(rainfall[lat_idx, lon_idx])
            print(f"City: {city}, Precip: {rainfall[lat_idx, lon_idx]}")
            
        dataset.close()

    return accumulated_precip, days

if __name__ == "__main__":
    for name in hurricane_lists:
        print("name",name)
        extent = hurricane_data[name]["extent"]
        print("extent", extent)
        cities = hurricane_data[name]["cities"]
        print("cities", cities)
        maps_folder = create_maps_folder(name)
        data_folder = create_data_folder(name)
        data_folder_full_path = os.path.join(data_folder, name)  # Change this according to your folder structure
        # Avoid re-downloading if data already exists
        if not os.listdir(data_folder):
            download_links(name, data_folder)
        else:
            print(f"Data for {name.capitalize()} already exists in {data_folder}")
        print("links downloaded")
        
        # Gather precipitation data across all .nc files for this hurricane
        accumulated_precip, days = gather_precipitation_data(cities, data_folder, name)
        # Plot the time series for all cities
        plot_smoothed_time_series_for_cities(cities, name)

        for file_name in os.listdir(data_folder):
            create_maps(maps_folder, data_folder, file_name, name, extent, cities)