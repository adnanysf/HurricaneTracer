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
import matplotlib.dates as mdates
import matplotlib.animation as animation
from PIL import Image
import imageio.v2 as imageio

hurricane_lists = ["katrina", "rita", "harvey", "irma", "michael", "laura","ida"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

def create_directories(name):
    folders = {}
    # Create folders dynamically for each hurricane
    # Parent folders
    data_folder = os.path.join(BASE_DIR, "Data_Folder", name.capitalize())
    maps_folder = os.path.join(BASE_DIR, "Maps", f"{name.capitalize()}_Maps")
    graphs_folder = os.path.join(BASE_DIR, "Graphs_Folder",f"{name.capitalize()}_Graphs")
    txt_folder = os.path.join(BASE_DIR, "Hurricane_txt")
    animation_folder = os.path.join(BASE_DIR, "Animations_Folder", f'{name.capitalize()}_Animation')
    
    # Create directories if they don't exist
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(maps_folder, exist_ok=True)
    os.makedirs(graphs_folder, exist_ok=True)
    os.makedirs(txt_folder, exist_ok=True)
    os.makedirs(animation_folder, exist_ok=True)

    print(f"Directories created for {name}:")
    print(f"Data folder: {data_folder}")
    print(f"Maps folder: {maps_folder}")
    print(f"Graphs folder: {graphs_folder}")
    print(f"TXT folder: {txt_folder}")
    print(f"Animation folder: {animation_folder}")
    print()
    
    # Store the paths in a dictionary for later use
    folders[name] = {
        'data_folder': data_folder,
        'maps_folder': maps_folder,
        'graphs_folder': graphs_folder,
        'txt_folder': txt_folder,
        'animation_folder' : animation_folder
    }
    return folders

def folder_exists_and_not_empty(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is not empty
        if os.listdir(folder_path):  # Returns True if the folder is not empty
            return True
    return False

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
        formatted_datetime = parsed_datetime.strftime('%Y-%m-%d_%H-%M')  # Replace colons with dashes
        
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
        output_file_path = os.path.join(maps_folder_full_path, 
                                        f"rainfall_map_{formatted_datetime}.png")
        plt.tight_layout()
        plt.gcf().canvas.draw()  # Force a canvas draw before saving
        # plt.show()
        plt.savefig(output_file_path, dpi=300)
        # print(f"Map saved to {output_file_path}")
        plt.close(fig)  # Close the figure after saving
        dataset.close()

def gather_precipitation_data(cities, data_folder, name):
    # Add new keys to store precipitation data
    hurricane_data[name]["daily_precipitation"] = {city: [] for city in cities.keys()}
    hurricane_data[name]["total_precipitation"] = {city: 0 for city in cities.keys()}
    
    # List to track days for time series plot
    days = []

    # Loop through each file in the data folder
    for file_name in (os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        dataset = Dataset(file_path)

        # Assuming the shape is (time, lat, lon)
        rainfall = np.sum(dataset.variables["Rainf"][:, :, :], axis=0)  # Sum over time axis
        # print(f"Processing file: {file_name}, Shape of rainfall data: {rainfall.shape}")

        # Extract the date from the file name (assuming the format like '20050823')
        date_str = "".join(file_name.split(".")[1:3])
        # Extract the part that contains date and time
        date_time_part = date_str[1:]  # Ignore the 'A' and extract the rest
        # Parse the string into a datetime object
        parsed_datetime = datetime.strptime(date_time_part, '%Y%m%d%H%M')
        days.append(parsed_datetime)  # Track the date

        for city, (city_lat, city_lon) in cities.items():
            # Find nearest indices for lat/lon in the dataset's grid
            lat_idx = np.abs(dataset.variables["lat"][:] - city_lat).argmin()
            lon_idx = np.abs(dataset.variables["lon"][:] - city_lon).argmin()

            # Append the precipitation for this city on this day
            precip_value = rainfall[lat_idx, lon_idx]
            hurricane_data[name]["daily_precipitation"][city].append(precip_value)

            # Sum total precipitation
            hurricane_data[name]["total_precipitation"][city] += precip_value
            # print(f"City: {city}, Precip: {precip_value} kg/m^2")
            
        dataset.close()

    return hurricane_data, days
        
def plot_time_series_for_cities(daily_precipitation, days, name, output_folder):
    fig, ax = plt.subplots(figsize=(10, 6))
    # Different line styles for each city
    line_styles = ['-', '--', '-.', ':', '-']  # Solid, dashed, dash-dot, dotted
    # Thicker line widths for clarity
    line_widths = [2, 2.5, 3, 2, 2.5]  # Adjust thickness
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct colors

    for i, (city, precip_data) in enumerate(daily_precipitation.items()):
        # print(f"City: {city}, Precipitation data: {precip_data}")
        # Use the index `i` to access the corresponding styles, colors, and line widths
        ax.plot(days, precip_data, label=city, linestyle=line_styles[i], 
                color=colors[i], linewidth=line_widths[i])

    
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Precipitation (kg/m^2)')
    ax.set_title(f'Total Daily Precipitation for Hurricane {name}')
    ax.legend()
    # Format the x-axis to show fewer date ticks
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45, ha='right')  # Rotate for better readability
    plt.tight_layout()
    
    # Save the plot to the output directory with a unique name
    output_file_path = os.path.join(output_folder, 
                                    f"Total Daily Precipitation for Hurricane {name}.png")
    plt.savefig(output_file_path, dpi=300)  # Save with high resolution
    # plt.show()
    
    # Close the figure to ensure no overlap with the next iteration
    plt.close(fig)

def plot_total_precipitation(accumulated_precip, name, output_folder):
    # Access the total precipitation directly from the dictionary
    total_precipitation = hurricane_data[name]["total_precipitation"]
    # Extract cities and their total precipitation
    cities = list(total_precipitation.keys())
    total_precip_values = list(total_precipitation.values())

    # Plotting the total accumulated precipitation for each city
    fig, ax = plt.subplots()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each city
    ax.bar(cities, total_precip_values, color=colors)
    ax.set_xlabel('City')
    ax.set_ylabel('Total Accumulated Precipitation (kg/m^2)')
    ax.set_title(f'Total Accumulated Precipitation for Hurricane {name}')

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to the output directory
    output_file_path = os.path.join(output_folder, f"Total_Daily_Precipitation_Hurricane_{name}.png")
    plt.savefig(output_file_path, dpi=300)  # Save with high resolution
    # plt.show()

    # Close the figure to ensure no overlap with the next iteration
    plt.close(fig)

def create_videos_from_maps(maps_folder, animation_folder, name, fps=2):
    # Get the list of map images
    image_files = sorted([os.path.join(maps_folder, f) for f in os.listdir(maps_folder) if f.endswith(".png")])
    # Save animation as mp4
    output_file = os.path.join(animation_folder, f'{name.capitalize()}_animation.mp4')  # or f'{name}_animation.mp4' if looping through names
    # Open writer
    with imageio.get_writer(output_file, fps=fps) as writer:
        for image_file in image_files:
            img_path = os.path.join(maps_folder, image_file)
            image = Image.open(img_path)

            # Resize to (3008x3008) if necessary
            resized_image = image.resize((3008, 3008))

            # Convert the PIL Image to a NumPy array
            resized_image_np = np.array(resized_image)

            # Append the NumPy array to the writer
            writer.append_data(resized_image_np)
    
    print(f"Video saved to {output_file}")
    
if __name__ == "__main__":
    for name in hurricane_lists:
        extent = hurricane_data[name]["extent"]
        cities = hurricane_data[name]["cities"]
        # Create directories and get the paths for each hurricane
        folder_paths = create_directories(name) 
        # Get the paths from the folder_paths dictionary
        data_folder = folder_paths[name]['data_folder']
        maps_folder = folder_paths[name]['maps_folder']
        graphs_folder = folder_paths[name]['graphs_folder']
        txt_folder = folder_paths[name]['txt_folder']
        animation_folder = folder_paths[name]['animation_folder']
        
        # Check if the data folder exists and is not empty
        if not folder_exists_and_not_empty(data_folder):
            download_links(name, data_folder)
        else:
            print(f"Data for {name.capitalize()} already exists and is not empty")
            
        # Gather precipitation data across all .nc files for this hurricane
        hurricane_data, days = gather_precipitation_data(cities, data_folder, name)
        
        # Check if the maps folder exists and is not empty
        if not folder_exists_and_not_empty(maps_folder):
            for file_name in os.listdir(data_folder):
                print(f'creating animation for {name.capitalize()}...')
                create_maps(maps_folder, data_folder, file_name, name, extent, cities)
        else:
            print(f"Map for {name.capitalize()} already exists and is not empty")
            
        # Check if the graphs folder exists and is not empty
        if not folder_exists_and_not_empty(graphs_folder):
            # Plot the daily time series for each city
            plot_time_series_for_cities(hurricane_data[name]["daily_precipitation"], days, name, graphs_folder)
            # Plot total precipitation if desired
            plot_total_precipitation(hurricane_data[name]["total_precipitation"], name, graphs_folder)
        else:
            print(f"Graphs for {name.capitalize()} already exist and are not empty")
    
        # Check if the data folder exists and is not empty
        if not folder_exists_and_not_empty(animation_folder):
            create_videos_from_maps(maps_folder, animation_folder, name, 2)
        else:
            print(f"Data for {name.capitalize()} already exists and is not empty")       
    
        