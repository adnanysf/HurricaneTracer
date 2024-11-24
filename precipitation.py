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
from matplotlib.patches import Rectangle

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
    """
    The `create_directories` function dynamically creates directories for a 
    hurricane and stores the paths in a dictionary.
    
    :param name: `name` parameter as input, which is the name
    of a hurricane. The function then creates directories for various purposes related to that
    hurricane, such as storing data, maps, graphs, text files, and animations. 
    :return: a dictionary containing paths to various folders created for a hurricane. 
    The keys in the dictionary are the name of the hurricane, and the
    values are dictionaries containing paths to the following folders:
    - 'data_folder': Path to the data folder for the hurricane
    - 'maps_folder': Path to the maps folder for the hurricane
    - 'graphs_folder': Path to graphs folder for the hurricane
    - 'txt_folder': Path to txt folder for the hurricane containing the download links to the .nc files
    - 'animation_folder': Path to animation folder for the hurricane
    """
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
    """
    The function `folder_exists_and_not_empty` checks if a folder exists and is not empty.
    
    :param folder_path: The `folder_path` parameter is a string that represents the path to a folder on
    the file system
    :return: returns `True` if the folder at the specified path exists.
    `folder_path` exists and is not empty. Otherwise, it returns `False`.
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is not empty
        if os.listdir(folder_path):  # Returns True if the folder is not empty
            return True
    return False

def download_links(name, data_folder_full_path):
    """
    The `download_links` function reads URLs from a text file, downloads corresponding NetCDF files, and
    saves them in a specified directory.
    
    :param name: The `name` parameter in the `download_links` function is used to specify the name of
    the .txt file that contains the list of URLs to download NetCDF files from
    :param data_folder_full_path: The `data_folder_full_path` parameter should be the full path to the
    directory where you want to save the downloaded files.
    """
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
    """
    The function `create_maps` generates a rainfall map 
    visualization for a specific dataset, focusing on a specified geographical 
    extent and marking cities on the map.
    
    :param maps_folder_full_path: The `maps_folder_full_path` parameter is the full path to the
    directory where you want to save the generated rainfall maps.
    :param data_folder_full_path: The `data_folder_full_path` parameter
    represents the full path to the folder where the data files are located. 
    :param file_name: The `file_name` parameter represents the name of the
    file being processed. It is used to extract the date and time information from the file name to be
    displayed on the generated map.
    :param name: The `name` parameter represents the name of the hurricane
    for which the rainfall map is being generated.
    :param extent: The `extent` parameter is used to specify the
    geographical extent of the map to be created. It defines the bounding box coordinates that determine
    the area to be displayed on the map.
    :param cities: The `cities` parameter is a dictionary where
    the keys are city names and the values are tuples containing latitude and longitude coordinates of
    those cities.
    """
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
            cmap="plasma",  # Try 'plasma', 'cividis', or stick to 'viridis'
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

        # Add a more detailed colorbar with the absolute min/max rainfall values
        cbar = plt.colorbar(
            img,
            ax=ax,
            label="Rainfall (kg/m^2, LogNorm)",
            ticks=np.linspace(min_rainfall, max_rainfall, 10),
        )

        # Add text to indicate the absolute min and max rainfall in the dataset
        absolute_min = np.nanmin(rainfall)  # Absolute minimum across all points
        absolute_max = np.nanmax(rainfall)  # Absolute maximum across all points

        # Update the colorbar to include the absolute values
        cbar.ax.text(1.1, 0, f'Abs Min: {absolute_min:.2f}', va='center', ha='left', transform=cbar.ax.transAxes, fontsize=10)
        cbar.ax.text(1.1, 1, f'Abs Max: {absolute_max:.2f}', va='center', ha='left', transform=cbar.ax.transAxes, fontsize=10)

        cbar.set_label('Rainfall (kg/m², LogNorm)')
        # Label the color bar with additional details
        cbar.ax.set_ylabel("Rainfall (kg/m^2, Log Scale)", rotation=-90, va="bottom")

        plt.title(f"Rainfall for Hurricane {name.capitalize()} at {formatted_datetime}")

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
    """
    The function `gather_precipitation_data` processes precipitation data from files in a specified
    folder for given cities and updates the hurricane data dictionary with daily and total precipitation
    values.
    
    :param cities: The `cities` parameter is a dictionary
    where the keys are city names and the values are tuples containing the latitude and longitude
    coordinates of each city. This dictionary is used to extract precipitation data for each city.
    :param data_folder: The `data_folder` parameter is the
    path to the folder where the precipitation data files are stored. This function reads precipitation
    data from files in this folder to gather information for each city specified in the `cities`
    dictionary.
    :param name: The `name` parameter is a string that
    represents the name of the hurricane event for which you are gathering precipitation data
    :return: The function `gather_precipitation_data` returns two main values: 
    1. `hurricane_data`: This is a dictionary containing precipitation data for each city. It includes
    keys for daily precipitation and total precipitation for each city.
    2. `days`: This is a list containing datetime objects representing the days for which precipitation
    data was gathered.
    """
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
    """
    The function `plot_time_series_for_cities` generates a time series plot of daily precipitation data
    for multiple cities during a hurricane event and saves the plot as an image file.
    
    :param daily_precipitation: The `daily_precipitation` parameter is a dictionary where the keys are
    city names and the values are lists of daily precipitation data for each city. Each list represents
    the daily precipitation values for that city over a period of time.
    :param days: The `days` parameter represents the
    x-axis values for the time series plot. It could be a list of dates or numerical values representing
    the days for which the daily precipitation data is available for different cities.
    :param name: The `name` parameter represents the name
    of the hurricane for which the total daily precipitation data is being plotted.
    :param output_folder: The `output_folder` parameter
    refers to the directory path where the generated plot will be saved as a PNG file.
    """
    # Convert kg/m² to inches (1 kg/m² = 0.0393701 inches)
    conversion_factor = 0.0393701
    daily_precipitation_inches = {
        city: [val * conversion_factor for val in values]
        for city, values in daily_precipitation.items()
    }

    # Find the maximum precipitation value across all cities
    max_precip = max(max(values) for values in daily_precipitation_inches.values())
    
    # Calculate y-axis limits with padding
    y_max = max_precip * 1.15  # Add 15% padding for annotations
    
    # Round up to the next 0.5 increment for a cleaner look
    y_max = np.ceil(y_max * 2) / 2

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define precipitation thresholds (in inches)
    thresholds = {
        'Light': (0, 0.1),
        'Moderate': (0.1, 0.3),
        'Heavy': (0.3, 0.5),
        'Very Heavy': (0.5, 2.0)
    }

    # Add colored bands for precipitation thresholds
    colors = ['#63C5DA', '#0492C2', '#2832C2', '#0A1172']
    for (label, (lower, upper)), color in zip(thresholds.items(), colors):
        # If threshold upper bound exceeds y_max, cap it
        upper_bound = min(upper, y_max)
        ax.add_patch(Rectangle((min(days), lower),
                            max(days) - min(days),
                            upper_bound - lower,
                            facecolor=color,
                            alpha=0.3,
                            label=f'{label} ({lower}-{upper} in)'))

    # Plot lines for each city
    line_styles = ['-', '--', '-.', ':', '-']
    line_widths = [2.5, 2.5, 2.5, 2.5, 2.5]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (city, precip_data) in enumerate(daily_precipitation_inches.items()):
        line = ax.plot(days, precip_data,
                    label=city,
                    linestyle=line_styles[i],
                    color=colors[i],
                    linewidth=line_widths[i])

        # Find and annotate peak precipitation
        peak_idx = np.argmax(precip_data)
        peak_value = precip_data[peak_idx]
        peak_date = days[peak_idx]

        # Add annotation with arrow
        ax.annotate(f'{peak_value:.2f} in',
                xy=(peak_date, peak_value),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5',
                        fc='white',
                        alpha=0.7),
                arrowprops=dict(arrowstyle='->',
                                connectionstyle='arc3,rad=0',
                                color=colors[i]))

    # Set y-axis limits and ticks
    ax.set_ylim(0, y_max)
    
    # Calculate number of y-axis ticks based on max value
    if y_max <= 1:
        y_ticks = np.arange(0, y_max + 0.1, 0.1)
    elif y_max <= 2:
        y_ticks = np.arange(0, y_max + 0.2, 0.2)
    else:
        y_ticks = np.arange(0, y_max + 0.5, 0.5)
    
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # Customize axes
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Precipitation (inches)', fontsize=12)
    ax.set_title(f'Daily Precipitation for Hurricane {name}',
                fontsize=14,
                pad=20)

    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    threshold_handles = handles[:len(thresholds)]
    city_handles = handles[len(thresholds):]
    threshold_labels = labels[:len(thresholds)]
    city_labels = labels[len(thresholds):]
    
    ax.legend(threshold_handles + city_handles,
            threshold_labels + city_labels,
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            fontsize=10)

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    output_file_path = os.path.join(
        output_folder,
        f"Daily_Precipitation_Hurricane_{name}_inches.png"
    )
    plt.savefig(output_file_path,
                dpi=300,
                bbox_inches='tight')
    plt.close(fig)

def plot_total_precipitation(accumulated_precip, name, output_folder):
    """
    The function `plot_total_precipitation` generates a bar plot showing the total accumulated
    precipitation for each city affected by a hurricane and saves the plot as a PNG file in the
    specified output folder.
    
    :param accumulated_precip: It seems like you were about to provide a description of the
    `accumulated_precip` parameter in the `plot_total_precipitation` function. Could you please continue
    with the description so that I can assist you further?
    :param name: The `name` parameter in the `plot_total_precipitation` function is used to specify the
    name of the hurricane for which you want to plot the total accumulated precipitation data. This name
    is used to access the corresponding data from the `hurricane_data` dictionary and generate a plot
    showing the
    :param output_folder: The `output_folder` parameter in the `plot_total_precipitation` function is a
    string that represents the directory path where the output plot will be saved. This parameter
    specifies the location where the generated plot image file will be stored after the function is
    executed. It should be a valid path on the
    """
    # Access the total precipitation
    total_precipitation = hurricane_data[name]["total_precipitation"]
    
    # Convert kg/m² to inches
    conversion_factor = 0.0393701
    total_precipitation_inches = {
        city: value * conversion_factor for city, value in total_precipitation.items()
    }
    
    # Extract cities and their total precipitation in inches
    cities = list(total_precipitation_inches.keys())
    total_precip_values = list(total_precipitation_inches.values())
    
    # Create figure with larger size for better readability
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors for each city
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create bars
    bars = ax.bar(cities, total_precip_values, color=colors)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}"',
                ha='center', va='bottom')
    
    # Customize axes
    ax.set_xlabel('City', fontsize=12)
    ax.set_ylabel('Total Accumulated Precipitation (inches)', fontsize=12)
    ax.set_title(f'Total Accumulated Precipitation for Hurricane {name}',
                fontsize=14, pad=20)
    
    # Add grid for better readability (only horizontal lines)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Calculate y-axis limit with some padding for labels
    y_max = max(total_precip_values)
    ax.set_ylim(0, y_max * 1.1)  # Add 10% padding
    
    # Format y-axis ticks to show 2 decimal places
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_file_path = os.path.join(
        output_folder,
        f"Total_Accumulated_Precipitation_Hurricane_{name}_inches.png"
    )
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_videos_from_maps(maps_folder, animation_folder, name, fps=2):
    """
    The function `create_videos_from_maps` generates a video animation from a series of map images and
    saves it as an mp4 file.
    
    :param maps_folder: The `maps_folder` parameter in the `create_videos_from_maps` function is the
    directory path where the map images are stored. This function reads all the PNG images from this
    folder to create a video animation
    :param animation_folder: The `animation_folder` parameter in the `create_videos_from_maps` function
    refers to the directory where the resulting animation video will be saved. This folder should be
    specified as a string representing the path to the directory where you want to save the animation
    video file
    :param name: The `name` parameter in the `create_videos_from_maps` function is a string that
    represents the name of the animation or video that will be created. It is used to generate the
    output file name for the video
    :param fps: The `fps` parameter in the `create_videos_from_maps` function stands for frames per
    second. It determines how many frames (images) are displayed per second in the resulting video
    animation. A higher FPS value will result in a smoother animation, while a lower FPS value will make
    the animation appear more, defaults to 2 (optional)
    """
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
            print(f'creating animation for {name.capitalize()}...')
            create_videos_from_maps(maps_folder, animation_folder, name, 2)
        else:
            print(f"Data for {name.capitalize()} already exists and is not empty")       
    
        