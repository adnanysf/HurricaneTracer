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
    Create and organize a structured directory hierarchy for hurricane data analysis.

    This function establishes a standardized folder structure for storing hurricane-related
    data, including raw data, visualization outputs, and animation files. All directories
    are created relative to a global BASE_DIR path.

    Parameters
    ----------
    name : str
        Name of the hurricane.
        Will be capitalized in folder names for consistency.

    Returns
    -------
    dict
        Nested dictionary containing all created directory paths.

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
    Verify if a specified folder exists and contains at least one file or subdirectory.

    This function performs a two-step validation: first checking if the path exists,
    then verifying that it contains at least one entry. It handles both files and
    subdirectories in its emptiness check.

    Parameters
    ----------
    folder_path : str or Path
        Path to the folder to be checked.
        Can be either absolute or relative path.

    Returns
    -------
    bool
        True if both conditions are met:
            - Folder exists
            - Folder contains at least one file or subdirectory
        False if either:
            - Folder doesn't exist
            - Folder exists but is empty
    """
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Check if the folder is not empty
        if os.listdir(folder_path):  # Returns True if the folder is not empty
            return True
    return False

def download_links(name, data_folder_full_path, txt_dir):
    """
    Download NetCDF files from URLs listed in a text file.

    This function reads a text file containing URLs for NetCDF files, downloads each file,
    and saves them to a specified directory. Each URL should point to a NetCDF file
    and be listed on a separate line in the text file.

    Parameters
    ----------
    name : str
        Base name of the text file containing URLs (without .txt extension).
        The file should be located in the directory specified by txt_dir.
        Example: if name='ian', function looks for 'ian.txt'

    data_folder_full_path : str or Path
        Directory path where downloaded NetCDF files will be saved.
        Must be an existing directory with write permissions.

    Returns
    -------
    None
        Prints confirmation message upon completion of downloads.
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
    Generate detailed rainfall visualization maps for hurricane data with geographic features and city markers.

    This function creates sophisticated rainfall maps using NetCDF data, incorporating geographic
    boundaries, city markers, contour lines, and a logarithmic color scale. Each map represents
    a specific time point during a hurricane event.

    Parameters
    ----------
    maps_folder_full_path : str or Path
        Directory path where generated map images will be saved.
        Must be an existing directory with write permissions.

    data_folder_full_path : str or Path
        Directory path containing the NetCDF data files to be processed.
        Files should be in .nc4 or .nc format.

    file_name : str
        Name of the NetCDF file to process.
        Must follow format with embedded datetime (e.g., 'prefix.YYYYMMDDHHMM.suffix').

    name : str
        Name of the hurricane being visualized.
        Used in map titles and output filenames.

    extent : list or tuple
        Geographic bounds for the map in format [lon_min, lon_max, lat_min, lat_max].
        Determines the visible area of the visualization.

    cities : dict
        Dictionary of cities to mark on the map.

    Returns
    -------
    None
        Saves the generated map as a PNG file and closes all figure resources.
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
    Process and aggregate precipitation data from NetCDF files for multiple cities during a hurricane event.

    This function reads precipitation data from NetCDF files, extracts values for specified city locations,
    and aggregates both daily and total precipitation values. It processes temporal data series and
    matches geographic coordinates to the nearest grid points in the dataset.

    Parameters
    ----------
    cities : dict
        Dictionary mapping city names to their coordinates.

    data_folder : str or Path
        Directory path containing the NetCDF precipitation data files.
        Files should follow naming convention with embedded datetime
        (e.g., 'prefix.YYYYMMDDHHMM.suffix').

    name : str
        Hurricane identifier used to organize data in the hurricane_data dictionary.
        Must be a key that exists in hurricane_data.

    Returns
    -------
    tuple
        A tuple containing two elements:
        - hurricane_data : dict
            Updated dictionary with new precipitation data
        - days : list
            List of datetime objects corresponding to the precipitation measurements
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
    Create a detailed time series plot of daily precipitation data for multiple cities during a hurricane event.

    This function generates a sophisticated visualization that includes precipitation threshold bands,
    city-specific trend lines, peak annotations, and automatically scaled axes. The precipitation data
    is converted from kg/m² to inches for display.

    Parameters
    ----------
    daily_precipitation : dict
        Dictionary containing precipitation data.
        Each city should have the same number of measurements corresponding to the days parameter.

    days : list or array-like
        List of datetime objects representing the dates for the precipitation measurements.
        Must be the same length as each city's precipitation data list.

    name : str
        Name of the hurricane being analyzed. Used in plot title and output filename.

    output_folder : str or Path
        Directory path where the plot will be saved. Must be an existing directory
        with write permissions.

    Returns
    -------
    None
        Saves the plot as a PNG file and closes the figure.
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
    Generate and save a bar plot visualizing the total accumulated precipitation by city for a specified hurricane.

    This function creates a bar chart showing total precipitation amounts for each affected city,
    converting the values from kg/m² to inches. The plot includes value labels, a grid, and
    is automatically scaled to accommodate the data range.

    Parameters
    ----------
    accumulated_precip : dict
        Dictionary containing city-wise precipitation data with structure:

    name : str
        Name of the hurricane being analyzed. Used in plot title and output filename.

    output_folder : str or Path
        Directory path where the plot will be saved. Must be an existing directory
        with write permissions.

    Returns
    -------
    str
        Path to the saved plot file.
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
    Create an MP4 video animation from a sequence of map images.

    This function reads PNG images from a specified folder, resizes them to a standard size (3008x3008),
    and combines them into a video animation. The resulting video is saved as an MP4 file with the
    specified frames per second.

    Parameters
    ----------
    maps_folder : str or Path
        Directory path containing the map images (PNG format) to be animated.
        Images will be processed in alphabetical order.

    animation_folder : str or Path
        Directory path where the output MP4 video will be saved.
        Must be an existing directory with write permissions.

    name : str
        Base name for the output video file. Will be capitalized and appended with '_animation.mp4'.
        Example: if name='ian', output will be 'Ian_animation.mp4'

    fps : int, optional
        Frames per second for the output video animation (default=2).
        Controls the playback speed of the animation:
        - Higher values create faster, smoother animations
        - Lower values create slower animations with more visible transitions

    Returns
    -------
    None
        Prints confirmation message with output file path upon completion.
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
            download_links(name, data_folder,txt_folder)
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
    
        