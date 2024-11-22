import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.ticker import AutoLocator
from matplotlib.dates import HourLocator, DateFormatter, DayLocator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csvFiles = os.path.join(BASE_DIR, "Wind_Data")

def create_directories(name):
    """
    Create directory structure for hurricane graphs.
    
    Parameters:
    name (str): Name of the hurricane file
    
    Returns:
    str: Path to the graph directory for this hurricane
    """
    processedName = " ".join(re.findall(r'[A-Z][a-z]*', name)).split()[0]
    graphs_folder = os.path.join(BASE_DIR, "Graphs_Folder", f"{processedName.capitalize()}_Graphs")
    os.makedirs(graphs_folder, exist_ok=True)
    return graphs_folder

def plot_hurricane_intensity(data, hurricane_name, output_dir):
    """
    Create and save intensity plot for a hurricane.
    
    Parameters:
    data (pd.DataFrame): Hurricane data
    hurricane_name (str): Name of the hurricane
    output_dir (str): Directory to save the plot
    """
    # Create new figure
    plt.figure(figsize=(12, 6))
    current_ax = plt.gca()
    
    # Plot the intensity line
    plt.plot(data['Date'], data['Intensity'], 'r-', linewidth=2.5, label='Wind Speed')
    
    # Add hurricane category thresholds
    categories = [
        (64, 82, '1'),
        (83, 95, '2'),
        (96, 112, '3'),
        (113, 136, '4'),
        (137, 200, '5')
    ]
    
    for i, (min_speed, max_speed, cat) in enumerate(categories):
            plt.axhspan(min_speed, max_speed, alpha=0.1, color=f'C{i}', 
                        label=f'Category {cat}')
        
    start_date = data['Date'].min().strftime('%B %d %H:%M')
    end_date = data['Date'].max().strftime('%B %d, %Y %H:%M')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Hurricane {hurricane_name} Intensity Timeline\n{start_date} - {end_date}', 
                fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Wind Speed (knots)', fontsize=12)
    
    # Use daily ticks for major intervals and hourly ticks for minor intervals
    current_ax.xaxis.set_major_locator(DayLocator())
    current_ax.xaxis.set_minor_locator(HourLocator(interval=3))

    # Format the major ticks (days) to include time
    current_ax.xaxis.set_major_formatter(DateFormatter('%m/%d\n%H:%M'))
    
    # Rotate x-axis labels for better readability
    plt.tick_params(axis='x', rotation=45)
    
    # Add legend
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add peak intensity annotation
    max_intensity = data.loc[data['Intensity'].idxmax()]
    plt.annotate(f'Peak Intensity: {int(max_intensity["Intensity"])} kt',
                xy=(max_intensity['Date'], max_intensity['Intensity']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"{hurricane_name}_intensity.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def process_file(file_path):
    """
    Process a single hurricane data file.
    
    Parameters:
    file_path (str): Path to the CSV file
    """
    try:
        # Extract hurricane name from file name
        hurricane_name = " ".join(re.findall(r'[A-Z][a-z]*', os.path.basename(file_path))).split()[0]
        # print(f"Processing hurricane {hurricane_name}...")
        
        # Create output directory
        output_dir = create_directories(os.path.basename(file_path))
        
        # Read and process data
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Create and save plot
        output_path = plot_hurricane_intensity(data, hurricane_name, output_dir)
        # print(f"Created plot: {output_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        raise

def main():
    """
    Main function to process all hurricane data files.
    """
    # Create main graphs folder if it doesn't exist
    if not os.path.exists('Graphs_Folder'):
        os.makedirs('Graphs_Folder')
    
    # Process all CSV files
    for file_name in os.listdir(csvFiles):
        if file_name.endswith('.csv'):
            file_path = os.path.join(csvFiles, file_name)
            process_file(file_path)

if __name__ == "__main__":
    main()