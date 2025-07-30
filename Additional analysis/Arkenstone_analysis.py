# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 08:08:16 2025

@author: Siva Kumar Valluri
"""

"""
Probability of hotspot as a function(particle size) 
"""
import matplotlib.pyplot as plt
import pandas as pd
import os

def data_bin_by_size(data_df):
    # Define bins and calculate bin centers
    bins = [1.3 ** x for x in range(6, 24)]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Create a dictionary to store dataframes for each unique combination of delay and exposure
    results = {}

    # Create unique keys based on delay and half the exposure
    data_df['key'] = data_df['delay'] + data_df['exposure'] / 2

    # Loop through each unique key
    for key in data_df['key'].unique():
        # Filter data for the current key
        key_data = data_df[data_df['key'] == key]

        # Use pandas cut function to bin data and count frequencies
        binned = pd.cut(key_data['particle diameter'], bins=bins, labels=bin_centers, include_lowest=True)
        frequency = binned.value_counts().sort_index()

        # Create a new DataFrame for this key
        result_df = pd.DataFrame({
            'particle diameter': bin_centers,
            'frequency': frequency.reindex(bin_centers, fill_value=0)
        })

        # Include and average values from columns ending with 'K' above 2000K based on particle diameter bins
        K_columns = key_data.columns[key_data.columns.str.endswith('K')]
        for column in K_columns:
            parts = column.split('K-')
            if len(parts) == 2 and all(part.rstrip('K').isdigit() for part in parts):
                low_temp, high_temp = int(parts[0].rstrip('K')), int(parts[1].rstrip('K'))
                if high_temp > 2000:
                    # Calculate the average of the temperature range
                    temp_avg = (low_temp + high_temp) / 2

                    # Group data by the binned category and calculate the mean
                    grouped = key_data.groupby(binned, observed=True)[column]
                    result_df[str(int(temp_avg))] = grouped.mean().reindex(bin_centers, fill_value=0).values
            else:
                print(f"Column '{column}' does not match expected format and will be skipped.")

        # Store the result DataFrame in the dictionary with the key as the key
        results[key] = result_df

    return results

data_df = pd.read_csv(r'C:\Users\sivak\OneDrive\Desktop\multiparticle\4.5 kms\3-350-24\3-350-24-hyperspectral_raw_data.txt', delimiter = "\t")
filtered_df = data_df[data_df['ignition'] == 1]
#results = data_bin_by_size(filtered_df)


def hotspot_probability(df, sample_name, header, folder_address):
    if df.empty:
        print("The DataFrame is empty. Cannot bin particles.")
        return None

    # Ensure the 'delay' column is clean and numeric
    df['delay'] = df['delay'].astype(str).str.strip()
    df['delay'] = pd.to_numeric(df['delay'], errors='coerce')
    print("Initial DataFrame shape:", df.shape)
    print("Unique delays in DataFrame:", df['delay'].unique())

    # Drop rows with NaN in 'delay'
    df = df.dropna(subset=['delay'])
    print("DataFrame shape after removing NaN delays:", df.shape)

    # Check for duplicates
    print("Number of duplicates:", df.duplicated().sum())


    """
    for unique_delay in df['delay'].unique():
        filtered_df = df[df['delay'] == unique_delay]
        print(f"Processing delay {unique_delay}: Total Rows = {filtered_df.shape[0]}")
        ignited_df = filtered_df[filtered_df['ignition'] == 1]
        print(f"Ignited Rows for delay {unique_delay} = {ignited_df.shape[0]}")
    
        # Save filtered data to text files
        # Construct file names using the folder_address
        base_name = sample_name.rpartition('/')[2]  # or use os.path.basename(sample_name)
        all_data_file = os.path.join(folder_address, f"{base_name}_{unique_delay}_all_data_{header}.txt")
        ign_data_file = os.path.join(folder_address, f"{base_name}_{unique_delay}_ign_data_{header}.txt")
        
        # Save filtered data to text files
        filtered_df.to_csv(all_data_file, sep='\t', index=False)
        ignited_df.to_csv(ign_data_file, sep='\t', index=False)
    """
    # Group by delay
    grouped = df.groupby('delay')
    size_distributions = {}

    bins = [1.3 ** x for x in range(6, 24)]  # Define bins with increments of 1 from 6 to 23
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]  # Calculate bin centers

    plt.figure(figsize=(7, 6))  # Initialize plot for size distributions

    for delay, group in grouped:
        # Bin particle sizes
        group['size_bin'] = pd.cut(group['particle diameter'], bins, labels=bin_centers, include_lowest=True)

        # Count total particles and ignited particles in each bin
        size_distribution = group.groupby('size_bin', observed=False).agg(
            total_particles=('particle diameter', 'count'),
            ignited_particles=('ignition', 'sum')
        ).reset_index()

        size_distributions[delay] = size_distribution

        # Save the size distribution to a text file
        file_name = os.path.join(folder_address, f"{sample_name}_delay_{delay}_{header}.txt")
        size_distribution.to_csv(file_name, index=False, sep='\t')

        # Add to scatter plot
        plt.plot(size_distribution['size_bin'], size_distribution['total_particles'], marker='o', label=f"Total - Delay {delay}")
        plt.plot(size_distribution['size_bin'], size_distribution['ignited_particles'], marker='x', linestyle='--', label=f"Ignited - Delay {delay}")

    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')
    plt.xlim(1,400)
    plt.ylabel("frequency (counts)", fontsize=18, fontname='Arial')
    plt.ylim(0,None)
    #plt.title('Shock-initiation', fontsize=18, fontname='Arial', fontweight='bold')
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(prop={'family': 'Arial', 'size': 14})
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    image_name = os.path.join(folder_address, f"{sample_name}_probability_shock_initiation.png")
    plt.savefig(image_name, dpi=700)
    plt.show()
    
    return size_distributions

"""
Hotspot temperature as a function(particle size)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import os

def hotspot_temperatures(data_df, sample_name, save_address):
    # Identify temperature-related columns dynamically
    bin_cols = [col for col in data_df.columns if "K-" in col]
    
    # Compute percentage data per row
    df_unbinned_percentage = data_df.copy()
    row_sums = df_unbinned_percentage[bin_cols].sum(axis=1)
    df_unbinned_percentage[bin_cols] = df_unbinned_percentage[bin_cols].div(row_sums, axis=0) * 100
    df_unbinned_percentage[bin_cols] = df_unbinned_percentage[bin_cols].fillna(0)
    
    delay_data_dict = {}  # Dictionary to store dataframes for each delay
    
    def plot_log_scatter_for_delay_filtered(df, delay):
        """Plots scatter plot of temperatures above 2000K as a function of diameter (log scale) for unbinned data, classified by delay."""
        df_delay = df[df['delay'] == delay].copy()
        
        # Filter out temperature bins below 2000K
        filtered_bin_cols = [col for col in bin_cols if int(col.split("K-")[0]) >= 2000]
    
        fig, ax = plt.subplots(figsize=(7, 6))
        norm = mcolors.Normalize(vmin=0, vmax=100)
        
        for _, row in df_delay.iterrows():
            diameters = [row['particle diameter']] * len(filtered_bin_cols)
            temperatures = [int(col.split("K-")[0]) + 250 for col in filtered_bin_cols]  # Midpoint of bin range
            values = row[filtered_bin_cols].values.astype(float)
    
            ax.scatter(diameters, temperatures, s=values * 10, alpha=0.7, edgecolors='black', c=values, cmap="viridis", norm=norm, marker='o')
        
        legend_sizes = [10, 30, 50, 70, 90]
        legend_handles = [plt.scatter([], [], s=size * 10, alpha=0.7, edgecolors='black', color="gray", marker='o', label=f"{size}%") for size in legend_sizes]
        ax.legend(handles=legend_handles, title="Percentage Contribution", fontsize=8, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        ax.set_xscale("log")
        ax.set_xlim(1, 400)
        ax.set_ylim(2000, 8000)  # Adjusted y-axis lower bound to 2000K
        
        ax.set_xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')  # Font size 20, Arial
        ax.set_ylabel("temperature (K)", fontsize=18, fontname='Arial')
        ax.set_title(f"observation period: {delay}ns", 
                     fontsize=18, fontname='Arial', fontweight='bold')  # Title with bold Arial
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        
        plt.show()
    
    
    def plot_log_box_for_delay_filtered(df, delay):
        """Plots box plot of temperatures above 2000K as a function of diameter (log scale) for unbinned data, classified by delay."""
        df_delay = df[df['delay'] == delay].copy()
    
        # Filter out temperature bins below 2000K
        filtered_bin_cols = [col for col in bin_cols if int(col.split("K-")[0]) >= 2000]
    
        # Prepare data for box plot
        data_to_plot = [df_delay[col].dropna() for col in filtered_bin_cols]
    
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.boxplot(data_to_plot, positions=[int(col.split("K-")[0]) + 250 for col in filtered_bin_cols], notch=True, patch_artist=True)
    
        # Customize x-axis to show the midpoint of bin range
        ax.set_xticklabels([int(col.split("K-")[0]) + 250 for col in filtered_bin_cols], rotation=45)
    
        ax.set_xscale("log")
        ax.set_xlim(1, 400)
        ax.set_ylim(2000, 8000)  # Adjusted y-axis lower bound to 2000K
    
        ax.set_xlabel("temperature (K)", fontsize=18, fontname='Arial')  # Font size 20, Arial
        ax.set_ylabel("Value", fontsize=18, fontname='Arial')
        ax.set_title(f"observation period: {delay}ns", fontsize=18, fontname='Arial', fontweight='bold')  # Title with bold Arial
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
    
        plt.show()
    
    
    def plot_heatmap_for_delay_filtered(df, delay):
        """Plots KDE heatmap for temperature vs. particle diameter for a given delay, only including temperatures above 2000K."""
        df_delay = df[df['delay'] == delay].copy()
        
        # Filter out temperature bins below 2000K
        filtered_bin_cols = [col for col in bin_cols if int(col.split("K-")[0]) >= 2000]
    
        heatmap_x = []
        heatmap_y = []
        heatmap_intensity = []
        
        for _, row in df_delay.iterrows():
            diameters = [row['particle diameter']] * len(filtered_bin_cols)
            temperatures = [int(col.split("K-")[0]) + 250 for col in filtered_bin_cols]
            values = row[filtered_bin_cols].values.astype(float)
            
            heatmap_x.extend(diameters)
            heatmap_y.extend(temperatures)
            heatmap_intensity.extend(values)
        
        fig, ax = plt.subplots(figsize=(7, 6))
        
        sns.kdeplot(
            x=heatmap_x, y=heatmap_y, weights=heatmap_intensity, cmap="Reds", fill=True, 
            alpha=0.7, levels=100, bw_adjust=0.5, ax=ax
        )
        
        ax.set_xscale("log")
        ax.set_xlim(1, 400)
        ax.set_ylim(2000, 8000)
        ax.set_xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')
        ax.set_ylabel("temperature (K)", fontsize=18, fontname='Arial')
        #ax.set_title(f"Delay {delay}", fontsize=18, fontname='Arial', fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
       
        image_name = os.path.join(save_address, f"{sample_name}_hotspot_temperatures_delay_{delay}.png")
        plt.savefig(image_name, dpi=700)
        plt.show()
        
        # Store data in dictionary for each delay
        
        result_df = pd.DataFrame({
            'particle diameter (µm)': heatmap_x,
            'temperature (K)': heatmap_y,
            'intensity': heatmap_intensity
        })
        delay_data_dict[delay] = result_df
    
        file_name = os.path.join(save_address, f"{sample_name}_hotspot_temperatures_delay_{delay}.txt")
        result_df.to_csv(file_name, index=False, sep='\t')
    
    if not df_unbinned_percentage.empty:
        unique_delays = df_unbinned_percentage['delay'].unique()
        for delay in unique_delays:
            plot_heatmap_for_delay_filtered(df_unbinned_percentage, delay)
            plot_log_scatter_for_delay_filtered(df_unbinned_percentage, delay)
            plot_log_box_for_delay_filtered(df_unbinned_percentage, delay)
    
    return delay_data_dict


"""
Area covered by emission as a function(particle size)
    Processes the dataframe to bin reacted area based on particle diameter for each unique delay,
    plots the results with a connected scatter plot, and saves each binned dataset as a text file.

    Parameters:
    - df: DataFrame containing 'particle diameter', 'reacted_area', and 'delay' columns.
    - save_address: Folder path where the binned data files will be saved.

    Returns:
    - A dictionary where keys are unique delays and values are DataFrames with binned statistics:
      'particle_diameter', 'reacted_area_mean', 'reacted_area_mad', 'reacted_area_std'.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


def hotspot_size(df, sample_name, save_address):
    bins = [1.3 ** x for x in range(6, 24)]  # Define bin edges
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]  # Bin centers
    
    # Ensure 'particle diameter' is numeric
    df['particle diameter'] = pd.to_numeric(df['particle diameter'], errors='coerce')
    
    # Initialize dictionary to store results
    binned_data = {}
    
    
    plt.figure(figsize=(7, 6))  # Initialize figure
    
    for delay in df['delay'].unique():
        sub_df = df[df['delay'] == delay].copy()
        sub_df['size_bin'] = pd.cut(sub_df['particle diameter'], bins, labels=bin_centers, include_lowest=True)
        
        # Aggregate statistics for each bin
        bin_stats = sub_df.groupby('size_bin', observed=False).agg(
            reacted_area_mean=('reacted_area', 'mean'),
            reacted_area_mad=('reacted_area', lambda x: np.mean(np.abs(x - np.mean(x)))),
            reacted_area_std=('reacted_area', 'std')
        ).reset_index()
        
        # Rename columns for clarity
        bin_stats.rename(columns={'size_bin': 'particle_diameter'}, inplace=True)
        
        # Save dataframe as a text file
        file_name = os.path.join(save_address, f"{sample_name}_reacted_area_delay_{delay}.txt")
        bin_stats.to_csv(file_name, index=False, sep='\t')
        
        # Store in dictionary
        binned_data[delay] = bin_stats
        
        # Plot with connected scatter points
        plt.errorbar(bin_stats['particle_diameter'], bin_stats['reacted_area_mean'], 
                     yerr=bin_stats['reacted_area_mad'], fmt='-o', capsize=5, label=f"Delay {delay}")

    # Formatting the plot
    plt.xscale('log')  # Set x-axis to log scale
    plt.xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')
    plt.xlim(1,400)
    plt.ylim(0,100)
    plt.ylabel("area covered (%)", fontsize=18, fontname='Arial')
    plt.title('Area covered by reaction across delays', fontsize=18, fontname='Arial', fontweight='bold')
    plt.legend(prop={'family': 'Arial', 'size': 14})
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    image_name = os.path.join(save_address, f"{sample_name}_hs_size.png")
    plt.savefig(image_name, dpi=700)
    plt.show()
    
    return binned_data

"""
Plots the temperature distribution over time using a sampled portion of the dataset with median value annotations.
    
Parameters:
    data (DataFrame): The dataset.
    folder_address (str): Path to save the plot.
    sample_name (str): Sample identifier for saving the image.
    sample_fraction (float): Fraction of data to sample for efficiency (default: 90%).
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature_distribution(data, folder_address, sample_name, sample_fraction=1):
    def extract_midpoint(column_name):
        """
        Extracts the midpoint from a temperature range column name.
        Example: '2000K-2250K' -> 2125
        """
        try:
            parts = column_name.replace('K', '').split('-')
            return (int(parts[0]) + int(parts[1])) / 2
        except (IndexError, ValueError):
            return None

    # Fill missing values with 0
    data.fillna(0, inplace=True)

    # Extract temperature columns
    temperature_columns = [col for col in data.columns if '-' in col and 'K' in col]

    # Compute midpoints for temperature ranges
    mid_points = [extract_midpoint(col) for col in temperature_columns]

    # Ensure valid midpoints
    valid_indices = [i for i, mp in enumerate(mid_points) if mp is not None and mp >= 2000]
    temperature_columns = [temperature_columns[i] for i in valid_indices]
    mid_points = [mid_points[i] for i in valid_indices]

    # Reduce data size by sampling
    sampled_data = data.sample(frac=sample_fraction, random_state=42)

    # Prepare the data for the box plot
    temp_values = []
    adjusted_delays = []
    exposure = sampled_data['exposure'].iloc[0] / 2  # Half of the exposure

    for _, row in sampled_data.iterrows():
        for temp_col, mid_point in zip(temperature_columns, mid_points):
            count = int(row[temp_col])
            if count > 0:
                temp_values.extend([mid_point] * count)
                adjusted_delays.extend([(row['delay'] + exposure)] * count)

    # Create DataFrame for plotting
    boxplot_data = pd.DataFrame({
        'Temperature': temp_values,
        'Adjusted Delay': adjusted_delays
    })

    # Creating the plot
    plt.figure(figsize=(7, 6))
    ax = sns.boxplot(x='Adjusted Delay', y='Temperature', data=boxplot_data, color='white', linewidth=2)
    ax.set_facecolor('none')  # No fill color for plot background

    # Compute median temperatures for each delay time
    medians = boxplot_data.groupby('Adjusted Delay')['Temperature'].median()

    # Annotate median values
    for xtick, median_val in zip(ax.get_xticks(), medians):
        ax.text(xtick, median_val + 50, f'{int(median_val)} K', horizontalalignment='center', color='black', fontsize=20, fontname='Arial')

    ax.set_xlabel('time (ns)', fontsize=24, fontname='Arial')
    ax.set_ylabel('temperature (K)', fontsize=24, fontname='Arial')
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Adding a border around the plot
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)

    # Ensure layout is tight and nothing is cut off
    plt.tight_layout()

    # Save the plot
    #image_name = os.path.join(folder_address, f"{sample_name}_temp_vs_time.png")
    #plt.savefig(image_name, dpi=300, bbox_inches='tight')
    plt.show()

    
"""
################################################################################################################################################
Creating gifs from saved images
The saved images are 
-the individual frames of particles identified in backlit image
-masks overlaid on the emission frame 
-enlarged roi of individual particle as heat map with color bar 
"""
import os
import imageio

def create_gifs_for_parameters(base_dir, name, delay, exposure):
    particles_pattern = f'Particles_{name}_{delay}ns_{exposure}ns_'
    emission_pattern = f'Emissions_{name}_{delay}ns_{exposure}ns_'
    heatmap_pattern = f'Heat_map_{name}_{delay}ns_{exposure}ns_'

    # Function to sort files based on file creation or modification time
    def sort_key_func(filename):
        # Get the full path of the file
        file_path = os.path.join(base_dir, filename)
        # Get the time the file was last modified
        return os.path.getmtime(file_path)

    def create_gif(image_files, output_filename):
        output_path = os.path.join(base_dir, output_filename)
        with imageio.get_writer(output_path, mode='I', duration=1000, loop=0) as writer:
            for filename in sorted(image_files, key=sort_key_func):
                image = imageio.imread(os.path.join(base_dir, filename))
                writer.append_data(image)
        print(f"GIF saved: {output_path}")

    # Collect files that match the specified patterns
    particles_files = [f for f in os.listdir(base_dir) if f.startswith(particles_pattern)]
    emission_files = [f for f in os.listdir(base_dir) if f.startswith(emission_pattern)]
    heatmap_files = [f for f in os.listdir(base_dir) if f.startswith(heatmap_pattern)]

    # Create GIFs for each category
    create_gif(particles_files, f'GIF_Particles_{name}_{delay}ns_{exposure}ns.gif')
    create_gif(emission_files, f'GIF_Emissions_{name}_{delay}ns_{exposure}ns.gif')
    create_gif(heatmap_files, f'GIF_Heat_map_{name}_{delay}ns_{exposure}ns.gif')


# Example usage
#base_directory = r'C:\Users\sivak\OneDrive\Desktop\New folder\saved images for training'  # Adjust this to your directory where images are stored
#name = '1'
#delay = '0'
#exposure = '50'
#create_gifs_for_parameters(base_directory, name, delay, exposure)

"""
###########################################################################################################################################################################################
Correlating nanostructure elements to hyperspectral temperatures
"""
def correlating_temperature_to_nanostructure(data_df, pore_diameter_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Filtering the dataset based on 'delay' and 'ignition'
    unique_delays = data_df['delay'].unique()
    result_df = data_df[data_df["delay"] == unique_delays[0]]
    result_df = result_df[result_df["ignition"] == data_df["ignition"].unique()[1]]

    # Identifying the temperature columns and filtering for >= 2000K
    temperature_bins = data_df.columns[9:]
    filter_bins = [bin for bin in temperature_bins if int(bin.split('-')[0].replace('K', '')) >= 2000]
    bin_labels = [(int(bin.split('-')[0].replace('K', '')) + int(bin.split('-')[1].replace('K', ''))) // 2 for bin in filter_bins]

    # Histogram data extraction for temperature
    histogram_data = result_df[filter_bins].sum()

    # Load and sort pore diameters
    pore_diameters = pd.read_csv(pore_diameter_path, header=None, names=["diameter"]).sort_values(by="diameter")

    # Map each sorted pore diameter to a temperature bin based on its rank
    scaled_indices = np.linspace(0, len(bin_labels) - 1, len(pore_diameters))
    temperatures_mapped = [bin_labels[int(index)] for index in scaled_indices]

    # Create DataFrame to return
    correlation_df = pd.DataFrame({
        "Pore Diameter": pore_diameters['diameter'].values,
        "Temperature": temperatures_mapped
    })

    # Plotting all three aspects
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))

    # Temperature histogram
    axs[0].bar(filter_bins, histogram_data, color='green', edgecolor='black')
    axs[0].set_xlabel('Temperature Bins')
    axs[0].set_ylabel('Sum of Temperatures')
    axs[0].set_title('Histogram of Temperatures for >= 2000K')
    axs[0].grid(True)

    # Pore diameter histogram
    axs[1].hist(pore_diameters['diameter'], bins=np.logspace(np.log10(pore_diameters['diameter'].min()), np.log10(pore_diameters['diameter'].max()), 50), color='blue', edgecolor='black')
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Pore Diameter (Log Scale)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_title('Histogram of Pore Diameters')
    axs[1].grid(True)

    # Correlation plot
    axs[2].scatter(correlation_df['Pore Diameter'], correlation_df['Temperature'], c='red', edgecolor='black')
    axs[2].set_xscale('log')
    axs[2].set_xlabel('Pore Diameter (Log Scale)')
    axs[2].set_ylabel('Temperature (K)')
    axs[2].set_title('Correlation of Pore Diameter to Temperature')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    return correlation_df
"""
data_df = pd.read_csv(read attached file) 
unique_delays = data_df['delay'].unique()
result_df = data_df[data_df["delay"] == unique_delays[0]]
result_df = result_df[result_df["ignition"] == data_df["ignition"].unique()[1]]
comprehensive_df = correlating_temperature_to_nanostructure(data_df, '/mnt/data/pore_diameter.txt')
"""
"""
Plots a box plot of temperature distributions for specified ignition and delay conditions.
    
Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    ignition (int): Ignition condition (1 for ignited, 0 for not ignited).
    delay (int): The delay value to filter the dataframe.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_temperature_vs_particle_size(df, ignition, delay):

    def extract_midpoint(temp_range):
        """Helper function to extract the midpoint from a temperature range string."""
        start, end = temp_range.replace('K', '').split('-')
        return (int(start) + int(end)) / 2
    # Extract temperature columns and compute midpoints
    temperature_columns = [col for col in df.columns if '-' in col and 'K' in col]
    mid_points = [extract_midpoint(col) for col in temperature_columns]

    # Ensure valid midpoints, i.e., >= 2000K
    valid_indices = [i for i, mp in enumerate(mid_points) if mp is not None and mp >= 2000]
    temperature_columns = [temperature_columns[i] for i in valid_indices]
    mid_points = [mid_points[i] for i in valid_indices]
    temperature_midpoints_dict = dict(zip(temperature_columns, mid_points))

    # Define particle diameter bins
    bins = [1.3 ** x for x in range(6, 24)]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Filter the data based on delay and ignition condition
    filtered_df = df[(df['delay'] == delay) & (df['ignition'] == ignition)]
    filtered_df['diameter_bin'] = np.digitize(filtered_df['particle diameter'], bins, right=False)

    # Prepare to expand temperatures into midpoints
    expanded_temps = {center: [] for center in bin_centers}
    test = []
    for _, row in filtered_df.iterrows():
        bin_index = int(row['diameter_bin']) -1
        test.append(bin_index)
        if bin_index < len(bin_centers):
            bin_center = bin_centers[bin_index]
            for temp_col in temperature_columns:
                count = int(row[temp_col])
                expanded_temps[bin_center].extend([temperature_midpoints_dict[temp_col]] * count)

    # Convert to DataFrame for plotting
    expanded_temps_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in expanded_temps.items() if v]))
    plot_data = expanded_temps_df.stack().reset_index()
    plot_data.columns = ['index', 'Particle Diameter (µm)', 'Temperature (K)']
    plot_data['Particle Diameter (µm)'] = pd.Categorical(plot_data['Particle Diameter (µm)'],
                                                         categories=bin_centers, ordered=True)

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=plot_data, x='Particle Diameter (µm)', y='Temperature (K)')
    plt.xscale('log')
    plt.xlabel('Particle Diameter (µm)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Temperature Distribution for Delay {delay} and Ignition {ignition}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()
    return expanded_temps_df


# Example of how to use the function:
#expanded_temps_df = plot_temperature_distribution(data_df, ignition=1, delay=0)
#name = os.path.join(folder_address, f"{sample_name}-TEST-ns.txt") 
#expanded_temps_df.to_csv(name, sep='\t', index=True)

"""
Plots a box plot of reacted area distributions across particle diameter bins
for specified ignition and delay conditions. Also returns a DataFrame with
reacted area values binned by particle size.

Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    ignition (int): Ignition condition (1 for ignited, 0 for not ignited).
    delay (int): The delay value to filter the dataframe.

Returns:
    pd.DataFrame: A DataFrame where each column represents a particle diameter
                  bin (by midpoint), and each row contains individual reacted
                  area values within that bin.
"""

def plot_reacted_area_vs_particle_size(df, ignition, delay):
    # Define particle diameter bins and centers
    bins = [1.3 ** x for x in range(6, 24)]
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # Filter based on ignition and delay
    filtered_df = df[(df['delay'] == delay) & (df['ignition'] == ignition)].copy()
    filtered_df['diameter_bin'] = np.digitize(filtered_df['particle diameter'], bins, right=False)

    # Filter out-of-range bins
    valid_rows = filtered_df[(filtered_df['diameter_bin'] > 0) & (filtered_df['diameter_bin'] <= len(bin_centers))]

    # Map bin index to bin center
    valid_rows['bin_center'] = valid_rows['diameter_bin'].apply(lambda i: bin_centers[i - 1])

    # Group reacted_area values into their respective bins
    bin_data = {center: [] for center in bin_centers}
    for _, row in valid_rows.iterrows():
        bin_data[row['bin_center']].append(row['reacted_area'])

    # Convert to DataFrame with bin centers as columns
    output_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in bin_data.items() if v]))

    # Optional: sort columns by bin center (just in case)
    output_df = output_df.reindex(sorted(output_df.columns), axis=1)

    # Plotting
    plot_df = output_df.stack().reset_index()
    plot_df.columns = ['index', 'Particle Diameter (µm)', 'Reacted Area']
    plot_df['Particle Diameter (µm)'] = pd.Categorical(plot_df['Particle Diameter (µm)'],
                                                       categories=sorted(output_df.columns), 
                                                       ordered=True)

    plt.figure(figsize=(12, 8))
    sns.boxplot(data=plot_df, x='Particle Diameter (µm)', y='Reacted Area')
    plt.xscale('log')
    plt.xlabel('Particle Diameter (µm)')
    plt.ylabel('Reacted Area')
    plt.title(f'Reacted Area Distribution for Delay {delay} and Ignition {ignition}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return output_df