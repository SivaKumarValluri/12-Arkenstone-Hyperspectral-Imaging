# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:07:26 2025

@author: Siva Kumar Valluri

Additional plotting / gif generation scripts using processed txt files / images
"""
"""
Getting histograms of temperatures observed for each row (particle diameter) for an observation period (delay)
-Read txt file "{sample name}-hyperspectral_raw_data.txt" as dataframe data_df
"""
import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os

def plot_histograms_for_particle_sizes(df):
    # Extract temperature bins
    temperature_bins = df.columns[2:]  # Assuming the first two columns are 'particle diameter' and 'reacted_area'

    # Convert temperature bins to midpoints for labels
    bin_labels = [int(bin.split('-')[0].replace('K', '')) for bin in temperature_bins]

    # Directory for saving frames
    frames_dir = 'hist_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    filenames = []

    # Plot histograms for each particle size
    for index, row in df.iterrows():
        particle_diameter = round(row['particle diameter'], 0)
        area = round(row['reacted_area'], 0)
        plt.figure(figsize=(7, 6))
        plt.bar(bin_labels, row[temperature_bins].values, width=250, color='darkgrey', alpha=0.7, edgecolor='black')
        plt.xlabel("temperature (K)", fontsize=18, fontname='Arial')
        plt.ylabel("frequency (counts)", fontsize=18, fontname='Arial')
        plt.title(f"Total emitting area: {area} % (diameter: {particle_diameter} µm)", fontsize=18, fontname='Arial')
        plt.xticks(bin_labels, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 200)
        
        # Save frame
        filename = f'{frames_dir}/frame_{index}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        filenames.append(filename)

    # Create GIF
    with imageio.get_writer('particle_sizes_histogram.gif', mode='I', duration=1000, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        

import matplotlib.pyplot as plt
import pandas as pd
import imageio
import os

def plot_percentage_temperature_coverage(df):
    # Extract temperature bins
    temperature_bins = df.columns[2:]  # Assuming the first two columns are 'particle diameter' and 'reacted_Area'

    # Convert temperature bins to midpoints for labels
    bin_labels = [int(bin.split('-')[0].replace('K', '')) for bin in temperature_bins]
    labels = [label if i % 2 == 0 else '' for i, label in enumerate(bin_labels)]
    # Directory for saving frames
    frames_dir = 'hist_frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    filenames = []

    # Plot histograms for each particle size
    for index, row in df.iterrows():
        particle_diameter = round(row['particle diameter'], 0)
        area = round(row['reacted_area'], 0)
        total_counts = row[temperature_bins].sum()
        frequency_percentages = (row[temperature_bins] / total_counts) * 100  # Convert counts to percentages

        plt.figure(figsize=(7, 6))
        plt.bar(bin_labels, frequency_percentages, width=250, color='darkgrey', alpha=0.7, edgecolor='black')
        plt.xlabel("temperature (K)", fontsize=18, fontname='Arial')
        plt.ylabel("relative area coverage (%)", fontsize=18, fontname='Arial')
        plt.title(f"Total emitting area: {area} % (diameter: {particle_diameter} µm)", fontsize=18, fontname='Arial')
        plt.xticks(bin_labels, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 50)  # Adjusted to max 100% for percentages
        plt.xlim(2000, 8000)  # Set x-axis limits from 2000 to 8000
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        filtered_labels = [label for label in labels if isinstance(label, int) and label >= 2000]
        filtered_label_positions = filtered_labels  # Adjust if needed based on label positioning
        
        ax = plt.gca()
        ax.set_xticks(filtered_label_positions)  # Correct method using the Axes object
        ax.set_xticklabels(filtered_labels, rotation=45)  # Adding rotation to xticklabels for clarity
        # Save frame
        filename = f'{frames_dir}/frame_{index}.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

        filenames.append(filename)

    # Create GIF
    with imageio.get_writer('particle_sizes_percentage_histogram.gif', mode='I', duration=1000, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

        # Optional: Remove frames after creating gif
        for filename in filenames:
            os.remove(filename)


# Load data from file (assuming you have loaded the data into a DataFrame)
# Change address as needed
data_df = pd.read_csv(r'C:\Users\sivak\OneDrive\Desktop\4.5 kms\3-250-24\3-250-24-hyperspectral_raw_data.txt', sep='\t')
unique_delays = data_df['delay'].unique()
#This style has all particles for one delay
#result_df = data_df[data_df["delay"] == unique_delays[0]] # Getting data for first delay setting
#This style has all delays for one particle
result_df = data_df[(data_df['centroidX']== 817) & (data_df['centroidY']== 329)] #specific particle identified by centroid location
result_df = result_df.drop(['run_no', 'frame_number', 'delay', 'exposure', 'centroidX', 'centroidY', 'ignition'], axis=1)
plot_histograms_for_particle_sizes(result_df)
plot_percentage_temperature_coverage(result_df)

"""
Plotting gif histograms particle-size distributions and probability of hotspots as a function of size
-Read txt file "{sample name}-hyperspectral_raw_data.txt" as dataframe data_df
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Using imageio version 2 explicitly
import os
import pandas as pd

def generate_histograms(df, delay):
    frames_dir = f'hist_frames_{delay}ns'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # Prepare bins and bin centers
    bins = [1.3 ** x for x in range(6, 24)]
    bin_centers = [round((bins[i] + bins[i + 1]) / 2, 1) for i in range(len(bins) - 1)]
    
    # Bin data
    df = df.copy()
    df['bin_index'] = pd.cut(df['particle diameter'], bins, labels=range(len(bins)-1))
    
    filenames = []
    ratio_filenames = []
    try:
        for frame in range(len(df)):
            sub_df = df.iloc[:frame+1].copy()  # Include data up to current frame
            
            # Counts for all particles
            all_counts = sub_df['bin_index'].value_counts().reindex(range(len(bins)-1), fill_value=0)
            
            # Counts for ignited particles only
            ignited_df = sub_df[sub_df['ignition'] == 1].copy()
            ignited_counts = ignited_df['bin_index'].value_counts().reindex(range(len(bins)-1), fill_value=0)
            
            # Histogram of all and ignited particles
            plt.figure(figsize=(7,6))
            plt.bar(bin_centers, all_counts, width=np.diff(bins)*0.8, color='black', label='All particles')
            plt.bar(bin_centers, ignited_counts, width=np.diff(bins)*0.8, color='red', label='Ignited particles', alpha=0.7)
            plt.title(f'PSD observed at {delay} ns', fontsize=18, fontname='Arial')
            plt.xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')
            plt.xscale('log')
            plt.ylabel("frequency (counts)", fontsize=18, fontname='Arial')
            plt.xlim([1, 400])
            plt.legend(prop={'family': 'Arial', 'size': 14})
            plt.xticks(fontsize=14, font='Arial')
            plt.yticks(fontsize=14, font='Arial')
            
            filename = os.path.join(frames_dir, f'frame_{frame}.png')
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)

            # Scatter plot of ignited/all particle ratio
            plt.figure(figsize=(7,6))
            ratios = ignited_counts / all_counts.clip(lower=1)  # Avoid division by zero
            plt.plot(bin_centers, ratios, 'o-', color='blue', alpha=0.7)  # Scatter connected by lines
            plt.title(f'Ignition probability at {delay} ns', fontsize=18, fontname='Arial')
            plt.xlabel("particle diameter (µm)", fontsize=18, fontname='Arial')
            plt.xscale('log')
            plt.ylabel("probability of hotspot formation", fontsize=18, fontname='Arial')
            plt.ylim([0, 1])
            plt.xlim([1, 400])
            plt.xticks(fontsize=14, font='Arial')
            plt.yticks(fontsize=14, font='Arial')
            
            ratio_filename = os.path.join(frames_dir, f'ratio_frame_{frame}.png')
            plt.savefig(ratio_filename, bbox_inches='tight')
            plt.close()
            ratio_filenames.append(ratio_filename)
        
        # Create GIFs
        with imageio.get_writer(f'{delay}ns_particles_distribution.gif', mode='I', duration=0.01, loop=0) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        with imageio.get_writer(f'{delay}ns_ratio_distribution.gif', mode='I', duration=0.01, loop=0) as writer:
            for filename in ratio_filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    finally:
        # Cleanup: attempt to remove files and directory
        for filename in filenames + ratio_filenames:
            try:
                os.remove(filename)
            except Exception as e:
                print(f"Error removing file {filename}: {e}")
        try:
            os.rmdir(frames_dir)
        except OSError as e:
            print(f"Error removing directory {frames_dir}: {e}")

# Assuming 'data_df' is already defined and properly structured with a 'particle diameter', 'ignition', and 'delay' column
# Change address as needed
data_df = pd.read_csv(r'C:\Users\sivak\OneDrive\Desktop\4.5 kms\3-250-24\3-250-24-hyperspectral_raw_data.txt', sep='\t')
unique_delays = data_df['delay'].unique()
for delay in unique_delays:
    data_df = data_df[data_df["delay"] == delay]
    generate_histograms(data_df, delay)

