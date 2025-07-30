# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:34:31 2024

@author: Primary-Siva Kumar Valluri using Secondary-Digitalsreeni (Sreenivas Bhattiprolu) 
"""
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from numpy import inf

def Arkenstone_Hyperspectral(name, frame_details, images_in_run, objective_used_value, mask_generator_ , calibration_constants, Intensity_Ratio, save_address, plotting_choice='n', saved_individual_images = 'y'):

    if objective_used_value == 10:
        scale = 0.63
    
    # Create a DataFrame for results
    df_rows = []
    
    # Filter frame details
    df_frame = pd.DataFrame(frame_details, columns=["delay", "exposure", "use_frame", "red_filter"])
    filtered_df = df_frame[df_frame["use_frame"]]
    detail_df = filtered_df[filtered_df["red_filter"]]
    
    #Getting static image for particle detail
    stat_image_index = filtered_df[filtered_df["red_filter"]].index[0]
    static_image = cv2.imread(images_in_run[stat_image_index], cv2.IMREAD_GRAYSCALE)
    normalized_img = cv2.normalize(static_image, None, 0, 255, cv2.NORM_MINMAX)
    static_image_rgb = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)
    print(f'Identifying Particles in run {name}...')
    masks = mask_generator_.generate(static_image_rgb)
    
    # Read, blurr and classify images into red and blue filtered images
    emission_images_red = []
    emission_images_blue = []
    for image_number in range(len(images_in_run)):
        a = 5
        kernel = np.ones((a,a),np.float32)/(a**2)
        if image_number > len(frame_details)-1:
            if frame_details[image_number - len(frame_details)][2] and frame_details[image_number - len(frame_details)][3]:
                emission_fr = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                blurred_image = cv2.filter2D(emission_fr,-1,kernel)
                emission_images_red.append(blurred_image)
            if frame_details[image_number - len(frame_details)][2] and not frame_details[image_number - len(frame_details)][3]:
                emission_fr = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                blurred_image = cv2.filter2D(emission_fr,-1,kernel)
                emission_images_blue.append(blurred_image)
    
    #Getting Hyperspectral maps
    T = np.arange(1500,8000,1)
    Intensityratio_blackbody = Intensity_Ratio(T)
    Hyperspectral_frames = []
    for red, blue, const in zip(emission_images_red, emission_images_blue, calibration_constants):
        with np.errstate(divide='ignore', invalid='ignore'):
            Ratio_image = np.nan_to_num(red / blue, nan=0, posinf=0, neginf=0)
            T_image = np.interp(Ratio_image / const, Intensityratio_blackbody, T)
        Hyperspectral_frames.append(T_image)
    
    def percentage_above_threshold(counts, bin_edges, threshold=2000):
        idx = np.searchsorted(bin_edges, threshold, side='right') - 1
        value = (counts[idx + 1:].sum() / counts.sum()) * 100 if counts.sum() > 0 else 0
        return value
         
    for frame_index, H_image in enumerate(Hyperspectral_frames):          
        # Prepare a copy of the emission image as a figure-compatible image
        em_image_copy = H_image.copy()
        st_image_copy = static_image_rgb
        
        for mask_index, mask in enumerate(masks):
            area = mask['area'] * scale**2
            if area < 20000 * scale**2:
                x_min, y_min, width, height = map(int, mask['bbox'])
                x_max, y_max = x_min + width, y_min + height
    
                # Draw bounding box on the emission image copy
                color = (0, 255, 0)  # Green color in BGR
                thickness = 2
                cv2.rectangle(em_image_copy, (x_min, y_min), (x_max, y_max), color, thickness)
                cv2.rectangle(st_image_copy, (x_min, y_min), (x_max, y_max), color, thickness)

                roi = H_image[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue
                
                vmin, vmax = 1500, 8000 #K
                bin_size = 250
                bins = np.arange(vmin, vmax + bin_size, bin_size)
        
                # Compute properties for naming and recording results
                flattened_data = roi.flatten()
                counts, bin_edges = np.histogram(flattened_data, bins=bins) # Gets counts of all T values seen withon ROI
                delay = filtered_df['delay'].iloc[frame_index]
                exposure = detail_df['exposure'].iloc[frame_index] 
                particle_diameter = 2 * (area / 3.14) ** 0.50
                centroidX = x_min + width // 2
                centroidY = y_min + height // 2
                diameter = round(particle_diameter, 2)
                ignition_state = 1 if np.mean(flattened_data > 2200) >= 0.05 else 0
                reacted_area = percentage_above_threshold(counts, bin_edges)
                # Create bin count values dynamically
                bin_counts = {f'{int(bin_edges[i])}K-{int(bin_edges[i+1])}K': counts[i] for i in range(len(counts))}

                #Plot detail
                if plotting_choice == 'y':
                    # Convert em_image_copy to uint8 before converting to RGB**
                    em_image_uint8 = cv2.normalize(em_image_copy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    em_image_rgb = cv2.cvtColor(em_image_uint8, cv2.COLOR_BGR2RGB)
                    # Create an overlay image for ROI visualization
                    roi_overlay = em_image_rgb.copy()
                    cv2.rectangle(roi_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)
                    # Preserve aspect ratio when resizing
                    aspect_ratio = roi.shape[0] / roi.shape[1]
                    new_height = int(600 * aspect_ratio)
                    resized_image = cv2.resize(roi, (600, new_height), interpolation=cv2.INTER_AREA)
                    #Plots with station image, temperature map and ROI isolated
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(st_image_copy)
                    axes[0].set_title(f'Particles in run {name}')
                    axes[0].axis('off')
                    axes[1].imshow(roi_overlay)
                    axes[1].set_title(f'Emissions at {delay}ns with an exposure of {exposure}ns')
                    axes[1].axis('off')
                    img3 = axes[2].imshow(resized_image, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
                    axes[2].set_title(f'Heat map of {diameter}µm sized particle')
                    axes[2].axis('off')        
                    fig.tight_layout(pad=1.0)
                    fig.subplots_adjust(right=0.85)  # Create space for colorbar
                    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjust colorbar position
                    cbar = fig.colorbar(img3, cax=cbar_ax)
                    cbar.set_label('Temperature (K)', fontsize=20)  # Adjust label font size
                    cbar.ax.tick_params(labelsize=15)
                    plt.show()
                    
                    """
                    #Only use for presentation purposes. 
                    if save_address:
                        if saved_individual_images == 'y':   
                            # Save the images
                            # Save the images with titles
                            def save_image_with_title(image, title, filename, cmap=None, vmin=None, vmax=None):
                                fig, ax = plt.subplots()
                                # Check if a color map is specified and apply it along with its range (vmin and vmax)
                                if cmap:
                                    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
                                    plt.colorbar(im, ax=ax)  # Add a colorbar to the plot
                                else:
                                    ax.imshow(image)
                                ax.set_title(title, fontdict={'fontsize': 14, 'fontname': 'Arial'})
                                ax.axis('off')
                                fig.tight_layout()
                                fig.savefig(os.path.join(save_address, filename))
                                plt.close(fig)
                    
                            save_image_with_title(st_image_copy, f'Particles in run-{name}', f'Particles_{name}_{delay}ns_{exposure}ns_{diameter}.png')
                            save_image_with_title(roi_overlay, f'Emission frame at {delay}ns with an exposure of {exposure}ns', f'Emissions_{name}_{delay}ns_{exposure}ns_{diameter}.png')
                            save_image_with_title(resized_image, f'Heat map of {diameter}µm sized particle', f'Heat_map_{name}_{delay}ns_{exposure}ns_{diameter}.png', cmap='inferno', vmin=vmin, vmax=vmax) 
                    """
                #Save the resized masked emission image if save_address is provided (non-empty)
                if save_address:
                    if not os.path.exists(save_address):  # Create directory if it doesn't exist
                        os.makedirs(save_address)
                    # Create a new figure with only the resized image and color bar
                    resized_image = cv2.resize(roi, (600, new_height), interpolation=cv2.INTER_AREA)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    img = ax.imshow(resized_image, cmap='inferno', vmin=vmin, vmax=vmax, interpolation='nearest', aspect='equal')
                    ax.axis('off')
                    cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Temperature (K)', fontsize=12)    
                    cbar.ax.tick_params(labelsize=12)
                    # Format the filename: exposure-particle diameter-centroidX-centroidY-ignition.png
                    filename = f"{delay}-{exposure}-{round(particle_diameter, 2)}-{centroidX}-{centroidY}-{ignition_state}.png"
                    file_path = os.path.join(save_address, filename)
                    # Save only the resized image with color bar
                    fig.savefig(file_path, bbox_inches='tight', dpi=700)
                    plt.close(fig)  # Close the figure to free memory
                    print(f"Saved image with color bar to {file_path}")

                # Add details to the DataFrame
                df_rows.append({
                    'run_no': name,
                    'frame_number': frame_index + 1,
                    'delay': delay,
                    'exposure': exposure,
                    'particle diameter': particle_diameter,
                    'centroidX': centroidX,
                    'centroidY': centroidY,
                    'ignition': ignition_state,
                    'reacted_area': reacted_area,
                    **bin_counts
                })
                
                

    df = pd.DataFrame(df_rows)
    return df

