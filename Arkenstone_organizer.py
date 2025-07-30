# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 19:16:15 2023

@author: Siva Kumar Valluri
"""
import os

def organize_images(folder_address):    
    image_set_names = []
    tiff_images_addresses = []
    
    # Collect image set names and file addresses
    for root, subfolders, filenames in os.walk(folder_address):
        # Ignore the specific folder
        if "saved images for training" in root:
            continue
        
        for filename in filenames:
            # Ignore .txt files
            if filename.lower().endswith(".txt"):
                continue
            
            image_set_names.append(root.rpartition('\\')[2])  # Assumes Windows-style paths
            tiff_images_addresses.append(os.path.join(root, filename))  # Use os.path.join for compatibility
    
    address_set = []
    name_set = []
    run_set = []
    previous_name = None  # Initialize previous_name properly
    
    # Organize image addresses by set names
    for i, current_name in enumerate(image_set_names):
        if current_name != previous_name:  # New set detected
            if run_set:  # Add the previous set if it exists
                address_set.append(run_set)
            name_set.append(current_name)  # Add the new name
            run_set = []  # Reset run_set for the new set
        run_set.append(tiff_images_addresses[i])  # Add current image to the run_set
        previous_name = current_name  # Update previous_name
    
    if run_set:  # Add the last run_set after the loop
        address_set.append(run_set)
    
    sample_name = folder_address.split("/")[-1]   # Using primary folder name as sample name
    
    return sample_name, address_set
