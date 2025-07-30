# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:09:18 2024

@author: Siva Kumar Valluri
"""
import tkinter as tk
import os
from Arkenstone_GUI import Arkenstone_GUI
from Arkenstone_organizer import organize_images
import pandas as pd
import gc
import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
# Check versions and CUDA availability
print("PyTorch version being used:", torch.__version__)
print("Torchvision version being used:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

root = tk.Tk()
gui = Arkenstone_GUI(root)
root.mainloop()

folder_address, emission_type, number_of_frames, frame_details, objective_used_value, selected_model,selected_processing_style, save_images_choice = gui.get_results()
sample_name, address_set = organize_images(folder_address)

####Creating a saved images folder if directed to##
if save_images_choice.lower() == 'y':
    base_folder = os.path.join(folder_address, "saved images for training")
    
    # Check if the folder exists, if so, create a new incremented folder
    if os.path.exists(base_folder):
        counter = 2
        new_folder = f"saved images-{counter}"
        save_address = os.path.join(folder_address, new_folder)
        
        while os.path.exists(save_address):  # Keep incrementing if the folder exists
            counter += 1
            new_folder = f"saved images-{counter}"
            save_address = os.path.join(folder_address, new_folder)
    else:
        save_address = base_folder

    os.makedirs(save_address, exist_ok=True)  # Create the determined folder

else:
    save_address = []


##Optical Emission Analysis####################################################################################################################################################################
if emission_type == "Optical Emissions":
    
    header = 'optical'
    from Arkenstone_RFC import Arkenstone_RFC
    from Arkenstone_SAM import Arkenstone_SAM
    
    columns = ['run_no', 'frame_number', 'delay', 'exposure', 'particle diameter', 'centroidX', 'centroidY', 'ignition']
    data_df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0, 0]], columns=columns)
    
    if selected_model == "Segment Anything Model":
        sam_checkpoint = r'C:\Users\sivak\OneDrive\Desktop\backup\Arkenstone\sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
        print(f"Segmentation running on {device}... Performance will be {'slow' if device == 'cpu' else 'fast'}")
    
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    
        mask_generator_ = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side = 32 if device == "cpu" else 64,  # Reduce points for CPU mode
            pred_iou_thresh = 0.86,
            stability_score_thresh = 0.96,
            crop_n_layers = 1,
            crop_n_points_downscale_factor = 2,
            min_mask_region_area = 100  # Adjust minimum mask size to prevent over-processing
            )
    
        del sam_checkpoint
    
    for images_in_run in address_set:
        name = images_in_run[0].rpartition('\\')[0].rpartition('\\')[2]
        if selected_model == "Segment Anything Model":
            df = Arkenstone_SAM(frame_details, images_in_run, objective_used_value,mask_generator_ ,save_address, name, 'n') 
            data_df  = pd.concat([data_df , df], ignore_index=True)
        else:
            df = Arkenstone_RFC(frame_details, images_in_run)
            data_df  = pd.concat([data_df , df], ignore_index=True)
        
        del df
        gc.collect()
    
    from Arkenstone_analysis import hotspot_probability
    size_distributions = hotspot_probability(data_df,sample_name, header, folder_address)


#Hyperspectral Emission Analysis################################################################################################################################################
elif emission_type == "Hyperspectral Emissions":
    
    header='hyperspectral'
    from Arkenstone_Hyperspectral_GUI import Arkenstone_Hyperspectral_GUI
    from Arkenstone_Hyperspectral import Arkenstone_Hyperspectral
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    root2 = tk.Tk()
    gui2 = Arkenstone_Hyperspectral_GUI(root2)
    root2.mainloop()  # Run the UI event loop
    
    # Retrieve the results after the GUI closes
    hyperspectral_user_choices = gui2.result
    calibration_lamp_path = hyperspectral_user_choices["calibration_lamp"]
    dark_field_path = hyperspectral_user_choices["dark_field"]
    
    sam_checkpoint = r'C:\Users\sivak\OneDrive\Desktop\backup\Arkenstone\sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Segmentation running on {device}... Performance will be {'slow' if device == 'cpu' else 'fast'}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side = 32 if device == "cpu" else 64,  # Reduce points for CPU mode
        pred_iou_thresh = 0.86,
        stability_score_thresh = 0.96,
        crop_n_layers = 1,
        crop_n_points_downscale_factor = 2,
        min_mask_region_area = 100  # Adjust minimum mask size to prevent over-processing
        )

    del sam_checkpoint
    
    
#####If calibration data is available##
    if calibration_lamp_path and dark_field_path:
        tiff_images_addresses = []
        image_set_names = []
        for root, subfolders, filenames in os.walk(calibration_lamp_path):
            for filename in filenames:
                image_set_names.append(root.rpartition('\\')[2])
                tiff_images_addresses.append(root + "/" + filename)

        tiff_images_addresses2 = []
        image_set_names2 = []
        for root, subfolders, filenames in os.walk(dark_field_path):
            for filename in filenames:
                image_set_names2.append(root.rpartition('\\')[2])
                tiff_images_addresses2.append(root + "/" + filename)
        
                
        #Blackbody detail
        def Intensity_Ratio(T, lambda1 = 0.5, lambda2 = 0.725):
            #constants
            kB = 1.380649*10**(-23)          # m2 kg s-2 K-1  Boltzman's constant
            h = 6.62607015*10**(-34)         # m2 kg / s      Plank's constant
            c = 299792458                    # m/s            Speed of light
    
            Intensity1_blackbody = (((2*h*(c**2))/(lambda1*10**(-6))**5)*(1/(np.exp((h*c)/((lambda1*10**(-6))*kB*T))-1)))/(10^6)
            Intensity2_blackbody = (((2*h*(c**2))/(lambda2*10**(-6))**5)*(1/(np.exp((h*c)/((lambda2*10**(-6))*kB*T))-1)))/(10^6)
            Intensityratio_blackbody = Intensity1_blackbody/Intensity2_blackbody
            return Intensityratio_blackbody
        
        T_calib = 2970                      # K              Calibration Tungsten lamp Temperature
        Intensityratio_blackbody_1 = Intensity_Ratio(T_calib)
                
        calibration_constants =[]
        for i in range(0,int(number_of_frames/2),1):
            image_B = cv2.imread(tiff_images_addresses[i],cv2.IMREAD_GRAYSCALE)
            image_R = cv2.imread(tiff_images_addresses[i+int(number_of_frames/2)],cv2.IMREAD_GRAYSCALE)    
            #image_B_D = cv2.imread(tiff_images_addresses[i],cv2.IMREAD_GRAYSCALE)
            #image_R_D = cv2.imread(tiff_images_addresses[i+int(number_of_frames/2)],cv2.IMREAD_GRAYSCALE)  
            R_av = np.mean(image_R) #- np.mean(image_R_D) 
            B_av = np.mean(image_B) #- np.mean(image_B_D) 
            Intensityratio_calibration = R_av/B_av
            calibration_constant = Intensityratio_calibration/Intensityratio_blackbody_1
            calibration_constants.append(calibration_constant)

#####If calibration data is not available##
    else:
        calibration_constants = 2.829*np.ones((4,), dtype=float)
        #Blackbody detail
        def Intensity_Ratio(T, lambda1 = 0.5, lambda2 = 0.725):
            #constants
            kB = 1.380649*10**(-23)          # m2 kg s-2 K-1  Boltzman's constant
            h = 6.62607015*10**(-34)         # m2 kg / s      Plank's constant
            c = 299792458                    # m/s            Speed of light
    
            Intensity1_blackbody = (((2*h*(c**2))/(lambda1*10**(-6))**5)*(1/(np.exp((h*c)/((lambda1*10**(-6))*kB*T))-1)))/(10^6)
            Intensity2_blackbody = (((2*h*(c**2))/(lambda2*10**(-6))**5)*(1/(np.exp((h*c)/((lambda2*10**(-6))*kB*T))-1)))/(10^6)
            Intensityratio_blackbody = Intensity1_blackbody/Intensity2_blackbody
            return Intensityratio_blackbody
    
    #Hyperspectral processing
    data_df = pd.DataFrame()
    for images_in_run in address_set:
        name = images_in_run[0].rpartition('\\')[0].rpartition('\\')[2]
        df = Arkenstone_Hyperspectral(name, frame_details, images_in_run, objective_used_value,mask_generator_ ,calibration_constants, Intensity_Ratio, save_address, 'n') 
        data_df  = pd.concat([data_df , df], ignore_index=True)
    
    #Additional analysis based on user's choice
    file_name = os.path.join(folder_address, f"{sample_name}-hyperspectral_raw_data.txt")
    data_df.to_csv(file_name, sep='\t', index=False)
    
    #Getting distribution of temperatures as a function of size for each delay chosen
    #from Arkenstone_analysis import data_bin_by_size
    #results = data_bin_by_size(data_df)
    #for key, df in results.items():
        # Format the filename with the delay key and ensure it is a string if it's numeric
        #df = df.drop('frequency', axis=1)
        #df_transposed = df.set_index('particle diameter').T
        #name = os.path.join(folder_address, f"{sample_name}-temp-dist-delay-{str(key)}ns.txt") 
        #df_transposed.to_csv(name, sep='\t', index=True)
        
    from Arkenstone_analysis import plot_temperature_vs_particle_size 
    from Arkenstone_analysis import plot_reacted_area_vs_particle_size   
    unique_delays = data_df['delay'].unique()
    for delay in unique_delays:
        temps_df = plot_temperature_vs_particle_size(data_df, 1, delay)
        areas_df = plot_reacted_area_vs_particle_size(data_df, 1, delay) 
        name = os.path.join(folder_address, f"{sample_name}-temperature vs size-{delay}ns.txt") 
        name2 = os.path.join(folder_address, f"{sample_name}-reacted area vs size-{delay}ns.txt")         
        temps_df.to_csv(name, sep='\t', index=False)
        areas_df.to_csv(name2, sep='\t', index=False)
            
    from Arkenstone_analysis import plot_temperature_distribution
    plot_temperature_distribution(data_df, folder_address, sample_name)
    
    if hyperspectral_user_choices['hotspot_ignition']:
        print("Analyzing probability of shock-initiated ignition of particles as a function of size")
        from Arkenstone_analysis import hotspot_probability
        probability_shock_initiation_data = hotspot_probability(data_df,sample_name, header, folder_address)

    if hyperspectral_user_choices['hotspot_temperature']:
        print("Analyzing distribution of hotspot temperatures in shock-initiated particles as a function of size")
        from Arkenstone_analysis import hotspot_temperatures
        temperature_data = hotspot_temperatures(data_df, sample_name, folder_address)
        
    if hyperspectral_user_choices['track_size']:
        print("Analyzing particle area covered by emission in shock-initiated particles as a function of size")
        from Arkenstone_analysis import hotspot_size
        reaction_size_data = hotspot_size(data_df, sample_name, folder_address)