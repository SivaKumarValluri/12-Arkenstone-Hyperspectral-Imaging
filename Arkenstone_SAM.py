
"""
Created on Mon Dec 30 00:34:31 2024

@author: Primary-Siva Kumar Valluri using Secondary-Digitalsreeni (Sreenivas Bhattiprolu) 

First make sure pytorch and torchcvision are installed, for GPU
https://pytorch.org/get-started/locally/

pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything.git'

OR download the repo locally and install
and:  pip install -e .

Download the default trained model: 
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

Other models are available:
    https://github.com/facebookresearch/segment-anything#model-checkpoints


#There are several tunable parameters in automatic mask generation that control 
# how densely points are sampled and what the thresholds are for removing low 
# quality or duplicate masks. Additionally, generation can be automatically 
# run on crops of the image to get improved performance on smaller objects, 
# and post-processing can remove stray pixels and holes. 
# Here is an example configuration that samples more masks:
#https://github.com/facebookresearch/segment-anything/blob/9e1eb9fdbc4bca4cd0d948b8ae7fe505d9f4ebc7/segment_anything/automatic_mask_generator.py#L35    

#Rerun the following with a few settings, ex. 0.86 & 0.9 for iou_thresh
# and 0.92 and 0.96 for score_thresh

Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:

segmentation : the mask
area : the area of the mask in pixels
bbox : the boundary box of the mask in XYWH format
predicted_iou : the model's own prediction for the quality of the mask
point_coords : the sampled input point that generated this mask
stability_score : an additional measure of mask quality
crop_box : the crop of the image used to generate this mask in XYWH format

"""
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import pygame
import time
import gc

def Arkenstone_SAM(frame_details, images_in_run, objective_used_value, mask_generator_ , save_address, name, sound_choice):

    if objective_used_value == 10:
        scale = 0.63
    
    # Create a DataFrame for results
    df = pd.DataFrame(columns=['run_no', 'frame_number', 'delay', 'exposure', 'particle diameter', 'centroidX', 'centroidY', 'ignition'])

    # Filter frame details
    df_frame = pd.DataFrame(frame_details, columns=["delay", "exposure", "choice"])
    filtered_df = df_frame[df_frame["choice"]]

    static_images = []
    emission_images = []

    # Read and preprocess images
    for image_number in range(len(images_in_run)):
        if image_number < len(frame_details):
            if frame_details[image_number][2]:
                static_image = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                normalized_img = cv2.normalize(static_image, None, 0, 255, cv2.NORM_MINMAX)
                static_image_rgb = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)
                static_images.append(static_image_rgb)
        else:
            if frame_details[image_number - len(frame_details)][2]:
                emission_fr = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                emission_image = cv2.normalize(emission_fr, None, 0, 255, cv2.NORM_MINMAX)
                emission_images.append(emission_image)

    # Process images and generate masks
    for frame_index, (st_image, em_image) in enumerate(zip(static_images, emission_images)):
        masks = mask_generator_.generate(st_image)

        # Prepare a copy of the emission image
        em_image_copy = cv2.cvtColor(em_image, cv2.COLOR_GRAY2BGR)
        st_image_copy = st_image.copy()
        
        if sound_choice == 'y':
            # Beeping to get attention: Get system audio device
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            # Save the original volume level
            original_volume = volume.GetMasterVolumeLevelScalar()
            # Set volume to 100%
            volume.SetMasterVolumeLevelScalar(1.0, None)  # Max volume
            # Initialize pygame and play sound
            pygame.mixer.init()
            sound = pygame.mixer.Sound("C:\\Windows\\Media\\Windows Ding.wav")  # Replace with desired sound
            sound.play()
            # Wait for the sound to finish playing
            time.sleep(sound.get_length())
            # Restore the original volume
            volume.SetMasterVolumeLevelScalar(original_volume, None)
        
        for mask_index, mask in enumerate(masks):
            area = mask['area']*scale**2
            if area < 20000*scale**2:
                x_min, y_min, width, height = map(int, mask['bbox'])
                x_max, y_max = x_min + width, y_min + height

                # Draw bounding box on the emission image copy
                color = (0, 0, 255)  # Red color in BGR
                thickness = 2
                cv2.rectangle(em_image_copy, (x_min, y_min), (x_max, y_max), color, thickness)
                cv2.rectangle(st_image_copy, (x_min, y_min), (x_max, y_max), color, thickness)

                # Crop and overlay ROI for user input
                roi = em_image[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                roi_overlay = em_image_copy.copy()
                cv2.rectangle(roi_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), thickness=2)

                aspect_ratio = roi.shape[0] / roi.shape[1]
                new_height = int(600 * aspect_ratio)
                resized_image = cv2.resize(roi, (600, new_height))

                user_input = None
                while user_input not in ['0', '1']:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(roi_overlay)
                    axes[0].set_title('Emission Image with ROI')
                    axes[0].axis('off')
                    axes[1].imshow(resized_image, cmap='gray')
                    axes[1].set_title('Cropped ROI')
                    axes[1].axis('off')
                    plt.tight_layout()
                    plt.show(block=False)

                    try:
                        user_input = input("Enter 1 to Accept or 0 to Deny: ")
                    except Exception as e:
                        print(f"Error: {e}. Please enter 0 or 1.")
                        user_input = None

                    plt.close()

                
                # Compute properties for naming and recording results
                exposure = filtered_df['exposure'].iloc[frame_index] if frame_index < len(filtered_df) else "NA"
                particle_diameter = 2 * (area / 3.14) ** 0.50
                centroidX = x_min + width // 2
                centroidY = y_min + height // 2
                ignition = int(user_input)

                #Save the resized masked emission image if save_address is provided (non-empty)
                if save_address:
                    if not os.path.exists(save_address):  # Create directory if it doesn't exist
                        os.makedirs(save_address)
                    # Format the filename: exposure-particle diameter-centroidX-centroidY-ignition.png
                    filename = f"{exposure}-{round(particle_diameter, 2)}-{centroidX}-{centroidY}-{ignition}.png"
                    file_path = os.path.join(save_address, filename)
                    cv2.imwrite(file_path, resized_image)
                    print(f"Saved image to {file_path}")

                # Add details to the DataFrame
                row_values = {
                    'run_no': name,
                    'frame_number': frame_index + 1,
                    'delay': filtered_df['delay'].iloc[frame_index] if frame_index < len(filtered_df) else None,
                    'exposure': exposure,
                    'particle diameter': particle_diameter,
                    'centroidX': centroidX,
                    'centroidY': centroidY,
                    'ignition': int(user_input)
                }
                df = pd.concat([df, pd.DataFrame([row_values])], ignore_index=True)

        # Visualization of results for the current frame
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=750)
        axes[0].imshow(st_image)
        axes[0].axis('off')  # Remove axis
        axes[0].set_title('Preprocessed Static Image')
        axes[1].imshow(st_image_copy)
        axes[1].axis('off')  # Remove axis
        axes[1].set_title('Image with Masks')
        axes[2].imshow(em_image_copy)
        axes[2].axis('off')  # Hide axis
        axes[2].set_title('Masked Emission')
        plt.tight_layout()
        plt.show()
        
        del st_image, em_image, masks
        gc.collect()  # Explicitly free memory

    return df
