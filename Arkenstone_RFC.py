# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 21:33:38 2025

@author: Siva Kumar Valluri
"""
import cv2
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from Arkenstone_Random_Forest_Classifier import RFC_segment_image


def Arkenstone_RFC(frame_details, images_in_run):
    def display_with_matplotlib(image, particle_no, size, frame):
        """Displays an image using Matplotlib and collects user input via the terminal."""
        plt.imshow(image, cmap='gray')
        plt.title(f"Particle n# {particle_no} of size {size} in frame {frame}")
        plt.axis('off')
        plt.show(block=False)

        user_input = None
        while user_input not in ['0', '1']:
            user_input = input("Enter 1 to Accept or 0 to Deny: ")
        plt.close()
        return int(user_input)

    df = pd.DataFrame(columns=['run_no', 'frame_number', 'delay', 'exposure', 'particle diameter', 'centroidX', 'centroidY', 'ignition'])
    df_frame = pd.DataFrame(frame_details, columns=["delay", "exposure", "choice"])
    filtered_df = df_frame[df_frame["choice"]]

    run_no = images_in_run[0].rpartition('\\')[0].rpartition('\\')[2]
    model = pickle.load(open('C:\\Users\\sivak\\OneDrive\\Desktop\\backup\\Arkenstone\\SIMXparticles', 'rb'))
    static_images = []
    segmented_images = []
    emission_images = []

    for image_number in range(len(images_in_run)):
        if image_number < len(frame_details):
            if frame_details[image_number][2]:
                static_image = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                segmented_image = RFC_segment_image(static_image, model)
                segmented_image = cv2.normalize(segmented_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                static_images.append(static_image)
                segmented_images.append(segmented_image)
        elif image_number > len(frame_details) - 1:
            if frame_details[int(image_number - 8)][2]:
                emission_fr = cv2.imread(images_in_run[image_number], cv2.IMREAD_GRAYSCALE)
                emission_image = cv2.normalize(emission_fr, None, 0, 255, cv2.NORM_MINMAX)
                emission_images.append(emission_image)

    frame_counter = 1
    df_counter = 0

    for st_image, em_image in zip(segmented_images, emission_images):
        contours, _ = cv2.findContours(st_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_new = [contour for contour in contours if len(contour) > 4 and cv2.moments(contour)['m00'] > 0]

        boundingboxes = []
        particle_sizes = []
        centroidX = []
        centroidY = []

        for contour in contour_new:
            M = cv2.moments(contour)
            area = cv2.contourArea(contour)
            diameter = (2 * (area / 3.14) ** 0.5)
            rect = cv2.minAreaRect(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boundingboxes.append(box)
            centroidX.append(cX)
            centroidY.append(cY)
            particle_sizes.append(diameter)
            
        number_of_particles = len(boundingboxes)
        particle_counter = 1
        for box, contour, particle_size, X, Y in zip(boundingboxes, contour_new, particle_sizes, centroidX, centroidY):
            if particle_size > 20:
                mask = np.zeros_like(em_image, dtype=np.uint8)
                #cv2.drawContours(image=mask, contours=contour, contourIdx=-1, color=255, thickness=1, lineType=cv2.LINE_AA)
                #cv2.drawContours(image=mask, contours=box, contourIdx=-1, color=255, thickness=1, lineType=cv2.LINE_AA)
                cv2.drawContours(mask, [box], -1, 255, thickness=cv2.FILLED)
                masked_emission_image = cv2.bitwise_and(em_image, em_image, mask=mask)
                x, y, w, h = cv2.boundingRect(box)
                roi = masked_emission_image[y:y + h, x:x + w]
    
                aspect_ratio = roi.shape[0] / roi.shape[1]
                new_height = int(600 * aspect_ratio)
                resized_image = cv2.resize(roi, (600, new_height))
    
                user_input = display_with_matplotlib(resized_image, str(particle_counter), str(round(particle_size)), str(frame_counter))
    
                row_values = [
                        run_no, frame_counter,
                        filtered_df["delay"].iloc[df_counter] if df_counter < len(filtered_df) else None,
                        filtered_df["exposure"].iloc[df_counter] if df_counter < len(filtered_df) else None,
                        particle_size, X, Y, user_input
                    ]
    
                df.loc[len(df)] = row_values
                particle_counter += 1
                
        frame_counter += 1
        df_counter += 1
            
    return df






