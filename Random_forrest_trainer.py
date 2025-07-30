# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 23:38:51 2025

@author: Siva Kumar Valluri (based on Srinivas Battiprolu's code)
"""

import numpy as np
import cv2
import pandas as pd
import tkinter as tk
import re
from tkinter import filedialog, messagebox
import os
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #Can try regressor which tries to predict exact value rather than classifying
from sklearn import metrics
import pickle


"""
GUI section
need address of two separate folders, one with actual images and other with 'ground truths' to train
"""
class FolderSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Folder Selection GUI")

        # Variables to store folder paths and model name
        self.optical_folder = None
        self.ground_folder = None
        self.model_name = None

        # Optical Images Folder
        self.optical_label = tk.Label(root, text="Optical Images Folder:")
        self.optical_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        
        self.optical_entry = tk.Entry(root, width=50)
        self.optical_entry.grid(row=0, column=1, padx=10, pady=10)
        
        self.optical_button = tk.Button(root, text="Browse", command=self.select_optical_folder)
        self.optical_button.grid(row=0, column=2, padx=10, pady=10)

        # Ground Truth Images Folder
        self.ground_label = tk.Label(root, text="Ground Truth Images Folder:")
        self.ground_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")
        
        self.ground_entry = tk.Entry(root, width=50)
        self.ground_entry.grid(row=1, column=1, padx=10, pady=10)
        
        self.ground_button = tk.Button(root, text="Browse", command=self.select_ground_folder)
        self.ground_button.grid(row=1, column=2, padx=10, pady=10)

        # Model Name Input
        self.model_label = tk.Label(root, text="Name of Trained Model:")
        self.model_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        
        self.model_entry = tk.Entry(root, width=50)
        self.model_entry.grid(row=2, column=1, padx=10, pady=10)

        # Confirm Button
        self.confirm_button = tk.Button(root, text="Confirm", command=self.confirm_selection)
        self.confirm_button.grid(row=3, column=1, padx=10, pady=20)

    def select_optical_folder(self):
        folder = filedialog.askdirectory(title="Select Optical Images Folder")
        if folder:
            self.optical_folder = folder
            self.optical_entry.delete(0, tk.END)
            self.optical_entry.insert(0, folder)

    def select_ground_folder(self):
        folder = filedialog.askdirectory(title="Select Ground Truth Images Folder")
        if folder:
            self.ground_folder = folder
            self.ground_entry.delete(0, tk.END)
            self.ground_entry.insert(0, folder)

    def confirm_selection(self):
        self.model_name = self.model_entry.get()

        if not self.optical_folder or not self.ground_folder:
            messagebox.showerror("Error", "Both folders must be selected.")
        elif not re.match("^[a-zA-Z0-9_]+$", self.model_name):
            messagebox.showerror("Error", "Model name must be alphanumeric.")
        else:
            self.root.destroy()

    def pass_to_function(self):
        # Return the selected folder paths and model name
        return self.optical_folder, self.ground_folder, self.model_name

def run_app():
    root = tk.Tk()
    app = FolderSelectionApp(root)
    root.mainloop()
    return app.pass_to_function()

def organize(address):
    image_set_names = []
    tiff_images_addresses = []
    for root, subfolders, filenames in os.walk(address):
        for filename in filenames:
            if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
                image_set_names.append(root.rpartition('/')[2])
                tiff_images_addresses.append(os.path.join(root, filename))
                      
    address_set = []
    name_set = []
    run_set = []
    previous_name = ''
    i = 0
    for image_name in image_set_names:
        current_name = image_name.rpartition('\\')[2]
        if current_name != previous_name and i != 0:
            address_set.append(run_set)
            name_set.append(current_name)
            run_set = []
            run_set.append(tiff_images_addresses[i])
        elif current_name != previous_name and i == 0:
            name_set.append(current_name)
            run_set = []
            run_set.append(tiff_images_addresses[i])
        else:
            run_set.append(tiff_images_addresses[i])
            
        previous_name = current_name
        i += 1
    # Last cycle data added
    address_set.append(run_set)
    return address_set, name_set


optical_folder_address, ground_folder_address, model_name = run_app()
image_addresses, _ = organize(optical_folder_address)
mask_addresses, _ = organize(ground_folder_address)

"""
Building feature database and labeling (ground truth)
"""

# Feature database with 41 features nad label column
df = pd.DataFrame()

# Loop through each pair of image and mask addresses
for image_address, mask_address in zip(image_addresses[0], mask_addresses[0]):
    img = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
    img2 = img.reshape(-1)

    # Create a temporary DataFrame for the current pair of images
    temp_df = pd.DataFrame()

    # Feature-1: Original pixel values
    temp_df['Original intensity'] = img2

    # Feature-2-33: Gabor features
    num = 1  
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    gabor_label = 'Gabor' + str(num)
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    temp_df[gabor_label] = filtered_img
                    num += 1

    # Feature-34: Canny edge detection feature
    edges = cv2.Canny(img, 100, 200)
    temp_df['Canny edge'] = edges.reshape(-1)

    # Feature-35: Roberts edge detection feature
    edge_roberts = roberts(img)
    temp_df['Roberts'] = edge_roberts.reshape(-1)

    # Feature-36: Sobel edge detection feature
    edge_sobel = sobel(img)
    temp_df['Sobel'] = edge_sobel.reshape(-1)

    # Feature-37: Scharr edge detection feature
    edge_scharr = scharr(img)
    temp_df['Scharr'] = edge_scharr.reshape(-1)

    # Feature-38: Prewitt edge detection feature
    edge_prewitt = prewitt(img)
    temp_df['Prewitt'] = edge_prewitt.reshape(-1)

    # Feature-39: Gaussian blurred with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    temp_df['Gaussian s3'] = gaussian_img.reshape(-1)

    # Feature-40: Gaussian blurred with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    temp_df['Gaussian s7'] = gaussian_img2.reshape(-1)

    # Feature-41: Median blurred sigma=3
    median_img = nd.median_filter(img, size=3)
    temp_df['Median s3'] = median_img.reshape(-1)

    # Ground truth (column 42)
    labeled_img = cv2.imread(mask_address, cv2.IMREAD_GRAYSCALE)
    temp_df['Label'] = labeled_img.reshape(-1)

    # Concatenate the temporary DataFrame with the main DataFrame
    df = pd.concat([df, temp_df], ignore_index=True)

Y = df['Label'].values
X = df.drop(labels = 'Label', axis = 1)

"""
Training the model 
"""    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 111)

model = RandomForestClassifier(n_estimators = 10, random_state = 1)
model.fit(X_train, Y_train)

prediction_train = model.predict(X_train)
prediction_test = model.predict(X_test)

print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))
features_list = list(X.columns)
feature_importance = pd.Series(model.feature_importances_, index = features_list).sort_values(ascending = False)
print(feature_importance)

pickle.dump(model, open(model_name,'wb'))


"""
Using the model
import matplotlib.pyplot as plt
load_model = pickle.load(open(model_name,'rb'))
result = load_model.predict(X)
segmented_image = result.reshape((img.shape))
plt.imsave('segmented.jpg', segmented_image, cmap='jet')
"""