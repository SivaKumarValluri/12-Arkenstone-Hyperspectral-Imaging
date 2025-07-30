# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 04:16:24 2025

@author: Siva Kumar Valluri
"""
import numpy as np
import cv2
import pandas as pd
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd


def RFC_segment_image(img, model):   #image should be rgayscale

    #img = cv2.imread(image_address, cv2.IMREAD_GRAYSCALE)
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
    
    X = temp_df
    result = model.predict(X)
    segmented_image = result.reshape((img.shape))
    return segmented_image


"""
address = input('copy paste address of picture for testing =')
segmented_image = RFC_segment_image(address)
plt.imshow(segmented_image)
plt.show()
"""