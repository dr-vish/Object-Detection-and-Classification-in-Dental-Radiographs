# Loading libraries

import cv2 as cv2
import cv2 as cv
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import glob
import json
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor, sobel
from scipy import ndimage as nd

# Build dental features for a given file
def build_dental_features(file):
    # Read in radiograph file
    img = cv2.imread(file)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Use erosion
    erosion = cv2.erode(opening, kernel, iterations=1)

    # Use dilation
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Find and draw contours
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        tooth = img[y:y+h, x:x+w]

    # Extract features from each segmented tooth:

    # Convert tooth to grayscale
    tooth_gray = cv2.cvtColor(tooth, cv2.COLOR_BGR2GRAY)

    # Haralick texture features
    glcm = graycomatrix(tooth_gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]

    # Gabor filter features
    frequency = 0.6
    theta = np.pi/4
    bandwidth = 1
    psi = 0
    sigma = 1
    gabor_real, gabor_imag = gabor(tooth_gray, frequency=frequency, theta=theta, bandwidth=bandwidth, sigma_x=sigma, sigma_y=sigma, n_stds=3, offset=psi)
    gabor_magnitude = np.sqrt(np.square(gabor_real) + np.square(gabor_imag))
    gabor_mean = np.mean(gabor_magnitude)
    gabor_std = np.std(gabor_magnitude)

    # Perform canny edge detection
    edges = cv.Canny(gray,100,200)
    
    # Compute edge stats
    edge_density = np.sum(edges) / (tooth_gray.shape[0] * tooth_gray.shape[1])
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)

    # Gaussian image
    gaussian_img = nd.gaussian_filter(img, sigma = 3)

    # Sobel image
    sobel_img = sobel(img)

    # Compute contour-based features
    perimeter = cv2.arcLength(contours[i], True)
    area = cv2.contourArea(contours[i])
    circularity = 4 * np.pi * area / (perimeter**2)
    aspect_ratio = w / h
    extent = area / (w * h)

    #hu moments- for shape
    hu_shape = cv2.HuMoments(img)

    # Add label to features
    label = 'diseased'
    features = [contrast, energy, homogeneity, gabor_mean, gabor_std,edge_density, edge_mean, edge_std,circularity,hu_shape,extent,label]

    # Append features to a list for each xray and save as a CSV file
    tooth_features = []
    tooth_features.append(features)
    df = pd.DataFrame(tooth_features, columns=['contrast', 'energy', 'homogeneity', 'gabor_mean', 'gabor_std','edge_density','edge_mean','edge_std','circularity','hu_shape','extent','label'])
    file_name = file[:file.index('.')]
    df.to_csv(file_name + '_' + 'tooth_features.csv', index=False)

# Function which goes through all the files and builds dental features for each file
def create_features():
    # Process all files in a directory
    directory = '/Users/drvish/Desktop/TuftsDentalData/Radiographs'
    # save_directory = '/Users/drvish/Desktop/TuftsDentalData/pre_processed_data'
    for radiographs in os.listdir(directory):
        if radiographs.endswith('.JPG'):
            filepath = os.path.join(directory, radiographs)
            build_dental_features(filepath)

# Load the expert json file into expert_contents
with open("/Users/drvish/Desktop/TuftsDentalData/Expert/expert.json", "r") as f:
    expert_contents = json.load(f)

# Function to get the label (diseased or normal)
def get_label(name):
    for i in range(len(expert_contents)):
        d = expert_contents[i]
        n = name + ".JPG"
        if d["External ID"] == n:
            # Label each tooth as either diseased or non-diseased based on the expert.json and save the features as a CSV file:
            if d["Label"]["objects"][0]["polygons"] == "none":                
                return 0
            else:
                return 1
        else:
            continue
    
# Put all the features together in a single tooth_features.csv file
def merge_features():
    csv_files = glob.glob('/Users/drvish/Desktop/TuftsDentalData/pre_processed_data/*.{}'.format('csv'))
    l = [] 
    for f in csv_files:
        df_ = pd.read_csv(f)
        df_["name"] = f.split("data/")[1].split("_tooth")[0]
        df_["label"] = get_label(f.split("data/")[1].split("_tooth")[0])
        l.append(df_)
    df_csv_concat = pd.concat(l, ignore_index=True)
    print(df_csv_concat)
    df_csv_concat.to_csv('tooth_features_1.csv', index=False)

#create_features()
#merge_features()
    

