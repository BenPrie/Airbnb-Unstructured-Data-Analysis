# Imports as always
import pandas as pd
import numpy as np

import urllib.request
from PIL import Image, ImageStat
import io

from scipy.ndimage import laplace, gaussian_filter


# Main method for generating image features.
def create_features(listings_directory, resolution=200, smoothing_factor=1.5):
    # Get the image URLs.
    images = pd.read_csv(listings_directory)[['id', 'picture_url']]

    # Features to extract from images.
    luminances = []
    global_contrasts = []
    local_contrasts = []
    red_means = []
    blue_means = []
    green_means = []
    red_stds = []
    blue_stds = []
    green_stds = []

    for i in range(len(images)):
        print(i)
        try:
            # Download the image data from the URL.
            image_data = urllib.request.urlopen(images.picture_url.values[i]).read()

            # Open the image and convert to luminance (i.e. brightness).
            image_grey = Image.open(io.BytesIO(image_data)).convert('L')

            # Get the width:height ratio of the image so that it can be maintained in resizing.
            width, height = image_grey.size
            aspect_ratio = width / height
            target_shape = (int(resolution * aspect_ratio), resolution)

            # Downsample the image to speed up processing.
            image_grey = image_grey.resize(target_shape)

            # Compute luminance.
            luminance = np.mean(image_grey)

            # Compute global contrast.
            global_contrast = ImageStat.Stat(image_grey).stddev[0]

            # Compute local contrast.
            image_gauss_laplace = laplace(gaussian_filter(image_grey, sigma=smoothing_factor))
            image_gauss_laplace[image_gauss_laplace > 252] = 0
            local_contrast = np.mean(image_gauss_laplace)

            # Compute the colour statistics.
            image_colour = Image.open(io.BytesIO(image_data)).convert('RGB')
            image_colour = image_colour.resize(target_shape)

            stats = ImageStat.Stat(image_colour)
            total_pixels = target_shape[0] * target_shape[1]

            red_mean = stats.sum[0] / total_pixels        
            red_std = stats.stddev[0]
    
            green_mean = stats.sum[1] / total_pixels 
            green_std = stats.stddev[1]
            
            blue_mean = stats.sum[2] / total_pixels     
            blue_std = stats.stddev[2]
        
        except:
            luminance = 0
            global_contrast = 0
            local_contrast = 0
            red_mean = 0       
            red_std = 0
            green_mean = 0
            green_std = 0
            blue_mean = 0     
            blue_std = 0 
        
        luminances.append(luminance)
        global_contrasts.append(global_contrast)
        local_contrasts.append(local_contrast)
        red_means.append(red_mean)
        red_stds.append(red_std)
        green_means.append(green_mean)
        green_stds.append(green_std)
        blue_means.append(blue_mean)
        blue_stds.append(blue_std)

    return pd.DataFrame({
        'id' : images.id.values,
        'luminance' : luminances,
        'global_contrast' : global_contrasts,
        'local_contrast' : local_contrasts,
        'red_mean' : red_means,
        'blue_mean' : blue_means,
        'green_mean' : green_means,
        'red_std' : red_stds,
        'blue_std' : blue_stds,
        'green_std' : green_stds
    })