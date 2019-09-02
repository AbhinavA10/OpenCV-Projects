### Updated Lane Detection Algo
## is also able to detect curved lanes

import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

import callibration
import perspective_warps

callibration.callibrateCamera()

### testing distortion using checkerboard
#img = cv2.imread('camera_cal/calibration1.jpg')
#dst = callibration.undistort(img)
### Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)
#plt.show()

#print("Testing")
# Calibrating on Road images
#img = cv2.imread('test_images/test3.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#dst = callibration.undistort(img)
# Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)
#plt.show()


def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = callibration.undistort(img)
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    ### Sobel Filtering Link: https://docs.opencv.org/3.2.0/d2/d2c/tutorial_sobel_derivatives.html
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist



img = cv2.imread('test_images/test3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#dst = pipeline(img)
dst = img
dst = perspective_warps.perspective_warp(dst, dst_size=(1280,720))

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst, cmap='gray')
ax2.set_title('Warped Image', fontsize=30)
plt.show()

