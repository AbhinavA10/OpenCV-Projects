import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle

### https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     #dst=np.float32([(0.43,0), (0.58, 0), (0.43,1), (0.58,1)])): # to visualize full birds eye view warp
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])): # we isolate reigon of interest in the perspective transform itself
                     #points defined in x,y from x bottom right corner, y top left
    """Perspective Warp
    Turns a camera view, into a bird's eye view
    """
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # arbitrarily choose source points to have nice fit for warped result
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def inv_perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    """Inverse Perspective Warp
    Maps a bird's eye view back to original camera view
    """
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # arbitrarily choose destination points to have nice fit for warped result
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped
