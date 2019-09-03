import numpy as np
def get_hist(img):
    """Returns a 1D array that represents histogram of an image
    """
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist