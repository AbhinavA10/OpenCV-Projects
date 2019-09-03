## Lane-Detection

Using OpenCV to perform Lane Detection from a Video feed

This method uses perspective transforms, a sobel filter, and the sliding window algorithim to find the lane lines.

Below is the pipeline that was used

Original image:

![Result](result/01_original.png)

Bird's eye perspective transform, and sobel filter in grayscale

![Result](result/02_warped-and-sobel.png)

Histogram of Warped image. This represents the number of white pixels in column `x`
![Result](result/03_histogram_small.png)

Sliding Window Algorithim and poly line fitting

![Result](result/04_sliding-window.png)

Inverse Warp and Overlayed:

![Result](result/05_overlayed.png)

Sobel Filtering and Sliding Window Algorithim result:

![Result](result/overlayed_result.gif)
