## Lane-Detection

Using OpenCV to perform Lane Detection from a Video feed

This method uses the Canny Edge Detection and Hough Line Transforms

Below is the pipeline that was used

Original:
![Original Video](Result/01_Original_Video.gif)

GreyScaled:
![Greyscaled Video](Result/02_Greyscaled.gif)

After performing Canny Edge Detection:
![Canny Edge Video](Result/03_Edges.gif)

After finding edges only in the area of interest
![Canny Edge Video](Result/04_Edges_close.gif)

Final Result after using Hough Transform:
![Result](Result/05_Result_large.gif)
