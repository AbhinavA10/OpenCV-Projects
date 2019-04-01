import numpy as np
import cv2

def processImage (image): #image is the frame sent by the video capture
    
    #Convert to Grey Image, in an attempt to use less resources in image processing
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Remove noise from the image by blurring it slightly using a low pass filter
    # Note: high freq. content like distinct edges are also removed
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(grey_image,(kernel_size, kernel_size),0)
    
# Use the Canny Edge Detection algorithim to detect edges. 
#Edge detection i.e. determines if pixel is a local max   
# Define thresholds for how strong the edge must be.
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold) #cv2.Canny returns an image
    #return edges #returning 'edges' at this point would return the image result of cv2.Canny that show edges


    
    # Next create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    
    # Defining Region of Interest
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (500, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    #create a trapeziodal area of interest -- our lanes will exist in a trapeziod area in the bottom half of the frame
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #only mark pixels classified as an edge, that are inside the trapezod, as a '1'
    masked_edges = cv2.bitwise_and(edges, mask)
    
    return masked_edges #returning 'masked_edges' at this point would return the image showing edges in the trapezoid area.

    # Use the Hough Line Transform to detect a line from the above, possibly circular, edges
    #Hough Transform:  transform a y=mx+b line to rho = x*cosTheta +y*sinTheta
    # rho = perpendicular distance to line, theta = angle b/w perpendicular line and x-axis
    #A line points in normal space will transform into lots of sine curves in Hough Space

    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 #minimum number of pixels making up a line
    max_line_gap = 30    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    # "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
    #TODO: check slope of detected line to check whether detected line is consistent with lane line; remove outliers.
    # Iterate over "lines" and draw lines on a blank image
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,100,0),5)
            # cv2.line(where, start coordinates, end coordinates, color (bgr), line thickness)
    
    # Draw the lines on the original image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges


#image_test = cv2.imread('test_images/whiteCarLaneSwitch.jpg')
#output = processImage(image_test)
#cv2.imshow('Lane Detection - Frame',output)
 	
video_capture = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')
#comment this next block out if testing only 1 image
while (video_capture.isOpened()):
    ret, frame = video_capture.read() #boolean indicating success or failure, the captured frame itself
    if ret: #if read was sucessful
        output = processImage(frame) #detect the lanes using our function
        cv2.imshow('Lane Detection - Frame',output) #show our image to the screen
        if cv2.waitKey(1) & 0xFF == ord('q'): #waitkey(1) means wait for 1ms. This allows the Frame by Frame processing
            break
    else:
        break
cv2.waitKey(0) #waitkey(0) means wait until a key is pressed
video_capture.release()
cv2.destroyAllWindows()
