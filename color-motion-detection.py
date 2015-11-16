'''
## Task 04. Color Object Detection 

Now we have a more complex scenario. We have several objects of the same colour, i.e. 5 yellow balls. They are diposed around the scene having diferent sizes. What we want now is to track the only one that is moving around, no matter its size. 


### Goals:
* Step 1: Basic motion detection 

* Step 2: Detect the presence of colored objects using computer vision techniques.

* Step 3: Track the object as it moves around in the video frames, drawing its previous positions as it moves, creating a tail behind it


__by Aurélien Hontabat & Xiaoqian Xiong__

note: this code was to be delivered in a python notebook
'''

# imports and helper functions
from collections import deque
import numpy as np
import cv2
import time
import datetime


def resize(image, target_width=None, target_height=None, inter=cv2.INTER_AREA):
    ''' 
    Convenience resize function
    initialize the dimensions of the image to be resized 
    and grab the image size
    '''
    dimention = None
    (img_height, img_width) = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        ratio = target_height / float(img_height)
        dimention = (int(img_width * ratio), target_height)
    else:
        ratio = target_width / float(img_width)
        dimention = (target_width, int(img_height * ratio))
    
    # resize the image
    resized = cv2.resize(image, dimention, interpolation=inter)
    
    return resized


def setup_trackbars(range_filter):
    cv2.namedWindow("Trackbars", 0)
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = 0 if i == "MIN" else 255
            cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, lambda x : None)
            
def get_trackbar_values(range_filter):
    values = {}
    for i in ["MIN", "MAX"]:
        for j in range_filter:
            v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
            values["%s_%s" % (j, i)] = v
    return values

# ------------------------------------------------------------------
# step one - basic motion detection

VIDEODEV = 'vid3.avi'

camera = cv2.VideoCapture(VIDEODEV); assert camera.isOpened()

# setup the background
firstFrame = None

while True:
    (grabbed, frame) = camera.read()
    frame = resize(frame, target_width=500)
    
    # make gray and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", gray)
    
    # if the `q` key is pressed, exit loop
    if cv2.waitKey(1) & 0xFF is ord('q'):
        break

firstFrame = gray # record frame

# cleanup
camera.release()    
cv2.destroyAllWindows()

# ------------------------------------------------------------
camera = cv2.VideoCapture(VIDEODEV); assert camera.isOpened()

# show motion
min_area = 500
diffDelta = 25

while True:
    (grabbed, frame) = camera.read()
    frame = resize(frame, target_width=500)
    
    # make gray and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # compute the absolute difference between frames
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, diffDelta, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes, then find contours
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # draw contours
    for contour in contours:
        
        # if the contour is too small, ignore it
        if cv2.contourArea(contour) < min_area:
            continue
            
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # show the frame and record if the user presses a key
    cv2.imshow("Security Feed", frame)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Frame Delta", frameDelta)
    
    # if the `q` key is pressed, exit loop
    if cv2.waitKey(1) & 0xFF is ord('q'):
        break

# cleanup
camera.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------
# step two - detect elements of a given color

camera = cv2.VideoCapture(VIDEODEV); assert camera.isOpened()

# query what color should be tracked
setup_trackbars('HSV')

while True:
    (grabbed, frame) = camera.read()
    frame = resize(frame, target_width=500)
    
    # calculate threshold 
    frame_to_thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = get_trackbar_values('HSV')
    thresh = cv2.inRange(frame_to_thresh, (v['H_MIN'], v['S_MIN'], v['V_MIN']), (v['H_MAX'], v['S_MAX'], v['V_MAX']))
    
    # show both images
    cv2.imshow("Original", frame)
    cv2.imshow("Thresh", thresh)
    
    # if the `q` key is pressed, exit loop
    if cv2.waitKey(1) & 0xFF is ord('q'):
        break

# store the desired color
colorLower = (v['H_MIN'], v['S_MIN'], v['V_MIN'])
colorUpper = (v['H_MAX'], v['S_MAX'], v['V_MAX'])

# cleanup
camera.release()
cv2.destroyAllWindows()

# ----------------------------------------------------------------
camera = cv2.VideoCapture(VIDEODEV); assert camera.isOpened()

# show color elements
while True:
    (grabbed, frame) = camera.read()
    frame = resize(frame, target_width=500)
    
    # resize the frame, blur it, and convert it to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color selected
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if(radius > 10):
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, "x: {}, y: {}".format(center[0], center[1]),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    
    # if the `q` key is pressed, exit loop
    if cv2.waitKey(1) & 0xFF is ord('q'):
        break
        
# cleanup
camera.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------
# step three - detect motion only from objects with a color

camera = cv2.VideoCapture(VIDEODEV); assert camera.isOpened()

tailsize = 64
pts = deque(maxlen = tailsize) # points that form the trail

while True: 
    (grabbed, frame) = camera.read()
    frame = resize(frame, target_width=500)
    
    # COLOR MASK
    # resize the frame, blur it, and convert it to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color selected
    c_mask = cv2.inRange(hsv, colorLower, colorUpper)
    c_mask = cv2.erode(c_mask, None, iterations=2)
    c_mask = cv2.dilate(c_mask, None, iterations=2)
    
    # MOVEMENT MASK
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    frameDelta = cv2.absdiff(firstFrame, gray)
    m_mask = cv2.threshold(frameDelta, diffDelta, 255, cv2.THRESH_BINARY)[1]
    m_mask = cv2.dilate(m_mask, None, iterations=2)
    
    # color/movement mask combined
    cm_mask = cv2.multiply(c_mask, m_mask)
    
    # find contours in the mask and initialize the current (x, y) center of the ball
    contours = cv2.findContours(cm_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    for contour in contours:
        ((x, y), radius) = cv2.minEnclosingCircle(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        if(radius > 10):
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    pts.appendleft(center)
    
    # loop over the set of tracked points
    for i in xrange(1, len(pts)):
        
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(tailsize / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    
            
    # show the images
    cv2.imshow("result", frame)
    cv2.imshow("color mask", c_mask)
    cv2.imshow("movement mask", m_mask)
    cv2.imshow("combination", cm_mask)
    
    # if the `q` key is pressed, exit loop
    if cv2.waitKey(1) & 0xFF is ord('q'):
        break

# cleanup
camera.release()
cv2.destroyAllWindows()

    

