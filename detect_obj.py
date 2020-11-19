#!/usr/bin/env python
# coding: utf-8



import cv2 
import argparse
import sys
import numpy as np




def connected_component_analysis(input_image):
    connections = 8 # can be 4 or 8
    num_labels, labels, properties, centroids = cv2.connectedComponentsWithStats(input_image , connections , cv2.CV_32S)
    return num_labels, properties
    


# 4. classify components as person,car or other
# cycle through labels detected in current frame
    
# need to get ratios right
def classify_component(num_labels, properties, frame_num):
    people = 0
    cars = 0
    others = 0
# if height
    for i in range(1, num_labels):
        height = properties[i][cv2.CC_STAT_HEIGHT]
        width = properties[i][cv2.CC_STAT_WIDTH]
        if height > (width):
            people += 1
        elif width > (height):
            cars += 1
        else: 
            others += 1
    
    object_num = people + cars + others
                  
    print('Frame ' +  str(frame_num) + ':  ' + str(num_labels) + ' objects ' +  '(' + str(people) 
          +' people, ' + str(cars)+ ' cars ' + str(others) + ' others)') 





def extract_object_in_color(frame, mask):
    #everywhere that binarized image is black, setbackground to black
    frame[mask == 0] = 0
    return frame



# using opencv functions makes for clearer (actual) components but with more background noise which -
# - interferes with the connected component analysis

# The more kernal iterations, the better component analysis (number of components), cuts out pieces of color object however

def remove_noise(mask):

    #orig_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, orig_kernel)
    
    # kernal to reduce noise
    # morphological operation of opening is used. First erode then dilate
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 3)
    dilation = cv2.dilate(erosion,kernel,iterations = 3)
    
    #opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    #gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel) looks cool but not part of task
    return dilation
    #eturn cleaned_mask




def display(top_left, top_right, bottom_left, bottom_right):
        top_row = np.concatenate((top_left, top_right), axis=1)
        bottom_left = cv2.cvtColor(bottom_left, cv2.COLOR_GRAY2RGB)        
        bottom_row = np.concatenate((bottom_left, bottom_right), axis=1)
        total_display_window = np.concatenate((top_row, bottom_row), axis = 0)
        cv2.resizeWindow('display',1080,720)
        cv2.imshow('display', total_display_window)




# foreground mask from link below
#https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html
def detect_obj(video_file):
    cap = cv2.VideoCapture(video_file)
    # mixture of gaussians for background modelling, disable shadow detection
    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    # counter for frames
    frame_num = 0
    # Guard against video not opening
    if not cap.isOpened():
        print("Error opening video")
        exit()
    
    ## current set up works for extracting color objects, be careful of python passing everything by reference
    cv2.namedWindow('display',cv2.WINDOW_NORMAL)
    ## LOOP VIDEO
    while(cap.isOpened()):
        frame_num += 1
        check, frame_original = cap.read()
        if check:
            # apply background subtraction 
            frame = frame_original.copy() 
            orginal_display = frame_original.copy()
            mask = subtractor.apply(frame)
            # 2. apply transformation to remove noise 
            # use morphological operators to reduce noise and isolate components 
            cleaned_mask = remove_noise(mask)
            # display estimated background image in top right
            bg_image = subtractor.getBackgroundImage(frame)
            
            # get detected objects in original colour
            colour_object = extract_object_in_color(frame_original, cleaned_mask)
            
            # 3. Count separate moving objects using connected component analysis
            # note this only works as image is inverted. Function counts white space as component
            # might have to threshold components, only accept those of given size 
            component_num, component_properties = connected_component_analysis(cleaned_mask)
            
            #4. Classify each component
            classify_component(component_num, component_properties, frame_num)
            
            #display 4 images in single window
            display_task1(orginal_display, bg_image, mask, colour_object)
            # Terminate Video Feed
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


    




# MAIN 
if __name__ == "__main__":
    argv = sys.argv[1:]
    task = argv[0]
    image = argv[1]
    task_1(image)





