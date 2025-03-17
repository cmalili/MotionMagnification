#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 17:52:50 2025

@author: cmalili
"""

import cv2
import matplotlib.pyplot as plt

input_video = "1377.mp4"
output_video = "output/1377-ideal-from-0.83-to-1.0-alpha-10-level-3-chromAtn-1.0.avi"


cap_input = cv2.VideoCapture(input_video)
cap_output = cv2.VideoCapture(output_video)

for i in range(100):
    ret_input, frame_input = cap_input.read()
    ret_output, frame_output = cap_output.read()
    if not ret_input or not ret_output:
        break
    
    if i%5==0:
        frame_input = cv2.cvtColor(frame_input, cv2.COLOR_BGR2RGB)
        plt.subplot(121)
        plt.imshow(frame_input)
        plt.title("original frame")
        plt.axis("off")
        #plt.show()
        
        frame_output = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
        plt.subplot(122)
        plt.imshow(frame_output)
        plt.title("frame after temporal filtering")
        plt.axis("off")
        plt.show()
    
cap_input.release()
cap_output.release()