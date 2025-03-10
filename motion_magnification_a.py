#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:34:26 2025

@author: cmalili
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import scipy.signal as signal

#part a

'''
Read the original paper on Eulerian video magnification https://people.csail.mit.edu/
mrub/papers/vidmag.pdf. 
Write a code to reproduce Figure 4, amplifying a 
 (1) a simple sinusoid as done in the figure; and 
 (2) pick an EEG or heart rate signal or equivalent physiological signal and amplify certain frequencies inside of it. For (2), note where you
got the signal (cite the source), and 
explain which set of frequencies you chose to amplify and why (i.e. their physiological relevance). (10 points)
'''
'''
# Reproducing figure 4 in the paper mentioned above
x = np.linspace(0, 4*np.pi, 100)
I = np.cos(x)
alphas = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = ['blue', 'skyblue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
I_true = np.cos(x - (1 + alphas[3])*np.pi/8)

plt.plot(x, I, color='black')
plt.plot(x, I_true, color=colors[3])
plt.grid(visible=True)
plt.show()

# using approximation
I_tf = np.cos(x) + (1 + alphas[3])*np.pi/8*np.sin(x)

plt.plot(x, I, color='black')
plt.plot(x, I_tf, color=colors[3])
plt.grid(visible=True)
plt.show()
'''


# Example usage
record_name = "iaf1_afw"
'''
eeg_signal = load_eeg_binary(file_path)
print("Loaded EEG shape:", eeg_signal.shape)

with open(file_path, "rb") as f:
    print(f.read(20))
'''

record = wfdb.rdrecord(record_name)
eeg_signal = record.p_signal


fs = record.fs
lowcut = 8
highcut = 12
nyquist = 0.5*fs

b, a = signal.butter(4, [lowcut/nyquist, highcut/nyquist], btype='band')
filtered_signal = signal.filtfilt(b, a, eeg_signal[:,0])





plt.plot(eeg_signal[:4000,0])
plt.plot(filtered_signal[:4000])
plt.show()




#Part b
'''
Take a video of your face with very little movement (try to be as still as possible!). Then
implement a version of Eulerian video magnification to amplify the color in your face due
to blood pulse. If you are having difficulty with amplifying the color in your face, document
your challenges (i.e. skin tone, etc) and find an alternative video that you capture to
magnify. Also, it would be good to try and capture RAW data if possible, using either a
DSLR camera or download an application to extract it from your cellphone, as that will
lend you better results. Document your method’s implementation and your results on your
data.
'''


#GTK
'''
Note: the existing code implementations here https://people.csail.mit.edu/mrub/
vidmag/#code may be useful to refer as you implement your method. It’s highly rec-
ommended to get your method working on their data first, then try to get it working on
your own data.
'''