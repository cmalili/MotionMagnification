#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:34:26 2025

@author: cmalili
"""

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

# Reproducing figure 4 in the paper mentioned above
# True motion magnification with lambda 2 pi
lam = np.pi
x = np.linspace(0, 4*np.pi, 100)
I = np.cos(x)
alphas = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = ['blue', 'skyblue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
I_true = np.cos(x - (1 + alphas[3])*np.pi/8)


for alpha, color in zip(alphas, colors):
    I_true = np.cos(x - (1 + alpha)*np.pi/8)
    plt.plot(x, I_true, color=color, label=alpha)
    
plt.plot(x, I, color='black',)
plt.title("True Motion Magnification")
plt.grid(visible=True)
plt.xlabel("x (space)")
plt.ylabel("Intensity")
plt.legend()
plt.show()


# True motion magnification with lambda pi
x = np.linspace(0, 4*np.pi, 100)
I = np.cos(2*x)
alphas = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
colors = ['blue', 'skyblue', 'cyan', 'lightgreen', 'yellow', 'orange', 'red']
I_true = np.cos(x - (1 + alphas[3])*np.pi/8)


for alpha, color in zip(alphas, colors):
    I_true = np.cos(2*x - (1 + alpha)*np.pi/8)
    plt.plot(x, I_true, color=color, label=alpha)
    
plt.plot(x, I, color='black',)
plt.title("True Motion Magnification")
plt.grid(visible=True)
plt.xlabel("x (space)")
plt.ylabel("Intensity")
plt.legend()
plt.show()

# using approximation
# approximation magnification with lambda 2pi
I = np.cos(x)
for alpha, color in zip(alphas, colors):
    I_tf = np.cos(x) + (1 + alpha)*np.pi/8*np.sin(x)
    plt.plot(x, I_tf, color=color, label=alpha)

plt.plot(x, I, color='black')
plt.title("Approximate Motion Magnification")
plt.grid(visible=True)
plt.xlabel("x (space)")
plt.ylabel("Intensity")
plt.legend()
plt.show()


I = np.cos(2*x)
for alpha, color in zip(alphas, colors):
    I_tf = np.cos(2*x) + (1 + alpha)*np.pi/8*np.sin(2*x)
    plt.plot(x, I_tf, color=color, label=alpha)

plt.plot(x, I, color='black')
plt.title("Approximate Motion Magnification")
plt.grid(visible=True)
plt.xlabel("x (space)")
plt.ylabel("Intensity")
plt.legend()
plt.show()











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



plt.plot(eeg_signal[:4000,0], label="eeg_signal")
plt.plot(filtered_signal[:4000], label="filtered signal")
#plt.plot(filtered_signal[:4000] + eeg_signal[:4000,0], label="eeg + filtered signal")
plt.title("eeg signals")
plt.xlabel("Time (space)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

plt.plot(filtered_signal[:4000] + eeg_signal[:4000,0], label="eeg + filtered signal")
plt.title("eeg with amplified signals")
plt.xlabel("Time (space)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


#GTK
'''
Note: the existing code implementations here https://people.csail.mit.edu/mrub/
vidmag/#code may be useful to refer as you implement your method. It’s highly rec-
ommended to get your method working on their data first, then try to get it working on
your own data.
'''