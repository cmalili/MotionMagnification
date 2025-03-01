#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:25:50 2025

@author: cmalili
"""
# amplify_spatial_Gdown_temporal_ideal.py
#
# Spatial Filtering: Gaussian blur and down sample
# Temporal Filtering: Ideal bandpass
#
# Converted from MATLAB to Python
# Original Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih,
# Original License: Please refer to the LICENCE file
# Original Date: June 2012

import os
import numpy as np
import cv2
from scipy import signal

def amplify_spatial_Gdown_temporal_ideal(vid_file, out_dir, alpha, level, 
                                        fl, fh, sampling_rate, chrom_attenuation):
    """
    Amplify subtle color variations in a video using Gaussian pyramid and ideal bandpass filtering.
    
    Parameters:
    -----------
    vid_file : str
        Path to input video file
    out_dir : str
        Directory to save output video
    alpha : float
        Amplification factor
    level : int
        Level of Gaussian pyramid (downsampling)
    fl : float
        Low frequency cutoff
    fh : float
        High frequency cutoff
    sampling_rate : float
        Video sampling rate
    chrom_attenuation : float
        Attenuation factor for chrominance
    """
    # Extract video name for output file
    vid_name = os.path.splitext(os.path.basename(vid_file))[0]
    out_name = os.path.join(out_dir, f"{vid_name}-ideal-from-{fl}-to-{fh}-alpha-{alpha}-level-{level}-chromAtn-{chrom_attenuation}.avi")
    
    # Read video
    cap = cv2.VideoCapture(vid_file)
    
    # Extract video info
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fr = cap.get(cv2.CAP_PROP_FPS)
    len_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_index = 0  # MATLAB is 1-indexed, Python is 0-indexed
    end_index = len_frames - 11  # Adjusting for 0-indexing and to match MATLAB's len-10
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You might need to change this codec
    vid_out = cv2.VideoWriter(out_name, fourcc, fr, (vid_width, vid_height))
    
    # Compute Gaussian blur stack
    print('Spatial filtering...')
    gdown_stack = build_GDown_stack(vid_file, start_index, end_index, level)
    print('Finished')
    
    # Temporal filtering
    print('Temporal filtering...')
    filtered_stack = ideal_bandpassing(gdown_stack, 1, fl, fh, sampling_rate)
    print('Finished')
    
    # Amplify
    filtered_stack[:, :, :, 0] = filtered_stack[:, :, :, 0] * alpha
    filtered_stack[:, :, :, 1] = filtered_stack[:, :, :, 1] * alpha * chrom_attenuation
    filtered_stack[:, :, :, 2] = filtered_stack[:, :, :, 2] * alpha * chrom_attenuation
    
    # Render on the input video
    print('Rendering...')
    
    # Reset video capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    
    for k in range(end_index - start_index):
        print(f"Frame {k+1}")
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to double and RGB to YIQ (similar to NTSC)
        rgb_frame = frame.astype(np.float32) / 255.0
        ntsc_frame = rgb2ntsc(rgb_frame)
        
        # Get filtered frame
        filtered = filtered_stack[k]
        filtered = cv2.resize(filtered, (vid_width, vid_height))
        
        # Add filtered frame
        combined = ntsc_frame + filtered
        
        # Convert back to RGB
        result = ntsc2rgb(combined)
        
        # Clamp values
        result = np.clip(result, 0, 1)
        
        # Write to video
        vid_out.write((result * 255).astype(np.uint8))
    
    # Release resources
    cap.release()
    vid_out.release()
    print('Finished')

def build_GDown_stack(vid_file, start_index, end_index, level):
    """
    Build a Gaussian pyramid downsampled stack from video frames.
    
    Parameters:
    -----------
    vid_file : str
        Path to input video file
    start_index : int
        First frame to process
    end_index : int
        Last frame to process
    level : int
        Pyramid level for downsampling
        
    Returns:
    --------
    gdown_stack : ndarray
        Downsampled frame stack in YIQ/NTSC color space
    """
    # Read video
    cap = cv2.VideoCapture(vid_file)
    
    # Get total number of frames to process
    num_frames = end_index - start_index
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    
    # Read first frame to get dimensions after downsampling
    ret, frame = cap.read()
    if not ret:
        raise Exception("Could not read the first frame")
    
    # Convert to YIQ (similar to NTSC) color space
    rgb_frame = frame.astype(np.float32) / 255.0
    ntsc_frame = rgb2ntsc(rgb_frame)
    
    # Create Gaussian pyramid
    down_frame = ntsc_frame.copy()
    for i in range(level):
        down_frame = cv2.pyrDown(down_frame)
    
    # Get dimensions of downsampled frame
    height, width, channels = down_frame.shape
    
    # Reset to start position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    
    # Initialize the stack
    gdown_stack = np.zeros((num_frames, height, width, channels), dtype=np.float32)
    
    # Fill the stack
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to YIQ
        rgb_frame = frame.astype(np.float32) / 255.0
        ntsc_frame = rgb2ntsc(rgb_frame)
        
        # Downsample
        down_frame = ntsc_frame.copy()
        for j in range(level):
            down_frame = cv2.pyrDown(down_frame)
            
        # Add to stack
        gdown_stack[i] = down_frame
    
    cap.release()
    return gdown_stack

def ideal_bandpassing(stack, dim, fl, fh, sampling_rate):
    """
    Apply ideal bandpass filter to the temporal dimension of the stack.
    
    Parameters:
    -----------
    stack : ndarray
        Input video stack
    dim : int
        Dimension to filter
    fl : float
        Low frequency cutoff
    fh : float
        High frequency cutoff
    sampling_rate : float
        Sampling rate
        
    Returns:
    --------
    filtered_stack : ndarray
        Filtered video stack
    """
    num_frames, height, width, channels = stack.shape
    
    # FFT
    fft_stack = np.fft.fft(stack, axis=0)
    
    # Frequencies
    frequencies = np.fft.fftfreq(num_frames, d=1.0/sampling_rate)
    
    # Create ideal bandpass filter
    mask = (np.abs(frequencies) >= fl) & (np.abs(frequencies) <= fh)
    
    # Apply mask
    mask_extended = np.zeros_like(fft_stack, dtype=bool)
    for i in range(len(mask)):
        if mask[i]:
            mask_extended[i, :, :, :] = True
    
    fft_stack = fft_stack * mask_extended
    
    # Inverse FFT
    return np.real(np.fft.ifft(fft_stack, axis=0))

def rgb2ntsc(rgb):
    """
    Convert RGB to YIQ (NTSC) color space.
    
    Parameters:
    -----------
    rgb : ndarray
        RGB image (float32, values 0-1)
        
    Returns:
    --------
    yiq : ndarray
        YIQ image
    """
    # RGB to YIQ conversion matrix
    matrix = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    
    # Reshape for matrix multiplication
    orig_shape = rgb.shape
    rgb_reshaped = rgb.reshape(-1, 3).T
    
    # Convert
    yiq = np.dot(matrix, rgb_reshaped).T
    
    # Reshape back
    return yiq.reshape(orig_shape)

def ntsc2rgb(yiq):
    """
    Convert YIQ (NTSC) to RGB color space.
    
    Parameters:
    -----------
    yiq : ndarray
        YIQ image
        
    Returns:
    --------
    rgb : ndarray
        RGB image
    """
    # YIQ to RGB conversion matrix
    matrix = np.array([
        [1, 0.956, 0.621],
        [1, -0.272, -0.647],
        [1, -1.106, 1.703]
    ])
    
    # Reshape for matrix multiplication
    orig_shape = yiq.shape
    yiq_reshaped = yiq.reshape(-1, 3).T
    
    # Convert
    rgb = np.dot(matrix, yiq_reshaped).T
    
    # Reshape back
    return rgb.reshape(orig_shape)