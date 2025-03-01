#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:34:39 2025

@author: cmalili
"""

import numpy as np

def ideal_bandpassing(input_signal, dim, wl, wh, sampling_rate):
    """
    Apply an ideal bandpass filter on input_signal along dimension dim.
    
    Parameters:
        input_signal (ndarray): Input array.
        dim (int): Dimension along which to filter (1-indexed, as in MATLAB).
        wl (float): Lower cutoff frequency.
        wh (float): Upper cutoff frequency.
        sampling_rate (float): Sampling rate of the signal.
        
    Returns:
        filtered (ndarray): The filtered signal (real part).
    """
    # Check that the dimension is valid (MATLAB-style dim is 1-indexed)
    if dim > input_signal.ndim:
        raise ValueError("Exceed maximum dimension")
    
    # Convert to 0-indexed axis for Python
    axis = dim - 1
    
    # Move the axis we want to filter to the front (axis 0)
    input_shifted = np.moveaxis(input_signal, axis, 0)
    
    # Get the size along the filtered dimension
    n = input_shifted.shape[0]
    
    # Create the frequency vector (same as MATLAB: Freq = (0:(n-1))/n*sampling_rate)
    freqs = np.arange(n) / n * sampling_rate
    
    # Create a mask selecting frequencies between wl and wh
    mask = (freqs > wl) & (freqs < wh).reshape((n,1))
    
    # Reshape mask so that it can be broadcast along the other dimensions.
    # New shape is (n, 1, 1, ..., 1) with as many 1's as input_shifted.ndim - 1.
    #mask_full = np.tile(mask, (1, F.shape[1]))
    
    # Compute FFT along axis 0
    F = np.fft.fft(input_shifted, axis=0)
    
    # Zero out frequencies outside the desired band
    mask_full = np.broadcast_to(mask, F.shape)
    F = F * mask
    
    # Inverse FFT to get the filtered signal and take the real part
    filtered_shifted = np.real(np.fft.ifft(F, axis=0))
    
    # Move the filtered axis back to its original position
    filtered = np.moveaxis(filtered_shifted, 0, axis)
    
    return filtered

# Example usage:
if __name__ == "__main__":
    # Create a sample 2D signal: e.g. 1000 samples, with 2 channels
    np.random.seed(0)
    signal_data = np.random.randn(1000, 2)
    
    # Parameters for filtering along the first dimension (dim=1)
    wl = 8      # Lower cutoff frequency
    wh = 12     # Upper cutoff frequency
    sampling_rate = 250  # Hz
    
    # Apply the ideal bandpass filter along dimension 1 (first dimension)
    filtered_data = ideal_bandpassing(signal_data, dim=1, wl=wl, wh=wh, sampling_rate=sampling_rate)
    print("Filtered signal shape:", filtered_data.shape)
