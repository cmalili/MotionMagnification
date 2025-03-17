#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:01:54 2025

@author: cmalili
"""

import os
from motion_magnification_b import amplify_spatial_Gdown_temporal_ideal

# Define parameters
input_video = "vid2.mp4"              # Path to your input video
output_directory = "output"           # Directory for the output video
alpha = 10                            # Amplification factor
level = 3                             # Level of Gaussian pyramid
fl = 0.83                             # Low frequency cutoff (Hz)
fh = 1.0                              # High frequency cutoff (Hz)
sampling_rate = 30                    # Frame rate of the video (assumed 30 fps)
chrom_attenuation = 1.0               # Chrominance attenuation factor (1.0 = no attenuation)

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Run the amplification
print(f"Processing {input_video}...")
amplify_spatial_Gdown_temporal_ideal(
    input_video,
    output_directory,
    alpha,
    level,
    fl,
    fh,
    sampling_rate,
    chrom_attenuation
)
print(f"Processing complete. Output saved to {output_directory}")