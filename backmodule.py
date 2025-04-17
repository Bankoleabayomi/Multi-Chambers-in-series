# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 11:07:44 2025

@author: banko
"""

import os
import cv2
import numpy as np
import pandas as pd
import time
import psutil
import csv
import tempfile
import shutil
from urllib.parse import urlparse
import numpy as np
from datetime import datetime
from skimage.measure import label, regionprops
from scipy.optimize import curve_fit
from pyswarm import pso
import matplotlib.pyplot as plt

# ------------------- Cloud Download Helpers -------------------

def download_images_from_s3(bucket_name, prefix):
    import boto3
    s3 = boto3.client('s3')
    tmp_dir = tempfile.mkdtemp()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith(('.jpg', '.png', '.tiff')):
            local_path = os.path.join(tmp_dir, os.path.basename(key))
            s3.download_file(bucket_name, key, local_path)
    return tmp_dir

def download_images_from_gcs(bucket_name, prefix):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    tmp_dir = tempfile.mkdtemp()
    for blob in blobs:
        if blob.name.lower().endswith(('.jpg', '.png', '.tiff')):
            local_path = os.path.join(tmp_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
    return tmp_dir

def parse_cloud_url(url):
    """
    Parse a cloud storage URL and return provider ('s3' or 'gcs'),
    bucket name, and prefix.
    """
    parsed = urlparse(url)
    netloc = parsed.netloc
    path = parsed.path
    if "s3.amazonaws.com" in netloc or netloc.endswith("s3.amazonaws.com"):
        # S3 URL
        if netloc.count('.') >= 3:
            bucket_name = netloc.split('.')[0]
        else:
            parts = path.split('/')
            bucket_name = parts[1] if len(parts) > 1 else None
            path = '/'+'/'.join(parts[2:]) if len(parts) > 2 else '/'
        prefix = path.lstrip('/')
        return 's3', bucket_name, prefix
    elif "storage.googleapis.com" in netloc:
        parts = path.split('/')
        bucket_name = parts[1] if len(parts) > 1 else None
        prefix = '/'.join(parts[2:]) if len(parts) > 2 else ''
        return 'gcs', bucket_name, prefix
    else:
        return None, None, None

def download_images_from_cloud(url):
    provider, bucket_name, prefix = parse_cloud_url(url)
    if provider == 's3':
        return download_images_from_s3(bucket_name, prefix)
    elif provider == 'gcs':
        return download_images_from_gcs(bucket_name, prefix)
    else:
        # If URL not recognized as cloud storage, assume local path.
        return url

# ------------------- Image Processing Function -------------------


def process_flocs_and_analyze(root_folder, thresh=None, pixel_to_micron=None, bins=None, labels=None):
    """
    Process floc images from multiple folders (time-series) to segment, measure region properties,
    and then perform additional analysis on equivalent diameters.
    
    Parameters:
        root_folder (str): Parent folder containing subfolders (named numerically) of images.
        thresh (int): Fixed threshold value to segment the images.
        pixel_to_micron (float): Conversion factor from pixel to micrometers.
        bins (list or array): Bins to cut the equivalent diameter values.
        labels (list): Labels corresponding to the bins.
    
    Returns:
        grouped_df (pd.DataFrame): DataFrame grouping the counts of equivalent diameter ranges
                                   per time (folder), including the additional analysis.
    """
    results = []
    
    # Loop through folders in numerical order
    for folder_name in sorted(os.listdir(root_folder), key=lambda x: int(x)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            # Process each image file in the folder (assumed to be a time series)
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(('.jpg', '.png', '.tiff')):
                    image_path = os.path.join(folder_path, filename)
                    # Read image in grayscale
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Warning: Could not load image: {image_path}")
                        continue
                    
                    # Fixed threshold segmentation
                    ret, binary_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
                    
                    # Label connected regions (using skimage's label)
                    labeled_img = label(binary_img)
                    
                    # Extract region properties
                    regions = regionprops(labeled_img)
                    
                    # Reset particle numbering for each image
                    particle_num = 1
                    
                    for region in regions:
                        # Convert perimeter and check if nonzero
                        perimeter = region.perimeter * pixel_to_micron
                        if perimeter > 0:
                            area = region.area * (pixel_to_micron ** 2)
                            equivalent_diameter = region.equivalent_diameter * pixel_to_micron
                            major_axis_length = region.major_axis_length * pixel_to_micron
                            minor_axis_length = region.minor_axis_length * pixel_to_micron
                            aspect_ratio = (major_axis_length / minor_axis_length) if region.minor_axis_length != 0 else np.nan
                            
                            # Build a dictionary of properties for this floc (region)
                            properties = {
                                'Folder': folder_name,
                                'Image': filename,
                                'Particle_Num': f'{particle_num:03d}',
                                f'Area_{thresh}': area,
                                f'Equivalent Diameter_{thresh}': equivalent_diameter,
                                f'Perimeter_{thresh}': perimeter,
                                f'Major Axis Length_{thresh}': major_axis_length,
                                f'Minor Axis Length_{thresh}': minor_axis_length,
                                f'Aspect Ratio_{thresh}': aspect_ratio
                            }
                            results.append(properties)
                            particle_num += 1
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    #results_df.to_csv('floc_properties.csv', index=False)
    
    # --------------- Additional Analysis Section ---------------
    # Prepare the equivalent length DataFrame using the "Equivalent Diameter" column.
    # Since our column is named with the threshold value, rename it for further analysis.
    eq_col = f'Equivalent Diameter_{thresh}'
    equivalent_length_df = results_df[['Folder', 'Image', eq_col]].copy()
    equivalent_length_df.rename(columns={'Folder': 'Time', eq_col: 'Equivalent_Diameter'}, inplace=True)
    
    # Cut the 'Equivalent_Diameter' values into the specified bins
    equivalent_length_df['diameter_range'] = pd.cut(
        equivalent_length_df['Equivalent_Diameter'],
        bins=bins,
        labels=labels
    )
    
    # Group by 'Time' and 'diameter_range' to count the occurrences
    grouped_df = equivalent_length_df.groupby(['Time', 'diameter_range']).size().unstack(fill_value=0).reset_index()
    
    # Ensure all labels are present
    for lab in labels:
        if lab not in grouped_df.columns:
            grouped_df[lab] = 0
    
    # Convert folder names (Time) to integer and compute Tf as Time*60
    grouped_df['Tf'] = grouped_df['Time'].astype(int) * 60
    # Rename one of the groups to 'No_20' (if needed; adjust based on your requirements)
    grouped_df.rename(columns={'Group1': 'No_20'}, inplace=True)
    #grouped_df = grouped_df.sort_values(by='Time')
    grouped_df = grouped_df.sort_values(by='Time').reset_index(drop=True)

    return grouped_df
#grouped_df = grouped_df.sort_values(by='Time')
# Example usage:
# Define your bins and labels (adjust these based on your analysis needs)
#bins = [0.27, 0.916, 1.562, 2.208]
#labels = ['Group1', 'Group2', 'Group3']  # Example labels

# Function to perform PSO optimization
def perform_pso_optimization(Flocs_df, Gf):
    Flocs_df['Gf'] = Gf
    Flocs_df['Time_min'] = Flocs_df['Tf'] / 60  # Convert to minutes
    Flocs_df.sort_values(by='Time_min', inplace=True)
    Flocs_df['ni_no'] = Flocs_df['No_20'].max() / Flocs_df['No_20']
    
    def A_K(Tf, Gf, Ka, Kb):
        return (Kb / Ka * Gf + (1 - Kb / Ka * Gf) * np.exp(-Ka * Gf * Tf)) ** -1

    # def A_K(Tf, Ka, Kb):
    #     return 1 / ( (Kb / (Ka * Gf)) + (1 - (Kb / (Ka * Gf))) * np.exp(-Ka * Gf * Tf) )
    
    Tf = Flocs_df['Tf'].values
    ni_no = Flocs_df['ni_no'].values

    # Define the objective function for PSO
    def objective(params):
        Ka, Kb = params
        ni_no_pred = A_K(Tf, Gf, Ka, Kb)
        mse = np.mean((ni_no - ni_no_pred) ** 2)
        return mse

    # Define the bounds for PSO
    lb = [1e-13, 1e-13]
    ub = [1e-3, 1e-3]

    # Perform PSO to find the best initial guesses
    best_params, best_mse = pso(objective, lb, ub)
    Ka_best, Kb_best = best_params

    # Perform the curve fitting with the optimized initial guesses
    popt, pcov = curve_fit(
        lambda Tf, Ka, Kb: A_K(Tf, Gf, Ka, Kb),
        Tf, ni_no,
        p0=[Ka_best, Kb_best]
    )

    Ka_fitted, Kb_fitted = popt

    # Generate the fitted curve
    ni_no_fitted = A_K(Tf, Gf, Ka_fitted, Kb_fitted)

    # # Plot the results
    # plt.figure(figsize=(12, 6), dpi=600)
    # plt.plot(Flocs_df['Time_min'], ni_no, 'bo', label='Observed Data')
    # plt.plot(Flocs_df['Time_min'], ni_no_fitted, 'r-', label='Fitted Curve')
    # plt.xlabel('Time (Minutes)', fontsize=12)
    # plt.xticks(np.arange(0, max(Flocs_df['Time_min']) + 10, 10), fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.tick_params(axis='both', which='both', color='black')
    # plt.tick_params(axis='x', which='minor', color='lightgrey')
    # plt.ylabel('ni/no', fontsize=12)
    # plt.legend(fontsize=12)
    # plt.savefig('fit_plot.png')  # Save the plot to a file
    # plt.show()
    # Plot the results
    plt.figure(figsize=(6, 3), dpi=300) # dpi=200
    plt.plot(Flocs_df['Time_min'], ni_no, 'bo', label='Observed Data')
    plt.plot(Flocs_df['Time_min'], ni_no_fitted, 'r-', label='Fitted Curve')
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.xticks(np.arange(0, max(Flocs_df['Time_min']) + 10, 10), fontsize=10)
    plt.yticks(fontsize=10)
    plt.tick_params(axis='both', which='both', color='black')
    plt.tick_params(axis='x', which='minor', color='lightgrey')
    plt.ylabel('ni/no', fontsize=12)
    plt.legend(fontsize=12)
    plt.savefig('fit_plot.png')  # Save to the fitted plot
    plt.close()  # Close the figure to avoid conflicts


    return Ka_fitted, Kb_fitted

def run_reactor_simulation(
    Ka_fitted, Kb_fitted, Gf_val, R_values, m=5,  
    csv_filename="THRT_simulation_result.csv"
): #T_initial=50, T0=50, T1=100,
    """
    Runs a reactor simulation with m compartments, using fitted Ka and Kb, 
    and a specified shear velocity (Gf_val). Solves for T using both 
    Newton-Raphson and Secant methods for each R in R_values, then saves results to CSV.

    Parameters:
    -----------
    Ka_fitted : float
        Fitted aggregation constant from your PSO/curve_fit step.
    Kb_fitted : float
        Fitted breakage constant from your PSO/curve_fit step.
    Gf_val : float
        Shear velocity (gradient of velocity).
    R_values : list
        A list of R values for which you want to solve for T.
    m : int
        Number of compartments (chambers) in the reactor.
    T_initial : float
        Initial guess for Newton-Raphson method.
    T0 : float
        First initial guess for Secant method.
    T1 : float
        Second initial guess for Secant method.
    csv_filename : str
        Name of the CSV file to save the results.

    Returns:
    --------
    None
        Results are written to the specified CSV file.
    """

    # ---------------------------------------------------
    # 1) Replicate Gf, Ka, and Kb arrays for each compartment
    # ---------------------------------------------------
    Gf = np.array([Gf_val] * m)
    Ka = np.array([Ka_fitted] * m)
    Kb = np.array([Kb_fitted] * m)
    
    #R_values =  r_values#[2, 3, 10]  # Array of R values for image analysis alone

    # # Initial guesses for T
    T_initial = 50  # Initial guess for Newton-Raphson
    T0 = 50  # First initial guess for Secant Method
    T1 = 100  # Second initial guess for Secant Method
    # m = len(Gf) # Number of compartments

    # ---------------------------------------------------
    # 2) Define the f(T) function for the entire system
    # ---------------------------------------------------
    def f(T, R_specified):
        product_term = 1
        n_prev = 1  # Start ratio for the first tank
        for i in range(m):
            denom = 1 + n_prev * Kb[i] * Gf[i]**2 * T / m
            if abs(denom) < 1e-12:
                # Set a default ratio or break out, depending on your model
                ratio = np.inf  # or some other fallback value
            else:
                ratio = (1 + Ka[i] * Gf[i] * T / m) / denom
            product_term *= ratio
            n_prev = ratio        
        R_calculated = product_term
        return R_specified - R_calculated

    # ---------------------------------------------------
    # 3) Define derivative f'(T) for Newton-Raphson
    # ---------------------------------------------------
    def f_prime(T, R_specified, epsilon=1e-6):
        return (f(T + epsilon, R_specified) - f(T, R_specified)) / epsilon

    # ---------------------------------------------------
    # 4) Newton-Raphson method
    # ---------------------------------------------------
    
    def newton_raphson(T_initial, R_specified, tol=1e-6, max_iter=1000):
        T = T_initial
        for iteration in range(max_iter):
            f_value = f(T, R_specified)
            f_derivative = f_prime(T, R_specified)
            
            if abs(f_value) < tol:
                return T
            
            # Update T using Newton-Raphson formula
            T = T - f_value / f_derivative
        
        return None

    # ---------------------------------------------------
    # 5) Secant method
    # ---------------------------------------------------
   
    def secant_method(T0, T1, R_specified, tol=1e-6, max_iter=1000):
        for iteration in range(max_iter):
            f_T0 = f(T0, R_specified)
            f_T1 = f(T1, R_specified)
            
            if abs(f_T1) < tol:
                return T1
            
            # Update T using the Secant Method formula
            T_new = T1 - f_T1 * (T1 - T0) / (f_T1 - f_T0)
            
            # Shift values for the next iteration
            T0, T1 = T1, T_new
        
        return None

    # ---------------------------------------------------
    # 6) Solve for T using both methods, save to CSV
    # ---------------------------------------------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
    # Header row
        writer.writerow([
            "Date", "R", "m", "Gf", "Ka", "Kb", "T (N-R)", "T (N-R) min", 
            "T (Secant)", "T (Secant) min"
        ])

        # For each R in R_values, compute T
        for R_specified in R_values:
            T_newton = newton_raphson(T_initial, R_specified)
            T_secant = secant_method(T0, T1, R_specified)

            # Convert to minutes if not None
            T_newton_min = T_newton / 60 if T_newton is not None else None
            T_secant_min = T_secant / 60 if T_secant is not None else None

            # Write row to CSV with rounded T values (3 decimal places)
            writer.writerow([current_time,R_specified, 
            m, Gf, Ka, Kb, round(T_newton, 3) if T_newton is not None else None, 
            round(T_newton_min, 3) if T_newton_min is not None else None, 
            round(T_secant, 3) if T_secant is not None else None, 
            round(T_secant_min, 3) if T_secant_min is not None else None
        ])


    print(f"Results have been saved to '{csv_filename}'.")


