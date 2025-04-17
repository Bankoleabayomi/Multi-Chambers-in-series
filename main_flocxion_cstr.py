# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 21:33:26 2025

@author: banko
"""

import numpy as np
import pandas as pd
import shutil
from backmodule import (
    process_flocs_and_analyze,
    perform_pso_optimization,
    run_reactor_simulation,
    download_images_from_cloud
)

#Declare Model input variables
root_folder = "demo_data" #Image file source
pixel_size = 0.27 #Image pixel size in millimeter
thresh = 120 #Desired threshold value for the image segmentation
Gf = 20  # Gradient of velocity
bins = [0.27, 0.916, 1.562, 2.208] #Desired floc size ranges
labels = ['Group1', 'Group2', 'Group3'] #Labels to be assigned to groups
R_values= [2, 3, 4, 5, 6, 7, 8, 9, 10] #Desired efficiencies; E = 1-(1/R)
m=5 #Number of Reactor tanks

# Run and modifying the Flocxion CSTR Model for flocculation kinetic modelling
if __name__ == "__main__":
     #root_folder = "C:/Users/banko/Documents/Python/FlocsData"  # Update with your actual path
     #Gf = 20  # Gradient of velocity

     # Analyze image data
     Flocs_df = process_flocs_and_analyze(root_folder, thresh=thresh, 
                                            pixel_to_micron=pixel_size, bins=bins, labels=labels)   #analyze_image_data(root_folder)

     # Perform PSO optimization
     Ka_fitted, Kb_fitted = perform_pso_optimization(Flocs_df, Gf)

     print(f"Fitted Ka: {Ka_fitted}")
     print(f"Fitted Kb: {Kb_fitted}")
    
     #run_reactor_simulation(Ka_fitted=Ka_fitted, Kb_fitted=Kb_fitted, Gf_val=Gf, R_values=R_values,
     #m=m, csv_filename="THRT_simulation_result.csv") #T_initial=T_initial, T0=T0, T1=T1,


