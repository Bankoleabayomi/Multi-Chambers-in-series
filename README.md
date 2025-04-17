# Flocxion CSTR
This repository contains Python scripts for Automatic modelling of flocculation kinetics and simulation of CSTR retention time.
The CSTR (Completely Stirred Tank Reactor) simulation is performed through the walkflow designed from our study on the flocculation kinetics.
The computing depencies specified in the requirement.txt file is essential for easy adaptation of this framework.
The walkflow starts from the segmentation of floc images acquired from a suitable flocculation monitoring imaging technique (a Non-intrusive Dynamic Image Analysis in this case).
The remarkable Argaman and Kaufman model is used to guide the Levenberge-Marqurts Non-linear Curve fitting algorithm which is optimised by Swarm Intelligence algorithm.
The Total Hydraulic Retention Time (THRT) is simulated using the Secant technique and verified with the Newton Raphson method.
Users are to declare the major input variables which are listed below;
Pixel size
Threshold value
Shear stress / Velocity gradient used for the flocculation assay
Particle size range (this is informed by the pixel size and user's interest)
Treatment efficiency (R). See the official article for detail information.
Number of Chambers in series.

Note: This is an ongoing project and updates shall be implemented on this page. Future update shall include the modelling with varying distribution of G values across the chambers.

