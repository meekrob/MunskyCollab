#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation_c_elegans_3d.py

Read in a set of images in brightfield and GFP.
Create a mask around the worm.
Identify nuclei that are flourescing.
Measure the intensity of nuclei individually.
"""
import sys
import pathlib  # Library to work with file paths
import os; from os import listdir; from os.path import isfile, join
import re
import glob
from skimage.io import imread        # Module from skimage to read images as numpy arrays

print("reading directory")

if len(sys.argv) > 1:
  path_dir = sys.argv[1]
else:
   path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_1'


# figure out provenance/ID from path info
#i.e.: path_dir = 32_David_Erin_Munskylab/Izabella_data/Keyence_data/201124_JM259_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM259_L1_ELT-2_worm_1

try:
  longname,RNAi,stage,shortname = path_dir.split(os.path.sep)[-4:]
  # i.e.  201124_JM259_elt-2_Promoter_Rep_1, ELT-2_RNAi, L1, JM259_L1_ELT-2_worm_1
  datestr, genotype, labl, _, __, repnum = longname.split('_')
except ValueError:
  print("Error parsing: `%s`" % longname)
  raise

wormnumber = shortname.split('_')[-1]
full_name_prefix = f"{genotype}_{RNAi}_{stage}_Rep{repnum}_Worm{wormnumber}"

os.chdir(path_dir)
current_dir = pathlib.Path().absolute()
#path_input = current_dir.joinpath(folder_name)
path_input = current_dir
# Reads the folder with the results and import the simulations as lists
list_files_names = sorted([f for f in listdir(path_input) if isfile(join(path_input, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
path_files = [ str(path_input.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file

list_images = [imread(str(f)) for f in path_files]
# Reading the microscopy data
number_images = len(path_files)
print('Number of images in file: ', number_images)
print('The images are stored in the following folder: ', path_dir)


print("importing modules...", end="", flush=True)

# !pip install tifffile
# !pip install imagecodecs
import tifffile as tiff

# # Loading libraries
import random                        # Library to generate random numbers
import skimage                       # Library for image manipulation
import numpy as np                   # Library for array manipulation
import matplotlib.pyplot as plt      # Library used for plotting

from skimage.filters import gaussian # Module working with a gaussian filter

from skimage.measure import label, regionprops
# 
from skimage.morphology import square, dilation
from skimage import measure
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.spatial import distance

# for the cellpose step (calculating the mask), time the passes through the loop, save the data to skip the processing in
# future invocations
import time
import pickle
# 
# ! pip install opencv-python-headless==4.7.0.72
# ! pip install cellpose==2.0
from cellpose import models
from cellpose import plot
print()

"""# Reading all images and converting them into form ZYXC."""

# Separating images based on the color channel
print("Separating images based on the color channel")
selected_elements_Overlay = [element for element in path_files if 'Overlay.tif' in element]
selected_elements_CH2 = [element for element in path_files if 'CH2.tif' in element]
selected_elements_CH4 = [element for element in path_files if 'CH4.tif' in element]

# Reading all images in each list.
print("Reading all images in each list.")
list_images_CH2_full = [imread(str(f)) for f in selected_elements_CH2]  # [Y,X,3C]  Color channels 0 and 2 are empty. Use channel 1
list_images_CH2 = [img[:,:,1] for img in list_images_CH2_full] #
list_images_CH4 = [imread(str(f)) for f in selected_elements_CH4]  # [Y,X]   It appears to be to contain the same information as Overlay
list_images_Overlay = [imread(str(f)) for f in selected_elements_Overlay] # [Y,X,3C]  # It has 3 color channels but all appear to contain the same information

# Creating 3D arrays with all images in the list
print("Creating 3D arrays with all images in the list")
images_CH2_3d = np.stack(list_images_CH2)
images_C4_3d = np.stack(list_images_CH4)

# Creating a 4D array with shape ZYXC.  GFP
print("Creating a 4D array with shape ZYXC.  GFP")
array4d = np.concatenate((images_CH2_3d[np.newaxis, ...], images_C4_3d[np.newaxis, ...]), axis=0)
# Move the axis from position 0 to position 2
print("Move the axis from position 0 to position 2")
image_ZYXC = np.moveaxis(array4d, 0, 3)
print('Final image shape: ', image_ZYXC.shape, '\nGFP is channel 0 \nBrightfield is channel 1')

# Plotting maximum projection
print("Plotting maximum projection")
max_GFP = np.max(image_ZYXC[:,:,:,0],axis=0)
max_Brightfield = np.max(image_ZYXC[:,:,:,1],axis=0)

print('Range in GFP: min', np.min(max_GFP), 'max',np.max(max_GFP))
print('Range in Brightfield: min', np.min(max_Brightfield), 'max',np.max(max_Brightfield))

#@title Plotting max projections
plotname = f"{full_name_prefix}_max_projections"
print("Plotting max projections:", plotname)
color_map = 'Greys_r'
fig, ax = plt.subplots(1,2, figsize=(10, 3))
fig.suptitle(f"{full_name_prefix} max. projections")
# Plotting the heatmap of a section in the image - MISPLACED LABEL? - DK
# print("Plotting the heatmap of a section in the image")
ax[0].imshow(max_Brightfield,cmap=color_map)
ax[1].imshow(max_GFP,cmap=color_map)
ax[0].set(title='max_Brightfield'); ax[0].axis('on');ax[0].grid(False)
ax[1].set(title='max_GFP'); ax[1].axis('on');ax[1].grid(False)
plt.savefig(plotname + '.png')
plt.close()

#@title Plotting all z-slices
plotname = f"{full_name_prefix}_z-slices"
print(f"Plotting {plotname}")
number_z_slices = image_ZYXC.shape[0]
fig, ax = plt.subplots(2,number_z_slices, figsize=(25, 5))
fig.suptitle(f"{full_name_prefix} z slices")
color_map = 'Greys_r'
# Plotting the heatmap of a section in the image - MISPLACED LABEL? -DK
# print("Plotting the heatmap of a section in the image")
for i in range (number_z_slices):
    # Channel 0
    temp_image_0= image_ZYXC[i,:,:,0]
    max_visualization_value = np.percentile(temp_image_0,100)
    ax[0,i].imshow(temp_image_0,vmax=max_visualization_value,cmap=color_map)
    ax[0,i].set(title='z_slice '+ str(i)); ax[0,i].axis('off');ax[0,i].grid(False); ax[0,i].axis('tight')
    # Channel 1
    temp_image_1= image_ZYXC[i,:,:,1]
    max_visualization_value = np.percentile(temp_image_1,99)
    ax[1,i].imshow(temp_image_1,vmax=max_visualization_value,cmap=color_map)
    ax[1,i].axis('off');ax[1,i].grid(False); ax[1,i].axis('tight')
plt.savefig(plotname + '.png')
plt.close()

#@title Using cellpose to segment image
print("Using cellpose to segment image")
color_map = 'Greys_r'

model = models.Cellpose(gpu=True, model_type='cyto2') # model_type='cyto', 'cyto2' or model_type='nuclei'
list_ranges =np.linspace(20, 300, 10, dtype='int')  #[50,100,150,200,250]
list_masks = []
masks_total = np.zeros_like(max_Brightfield)

total_time = 0
pickled_mask_path = os.path.join(current_dir, f"{datestr}_{genotype}_{RNAi}_{repnum}_mask.pickle")

if os.path.exists( pickled_mask_path ):
   print("reading pickle...", end=" ")
   with open(pickled_mask_path, "rb") as inpickle:
    masks_total = pickle.load(inpickle)
   print("done")
else:
  print("calculating mask")
  for i,diameter in enumerate (list_ranges):
    begin_time = time.time()
    print("\tmodel.eval(max_Brightfield ...) %d/%d " % (i+1,len(list_ranges)), flush=True, end='')

    masks = model.eval(max_Brightfield, diameter=diameter, flow_threshold=1, channels=[0,0], net_avg=True, augment=True)[0]
    masks_total = masks_total+ masks

    end_time = time.time()
    total_time += end_time - begin_time
    print(round(end_time - begin_time), "seconds.", round(total_time), "total.")
  with open(pickled_mask_path,"wb") as outpickle:
     print("writing", pickled_mask_path, "...", end="")
     pickle.dump(masks_total, outpickle)
     print("done")

print("Total time: %d seconds" % round(total_time))

# Binarization
print("Binarization")
new_mask = masks_total.copy()
new_mask[new_mask>0]=1
dilated_mask = dilation(new_mask, square(30))

# Removing elements that are not part of the main mask
radius_threshold = 50
# Label connected regions in the mask
labeled_mask = label(dilated_mask)
# Get region properties of each labeled region
props = regionprops(labeled_mask)
# Create a new mask to store regions with radius greater than the threshold
final_mask = np.zeros_like(dilated_mask, dtype=np.uint8)
# Iterate over the region properties
for prop in props:
    # Get the radius of the region
    radius = prop.equivalent_diameter / 2
    # Check if the radius is greater than the threshold
    if radius > radius_threshold:
        # Get the coordinates of the region
        coords = prop.coords
        # Set the region in the new mask
        final_mask[coords[:, 0], coords[:, 1]] = 1
# Mask by image
segmented_image = np.multiply(final_mask,max_Brightfield)

# Find center of mass
cm = center_of_mass(segmented_image)

# Plotting
plotname = f"{full_name_prefix}_brightfield_w_mask"
fig, ax = plt.subplots(1,3, figsize=(15, 5))
fig.suptitle(f"{full_name_prefix} brightfield w/ mask")
# Plotting the heatmap of a section in the image
ax[0].imshow(max_Brightfield,cmap=color_map)
ax[1].imshow(final_mask,cmap=color_map)
ax[2].imshow(segmented_image,cmap=color_map)
ax[0].set(title='brightfield'); ax[0].axis('on');ax[0].grid(False)
ax[1].set(title='Mask'); ax[1].axis('on');ax[1].grid(False)
ax[2].set(title='brightfield * mask'); ax[2].axis('on');ax[0].grid(False)
plt.savefig(plotname + '.png')
plt.close()

"""# Nuclei segmentation using trackpy"""
print("Nuclei segmentation using trackpy")

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# ! pip install trackpy
import trackpy as tp # Library for particle tracking

GFP = max_GFP.copy()

particle_size = 21 # according to the documentation must be an odd number 3,5,7,9 etc.
#fig, ax = plt.subplots(1,1, figsize=(4, 4))
spots_detected_dataframe_all = tp.locate(GFP, diameter=particle_size, minmass=0)

# This section generates an histogram with the intensity of the detected particles in the image.
fig, ax = plt.subplots(1,1, figsize=(4, 4))
plotname = f"{full_name_prefix}_histogram_mass"
print(f"plotting {plotname}")
fig.suptitle(f"{full_name_prefix}\nhistogram of intensities")
ax.hist(spots_detected_dataframe_all['mass'], bins=50, color = "orangered", ec="orangered")
ax.set(xlabel='mass', ylabel='count')
plt.savefig(plotname + '.png')
plt.close()

plotname = f"{full_name_prefix}_spots_detected"
print("Plotting", plotname)
plt.figure(figsize=(5,4))
plt.suptitle(f"{full_name_prefix} spots detected")
spots_detected_dataframe = tp.locate(GFP,diameter=particle_size, minmass=400) # "spots_detected_dataframe" is a pandas data freame that contains the infomation about the detected spots
tp.annotate(spots_detected_dataframe,GFP,plot_style={'markersize': 1.5})  # tp.anotate is a trackpy function that displays the image with the detected spots
plt.savefig(plotname + '.png')
plt.close()

# save a file
spots_detected_dataframe['Worm'] = wormnumber
spots_detected_dataframe['Rep'] = repnum
spots_detected_dataframe['RNAi'] = RNAi
spots_detected_dataframe['Genotype'] = genotype
spots_detected_dataframe.to_csv(f"{datestr}_{genotype}_{RNAi}_{repnum}_segmented_plot.csv")

number_of_detected_cells = len(spots_detected_dataframe)
number_of_detected_cells

"""# Calculating total intensities. Sum of intensity in all pixels inside of a cell mask."""
print("Calculating total intensities. Sum of intensity in all pixels inside of a cell mask.")

# Total intensity values
plotname = f"{full_name_prefix}_total_intensity_hist"
print("Plotting Total intensity values", plotname)
fig, ax = plt.subplots(1,1, figsize=(5, 5))
fig.suptitle(f"{full_name_prefix} histogram of\ntotal intensities inside cell mask")
ax.hist(spots_detected_dataframe['mass'], bins=15, color = "orangered", ec="orangered")
ax.set(xlabel='total intensity', ylabel='count')
plt.savefig(plotname + '.png')
plt.close()

# Nuclei size
plotname = f"{full_name_prefix}_nuclei_size_hist"
print("Plotting Nuclei size", plotname)
fig, ax = plt.subplots(1,1, figsize=(5, 5))
fig.suptitle(f"{full_name_prefix} histogram of nuclei size")
ax.hist(spots_detected_dataframe['size'], bins=15, color = "orangered", ec="orangered")
ax.set(xlabel='nuclei size', ylabel='count')
plt.savefig(plotname + '.png')
plt.close()

