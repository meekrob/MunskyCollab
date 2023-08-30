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
import time

DO_MULTI = True
    
#def do_mask_mp(model, max_Brightfield, diameter, flow_threshold=1, channels=[0,0], net_avg=True, augment=True, thread_number = -1):
def do_mask_mp(arglist):
  model, max_Brightfield, diameter, flow_threshold, channels, net_avg, augment, thread_number = arglist
  pid = os.getpid()
  print(f"Starting task: {thread_number} on {pid}. diameter ${diameter}. ", flush=True)
  
  begin_time = time.time()
  masks = model.eval(max_Brightfield, 
                     diameter=diameter, 
                     flow_threshold=flow_threshold, 
                     channels=channels, 
                     net_avg=net_avg, augment=augment)[0]
  
  end_time = time.time()
  duration = round(end_time - begin_time)
  print(f"Finishing task {thread_number} on {pid}:  diameter ${diameter}. {duration} seconds", flush = True)
  return masks

def parse_input_directory(splitpath):
#                         ^ last four parts (/ -separated) of Izabella's directory structure
# complex fields about each image set are specified in the path
  try:
    longname,RNAi,stage,shortname = splitpath
    # i.e.  201124_JM259_elt-2_Promoter_Rep_1, ELT-2_RNAi, L1, JM259_L1_ELT-2_worm_1
    datestr, genotype, labl, _, __, repnum = longname.split('_')
    wormnumber = shortname.split('_')[-1]
  except ValueError:
    print("Error parsing: `%s`" % longname)
    raise

  return datestr,RNAi,stage,genotype,repnum,wormnumber

def main():

  print("reading directory")

  if len(sys.argv) > 1:
    path_dir = sys.argv[1]
  else:
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_1'
    path_dir = '/Users/david/work/MunskyColab/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_5'

#path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data
# ...   /201002_JM149_elt-2_Promoter_Rep_1 / L4440_RNAi / L1 / JM149_L1_L4440_worm_1
#       /               -4                 /   -3       / -2 /         -1
#       datestr genotype, X,    X,   X, repnum                     X _ X_  X  _ X  _ wormnumber
  datestr,RNAi,stage,genotype,repnum,wormnumber = parse_input_directory(path_dir.split(os.path.sep)[-4:])
  full_name_prefix = f"{genotype}_{RNAi}_{stage}_Rep{repnum}_Worm{wormnumber}"

  
  print("import images", end='')
  os.chdir(path_dir)
  current_dir = pathlib.Path().absolute()
  path_input = current_dir
  # Reads the folder with the results and import the simulations as lists
  list_files_names = sorted([f for f in listdir(path_input) if isfile(join(path_input, f)) and ('.tif') in f], key=str.lower)  # reading all tif files in the folder
  list_files_names.sort(key=lambda f: int(re.sub('\D', '', f)))  # sorting the index in numerical order
  path_files = [ str(path_input.joinpath(f).resolve()) for f in list_files_names ] # creating the complete path for each file

  list_images = [imread(str(f)) for f in path_files]
  # Reading the microscopy data
  number_images = len(path_files)
  print('...done')
  print('Number of images in file: ', number_images)
  print('The images are stored in the following folder: ', path_dir)

  # Additional modules that take some time to initialize
  print("importing modules...", end=" ", flush=True)
  if DO_MULTI:
    import multiprocessing


  # # Loading libraries
  import tifffile as tiff
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
  import pickle
  # 
  # ! pip install opencv-python-headless==4.7.0.72
  # ! pip install cellpose==2.0
  from cellpose import models
  from cellpose import plot
  print("done")

  
  # Separating images based on the color channel
  print("Separating images based on the color channel")
  selected_elements_Overlay = [element for element in path_files if 'Overlay.tif' in element]
  selected_elements_CH2 = [element for element in path_files if 'CH2.tif' in element]
  selected_elements_CH4 = [element for element in path_files if 'CH4.tif' in element]

  # Reading all images in each list.
  print("Reading all images in each list.")
  list_images_CH2_full = [imread(str(f)) for f in selected_elements_CH2]  # [Y,X,3C]  Color channels 0 and 2 are empty. Use channel 1
  list_images_CH2 = [img[:,:,1] for img in list_images_CH2_full] #
  list_images_CH4 = [imread(str(f)) for f in selected_elements_CH4]  # [Y,X]   It appears to contain the same information as Overlay
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
  plotname = f"{full_name_prefix}_max_projections"
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
  if DO_MULTI:
    begin_time = time.time()

    with multiprocessing.Pool() as pool:
      arglist = [[model, max_Brightfield, diameter, 1, [0,0], True, True,i]  for i, diameter in enumerate(list_ranges)]
      mask_output = pool.map(do_mask_mp,arglist)
    print("Adding masks together", end='')
    for masks in mask_output:
      masks_total = masks_total + masks
    print
    end_time = time.time()
    total_time += end_time - begin_time

  elif os.path.exists( pickled_mask_path ):
    begin_time = time.time()
    print("reading pickle...", end=" ")
    with open(pickled_mask_path, "rb") as inpickle:
      masks_total = pickle.load(inpickle)
    print("done")
    end_time = time.time()
    total_time += end_time - begin_time

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
      begin_time = time.time()
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


  # Try to figure out which way the worm is pointing so we can rotate everything
  x, y = segmented_image.nonzero()
  
  # horizontal line through worm
  A = np.vstack([x, np.ones(len(x))]).T
  horiz_fit = np.linalg.lstsq(A, y, rcond=None)
  m, c = horiz_fit[0]
  print(f"Horizontal: slope {m}, intercept {round(c)}")

  # vertical line through worm
  A = np.vstack([y, np.ones(len(y))]).T
  vertical_fit = np.linalg.lstsq(A, x, rcond=None)
  m, c = vertical_fit[0]
  print(f"Vertical: slope {m}, intercept {round(c)}")

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

  import trackpy as tp # Library for particle tracking
  GFP = max_GFP.copy()

  # This section generates an histograme with the intensity of the detected particles in the image.
  particle_size = 21 # according to the documentation must be an odd number 3,5,7,9 etc.
  spots_detected_dataframe_all = tp.locate(GFP, diameter=particle_size, minmass=0)

  plotname = f"{full_name_prefix}_histogram_mass"
  print(f"plotting {plotname}")
  fig.suptitle(f"{full_name_prefix}\nhistogram of intensities")
  ax.hist(spots_detected_dataframe_all['mass'], bins=50, color = "orangered", ec="orangered")
  ax.set(xlabel='mass', ylabel='count')
  plt.savefig(plotname + '.png')
  plt.close()

  # The selected spots with 21,400
  plotname = f"{full_name_prefix}_spots_detected"
  print("Plotting", plotname)
  fig, ax = plt.subplots(1,1, figsize=(5, 4))
  fig.suptitle(f"{full_name_prefix} spots detected")
  spots_detected_dataframe = tp.locate(GFP,diameter=particle_size, minmass=400) # "spots_detected_dataframe" is a pandas data freame that contains the infomation about the detected spots
  tp.annotate(spots_detected_dataframe,GFP,plot_style={'markersize': 1.5}, ax=ax)  # tp.anotate is a trackpy function that displays the image with the detected spots
  plt.savefig(plotname + '.png')
  plt.close()


  # Plot GFP and the tp.annotate graph together
  plotname = f"{full_name_prefix}_comparison"
  print("GFP and segmentation together", plotname)
  color_map = 'Greys_r'


  # try multiple parameters
  minmasses = range(200,450,50)
  particle_sizes = [15,17,19,21]

  fig, ax = plt.subplots(len(minmasses),len(particle_sizes)+1, 
                       figsize=(6*len(minmasses), 
                                6*len(particle_sizes)), 
                                dpi=300)
  fig.suptitle(f"{full_name_prefix} parameter comparison")

  # GFP image will be in upper left
  ax[0,0].imshow(max_GFP,cmap=color_map)
  ax[0,0].set(title='max_GFP')
  ax[0,0].axis('on')
  ax[0,0].grid(False)

  # these are blank slots
  for i in range(1, len(minmasses)):
     ax[i,0].axis('off')
     ax[i,0].grid(False)

  for i, mm in enumerate(minmasses):
    for j, particle_size in enumerate(particle_sizes):
      print(f"i: {i}, particle_size: {particle_size}; j: {j}, minmass: {mm}")
      spots_detected_dataframe = tp.locate(GFP,diameter=particle_size, minmass=mm) 

      x = list(spots_detected_dataframe.loc[:,'x'])
      y = list(spots_detected_dataframe.loc[:,'y'])
      markersizes = list(spots_detected_dataframe.loc[:,'size'] * 1.5) 
  #    tp.annotate(spots_detected_dataframe,GFP,plot_style={'markersize': markersizes, 'markeredgewidth':.5},ax=ax[i,j+1])
      _imshow_style = dict(origin='lower', interpolation='nearest',
                           cmap=plt.cm.gray)
      ax[i,j+1].imshow(GFP, **_imshow_style)
      ax[i,j+1].set_xlim(-0.5, GFP.shape[1] - 0.5)
      ax[i,j+1].set_ylim(-0.5, GFP.shape[0] - 0.5)
      ax[i,j+1].scatter(x, y, s=markersizes, edgecolors="r", linewidths=2, alpha=.5)
      bottom, top = ax[i,j+1].get_ylim()
      if top > bottom:
        ax[i,j+1].set_ylim(top, bottom, auto=None)
      ax[i,j+1].set(title=f'{particle_size};{mm}')
  
  
  plt.savefig(plotname + '.png')
  plt.close()

  # Add experiment info about the worm
  spots_detected_dataframe['Worm'] = wormnumber
  spots_detected_dataframe['Rep'] = repnum
  spots_detected_dataframe['RNAi'] = RNAi
  spots_detected_dataframe['Genotype'] = genotype

  rel_to_center = spots_detected_dataframe.loc[:,['x','y']]-cm
  spots_detected_dataframe['x_rel_to_center'] = rel_to_center.iloc[:,0]
  spots_detected_dataframe['y_rel_to_center'] = rel_to_center.iloc[:,1]
  spots_detected_dataframe['distance_from_center'] = (rel_to_center**2).sum(axis=1)**.5 # euclidean
  spots_detected_dataframe.to_csv(f"{datestr}_{genotype}_{RNAi}_{repnum}_dist_segmented.csv")

  number_of_detected_cells = len(spots_detected_dataframe)
  print("We got", number_of_detected_cells, "nuclei/spots!")

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

if __name__ == "__main__":
  if DO_MULTI:
    import multiprocessing
    multiprocessing.freeze_support()
  main()
