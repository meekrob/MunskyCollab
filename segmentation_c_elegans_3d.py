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
from skimage import transform
import tifffile as tiff
import math
# # Loading libraries
import random                        # Library to generate random numbers
import skimage                       # Library for image manipulation
import numpy as np                   # Library for array manipulation
import matplotlib.pyplot as plt      # Library used for plotting
from matplotlib import transforms
from skimage.filters import gaussian # Module working with a gaussian filter
from skimage.measure import label, regionprops
from skimage.morphology import square, dilation
from skimage import measure
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.spatial import distance

# for the cellpose step (calculating the mask), time the passes through the loop, save the data to skip the processing in
# future invocations
import time
import pickle
from cellpose import models
from cellpose import plot

def make_pickle_filename(key):
   genotype, rep, stage, RNAi, worm = key
   return f"{genotype}_{rep}_{stage}_{RNAi}_{worm}.pick"

def make_key(genotype, rep, stage, RNAi, worm):
   return genotype, rep, stage, RNAi, worm

def init_data(genotype, rep, stage, RNAi, worm):
   k = make_key(genotype, rep, stage, RNAi, worm)
   data = {} 
   data[k] = {}
   # add objects to "data[k]", but call save_data(data). This will allow the key to be included in the dict.
   return k, data[k], data 

def save_data(data):
   key = list(data.keys())[0]
   pickname = make_pickle_filename(key)
   with open(pickname,"wb") as outpickle:
     print("writing", pickname, "...", end="")
     pickle.dump(data, outpickle)
     print("done")

def load_data(filename):
   print("reading pickle...", end=" ")
   with open(filename, "rb") as inpickle:
    data = pickle.load(inpickle)
    print("done")
   key = list(data.keys())[0]
   return key, data[key]
# helper functions to handle coordinates

def in_range(x, low, high):
  if x >= low and x <= high: return x
  return None

def line_in_box(line, x0, y0, w, h):
  # return (x,y) the coordinates of the line intersecting
  # the box in the order: bottom, left, top, right
  # If the line does not intersect the given boundary,
  # the tuple is None.
  # An intersecting line has two intersected boundaries,
  # none if it doesn't (returns None, None, None, None)
  m, b = line

  # x or y coordinate will be None if out of range
  # otherwise it's the point of intersection for the
  # given boundary
  bottom = (in_range(-b/m, 0, w), 0)
  right = (w, in_range(m*w + b, 0, h))
  left = (0, in_range(b, 0, h))
  top = (in_range((h-b)/m, 0, w), h)

  # for any out-of-range coordinates, use None for the whole tuple
  if bottom[0] is None: bottom = None
  if right[1] is None: right = None
  if left[1] is None: left = None
  if top[0] is None: top = None

  return bottom, left, top, right

def isect_line_box(line, x0, y0, w, h):
  bltr = line_in_box(line, x0, y0, w, h)
  x = [ coord[0] for coord in bltr if coord is not None]
  y = [ coord[1] for coord in bltr if coord is not None]
  # returns empty lists if the line doesn't intersect
  return x, y

# substitute for trackpy.annotate: more choices in graphing 
def annotate_spots(df, GFP, ax, plot_styles = {}):
  # default arguments to https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
  scatter_args = {
  'edgecolors' : "r",
  'linewidths' : 2,
  'alpha' : .5,
  'marker' : 'o',
  's': 5,
  'facecolors' : 'g'
  }
  plot_styles = plot_styles.copy()
  for k in scatter_args.keys():
    if k in plot_styles:
      scatter_args[k] = plot_styles[k]
      del plot_styles[k]

  print(scatter_args)

  x = list(df.loc[:,'x'])
  y = list(df.loc[:,'y'])
  markersizes = list(df.loc[:,'size'] * 1.5)

  
  print("plot_styles ", plot_styles)

  # basically transferred this from trackpy.annotate, allowing for more control
  _imshow_style = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
  ax.imshow(GFP, **_imshow_style)
  ax.set_xlim(-0.5, GFP.shape[1] - 0.5)
  ax.set_ylim(-0.5, GFP.shape[0] - 0.5)
  ax.scatter(x, y, s = scatter_args['s'], edgecolors = scatter_args['edgecolors'], 
             linewidths = scatter_args['linewidths'], 
             alpha = scatter_args['alpha'],
             marker = scatter_args['marker'], facecolors = scatter_args['facecolors'],
             **plot_styles)
  
  bottom, top = ax.get_ylim()
  if top > bottom:
    ax.set_ylim(top, bottom, auto=None)

  return ax


# from Luis
def spots_in_mask(df,masks):
    # extracting the contours in the image
    coords = np.array([df.y, df.x]).T # These are the points detected by trackpy
    coords_int = np.round(coords).astype(int)  # or np.floor, depends
    values_at_coords = masks[tuple(coords_int.T)] # If 1 the value is in the mask
    df['In Mask']=values_at_coords # Check if pts are on/in polygon mask
    condition = df['In Mask'] ==1
    selected_rows = df[condition]
    
    return selected_rows.drop(columns=['In Mask'])

def transform_and_crop(image, tform):
  tformed = transform.warp(image, tform)
  x,y = tformed.nonzero()
  return tformed[:max(x),:max(y)]

def main():
  print("reading directory")

  if len(sys.argv) > 1:
    path_dir = sys.argv[1]
  else:
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_1'
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201124_JM259_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM259_L1_ELT-2_worm_4'
    path_dir = '/Users/david/work/MunskyColab/data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_3'


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
  k, datasave, data = init_data(genotype, repnum, stage, RNAi, wormnumber)


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

  datasave['max_GFP'] = max_GFP
  datasave['max_Brightfield'] = max_Brightfield

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
      temp_image_1 = image_ZYXC[i,:,:,1]
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
    begin_time = time.time()
    with open(pickled_mask_path, "rb") as inpickle:
      masks_total = pickle.load(inpickle)
    end_time = time.time()
    print("done")
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
  radius_threshold = 250
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
  datasave['final_mask'] = final_mask
  datasave['segmented_image'] = segmented_image
  
  plot_center = np.array(segmented_image.T.shape)[:2]/2
  print(segmented_image.shape)
  print(-plot_center)
  shift_to_plot_center = transform.EuclideanTransform(translation=-plot_center)
  cm = center_of_mass(segmented_image.T)
  print(cm)
  pc = np.array(segmented_image.T.shape)[:2]/2
  print(pc)

  """## Linear regression"""

  minmax = lambda x: (min(x), max(x))
  x,y = segmented_image.T.nonzero()
  print("min/max y:", minmax(y))
  print("min/max x:", minmax(x))
  # horizontal line through worm
  A = np.vstack([x, np.ones(len(x))]).T
  horiz_fit = np.linalg.lstsq(A, y, rcond=None)
  horiz_slope, horiz_intercept = horiz_fit[0]
  print(f"Horizontal: {horiz_slope=}, intercept {round(horiz_intercept)}, SS Resid {horiz_fit[1]}")

  # vertical line through worm
  #x, y = segmented_image.T.nonzero()
  A = np.vstack([y, np.ones(len(y))]).T
  vertical_fit = np.linalg.lstsq(A, x, rcond=None)
  vertical_slope, vertical_intercept = vertical_fit[0]
  print(f"Vertical: {vertical_slope=}, intercept {round(vertical_intercept)}, SS Resid {vertical_fit[1]}")

  horiz_line = lambda a: a*horiz_slope + horiz_intercept
  vertical_line = lambda a: a*vertical_slope + vertical_intercept

  horiz_line_inv = lambda a: (a-horiz_intercept)/horiz_slope
  vertical_line_inv = lambda a: (a-vertical_intercept)/vertical_slope

  m,b = vertical_fit[0]
  print(f"{m=}, {b=}")
  inverse_m = 1 / m
  print(f"{inverse_m=}")
  horiz_angle_correction = math.atan(horiz_slope)
  inverse_vertical_angle_correction = math.atan(inverse_m)
  print(f"{math.degrees(horiz_angle_correction)=}")
  print(f"{math.degrees(inverse_vertical_angle_correction)=}")

  img_height, img_width = segmented_image.shape

  xmin = 0
  xmax = img_width
  ymin = 0
  ymax = img_height

  # horiz
  print(f"{horiz_fit[0]=}")
  horiz_xbounds, horiz_ybounds = isect_line_box(horiz_fit[0], 0, 0, img_width, img_height)
  horiz_xseries = np.linspace(horiz_xbounds[0], horiz_xbounds[1],3)
  horiz_yseries = np.linspace(horiz_ybounds[0], horiz_ybounds[1],3)
  print("horizontal line coordinates")
  print(f"{horiz_xseries=}")
  print(f"{horiz_yseries=}")

  # vert
  print(f"{vertical_fit[0]=}")
  vertical_ybounds, vertical_xbounds = isect_line_box(vertical_fit[0], 0, 0, img_height, img_width)
  vertical_xseries = np.linspace(vertical_xbounds[0], vertical_xbounds[1],3)
  vertical_yseries = np.linspace(vertical_ybounds[0], vertical_ybounds[1],3)
  print("vertical line coordinates")
  print(f"{vertical_xseries=}")
  print(f"{vertical_yseries=}")

  # center of fitted line (horizontal)
  cf_horiz = (horiz_xseries[1],horiz_yseries[1])

  # center of fitted line (vertical)
  cf_vertical = (vertical_xseries[1],vertical_yseries[1])

  # plot 4 panels, top row horiz, vertical line fits
  fig, ax  = plt.subplots(3,2, figsize=(15,10))
  plotname = f"{full_name_prefix}_rotation"
  fig.suptitle(f"{full_name_prefix} rotation/translation")
  ax[0,0].imshow(segmented_image,cmap=color_map)
  ax[0,0].scatter(cf_horiz[0], cf_horiz[1],  s=10, edgecolors='b')
  ax[0,0].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[0,0].plot(horiz_xseries, horiz_yseries,'r')
  ax[0,0].set(title='horizontal line fit')

  # vertical
  ax[0,1].imshow(segmented_image,cmap=color_map)
  ax[0,1].scatter(cf_vertical[0], cf_vertical[1], s=10, edgecolors='b')
  ax[0,1].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[0,1].plot(vertical_xseries, vertical_yseries,'r')
  ax[0,1].set(title='vertical line fit')


 
  # shift the whole image from the fitted line midpoint to the origin
  cf_horiz_shift = transform.EuclideanTransform(translation=[cf_horiz[0],cf_horiz[1]])
  cf_vertical_shift = transform.EuclideanTransform(translation=[-cf_vertical[0],-cf_vertical[1]])

  print("translate center of horizontal fit", cf_horiz_shift)
  print("translate center of vertical fit", cf_vertical_shift)

  # shift the origin back to the plot center
  shift_to_plot_center = transform.EuclideanTransform(translation= [plot_center[0],plot_center[1]])
  print("shift_to_plot_center", shift_to_plot_center)

  # horizontal angle
  angleH = math.atan(horiz_slope)
  print("horiz angle", math.degrees(angleH))
  rotation = transform.EuclideanTransform(rotation=angleH)
  matrix_h = cf_horiz_shift.params @ rotation.params @ np.linalg.inv(shift_to_plot_center.params)
  tform_h = transform.EuclideanTransform(matrix_h)
  print("tform horiz", tform_h)
  tf_img_h = transform.warp(segmented_image, tform_h)

  # vertical
  angleV = inverse_vertical_angle_correction

  print("vertical angle", math.degrees(angleV))
  rotation = transform.EuclideanTransform(rotation=angleV)
  matrix_v =  shift_to_plot_center.params @ np.linalg.inv(rotation.params) @ cf_vertical_shift.params
  tform_v = transform.EuclideanTransform(matrix_v)
  print("tform vertical", tform_v)
  tf_img_v = transform.warp(segmented_image, tform_v.inverse)

  ax[1,0].imshow(tf_img_h, cmap=color_map)
  ax[1,0].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[1,0].set(title='rotate %d degrees' % math.degrees(angleH))
  ax[1,1].imshow(tf_img_v, cmap=color_map)
  ax[1,1].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[1,1].set(title='rotate %d degrees' % math.degrees(angleV))

  # choose the transform coming from the better fit
  if horiz_fit[1] < vertical_fit[1]:
    print("using horiz fit")
    matrix = matrix_h
    tform = tform_h
  else:
    print("using vertical fit")
    matrix = np.linalg.inv(matrix_v)
    tform = tform_v.inverse

  # rotate onto a much larger canvas
  rotated_image = transform.warp(segmented_image, tform, output_shape = (img_height * 2, img_width * 2))
  ax[2,0].imshow(rotated_image, cmap=color_map)
  ax[2,0].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[2,0].set(title='Chosen rotation')

  x,y = rotated_image.nonzero()
  ax[2,0].axvline(x = min(y), color = 'w', linestyle = '-')
  ax[2,0].axhline(y = min(x), color = 'w', linestyle = '-')

  # prepare to crop by shifting bounding box to origin (it will also be necessary to shift the spot coordinates)
  shift_bounding_box_to_origin = transform.EuclideanTransform(translation = (min(y), min(x)))

  final_matrix = matrix @ shift_bounding_box_to_origin.params
  final_tform = transform.EuclideanTransform(final_matrix)
  datasave['final_matrix'] = final_matrix
  datasave['final_tform'] = final_tform

  rotated_image = transform.warp(segmented_image, final_tform, output_shape = (img_height * 2, img_width * 2))

  x,y = rotated_image.nonzero()
  print("after rotation min/max y:", minmax(y))
  print("after rotation min/max x:", minmax(x))
  cropped_image = rotated_image[:max(x),:max(y)]
  datasave['cropped_image'] = cropped_image
  ax[2,1].imshow(cropped_image, cmap=color_map)
  ax[2,1].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
  ax[2,1].set(title='rotated and cropped')
  ax = ax[2,1]; ax.axis('tight');# ax.axis('off'); ax.grid(False)
  plt.savefig(plotname + '.png')
  plt.close()

  """# Nuclei segmentation using trackpy"""
  print("Nuclei segmentation using trackpy")

  import trackpy as tp # Library for particle tracking

  GFP = max_GFP.copy()

  particle_size = 21 # according to the documentation must be an odd number 3,5,7,9 etc.
  #fig, ax = plt.subplots(1,1, figsize=(4, 4))
  spots_detected_dataframe_all = tp.locate(GFP, diameter=particle_size, minmass=0)

  # This section generates a histogram with the intensity of the detected particles in the image.
  fig, ax = plt.subplots(1,1, figsize=(4, 4))
  plotname = f"{full_name_prefix}_histogram_mass"
  print(f"plotting {plotname}")
  fig.suptitle(f"{full_name_prefix}\nhistogram of intensities")
  ax.hist(spots_detected_dataframe_all['mass'], bins=50, color = "orangered", ec="orangered")
  ax.set(xlabel='mass', ylabel='count')
  plt.savefig(plotname + '.png')
  plt.close()


  # Plot GFP and the tp.annotate graph together
  plotname = f"{full_name_prefix}_comparison"
  print("GFP and segmentation together", plotname)
  color_map = 'Greys_r'

  SPOT_PARAMS = (472,25)

  if SPOT_PARAMS is None:
    # Optimization from Luis!
    # Creating vectors to test all conditions for nuclei detection.
    number_optimization_steps = 10
    particle_size_vector = [num for num in range(13, 25 + 1) if num % 2 != 0][:number_optimization_steps]
    print('particle_size_vector: ', particle_size_vector)
    minmass_vector = np.linspace(250, 500, num=number_optimization_steps, endpoint=True,dtype=int)
    print('minmass_vector: ', minmass_vector)

    fig, ax = plt.subplots(len(minmass_vector),len(particle_size_vector)+1, 
                          figsize=(6*len(minmass_vector), 
                                    6*len(particle_size_vector)), 
                                    dpi=300)
    fig.suptitle(f"{full_name_prefix} parameter comparison")


    # Left column is a replot of previous steps, subsequent columns are 
    # spot finding at different params
    ax[0,0].imshow(max_Brightfield,cmap=color_map)
    ax[1,0].imshow(final_mask,cmap=color_map)
    ax[2,0].imshow(segmented_image,cmap=color_map)
    ax[3,0].imshow(max_GFP,cmap=color_map)
    # ax[4,0] will be a plot of the selected params

    ax[0,0].set(title='brightfield')
    ax[1,0].set(title='Mask')
    ax[2,0].set(title='brightfield * mask')
    ax[3,0].set(title='max GFP')
    # ax[4,0] will be a plot of the selected params

    # keep the remaining blank slots in the first column 
    # from showing empty axes by turning them off
    for i in range(5, len(minmass_vector)):
      ax[i,0].axis('off')
      ax[i,0].grid(False)

    # optimization from Luis!
    metric = np.zeros((number_optimization_steps,number_optimization_steps))

    for i, mm in enumerate(minmass_vector):
      for j, particle_size in enumerate(particle_size_vector):
        print(f"i: {i}, particle_size: {particle_size}; j: {j}, minmass: {mm}")
        spots_detected_dataframe = tp.locate(GFP,diameter=particle_size, minmass=mm) 
        # Selecting only spots located inside mask
        df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)
        if len(df_in_mask) > 0:
          metric[i,j] = np.sum(df_in_mask['mass']) / len(df_in_mask) # maximizes the mean intensity in all spots
        else:
          metric[i,j] = 0

        annotate_spots(df_in_mask, GFP, ax[i,j+1])
        ax[i,j+1].set(title=f'{particle_size};{mm}; {len(df_in_mask)} spots')

        del spots_detected_dataframe, df_in_mask


    metric = metric.astype(int)
    print(metric)
    # selecting indces that maximize metric
    selected_minmass_index, selected_particle_size_index = np.unravel_index(metric.argmax(), metric.shape)
    selected_minmass = minmass_vector[selected_minmass_index]
    selected_particle_size = particle_size_vector[selected_particle_size_index]
    print(selected_minmass)
    print(selected_particle_size)
    datasave['minmass'] = selected_minmass
    datasave['particle_size'] = selected_particle_size

    spots_detected_dataframe = tp.locate(GFP,diameter=selected_particle_size, minmass=selected_minmass) 
    df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)
    # plot the "best" one underneath the brightfield/mask/GFP
    ax[4,0].set(title=f'Selected params: {selected_particle_size}, {selected_minmass}. {len(df_in_mask)} spots')
    # add plot with different parameters to grid    
    annotate_spots(df_in_mask, GFP, ax[4,0])
    # write the comparison figure
    plt.savefig(plotname + '.png')
    plt.close()
  else:
    selected_minmass, selected_particle_size = SPOT_PARAMS
    spots_detected_dataframe = tp.locate(GFP,diameter=selected_particle_size, minmass=selected_minmass) 
    df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)
  # rotating the image with the transform matrix based on the vertical fit
  df_mx = df_in_mask.loc[:,('x','y')]
  df_mx['1'] = 1

  rotated = (np.linalg.inv(final_matrix) @ df_mx.T).T
  df_in_mask_rotated = df_in_mask.copy()
  df_in_mask_rotated['x'] = rotated.iloc[:,0]
  df_in_mask_rotated['y'] = rotated.iloc[:,1]


  # write the selected spot find
  plotname = f"{full_name_prefix}_spots_detected"
  print("Plotting", plotname)

  fig, ax = plt.subplots(1,3, figsize=(15, 5),dpi=1000)
  fig.suptitle(f"{full_name_prefix} spots detected")

  def rotate_df(df, final_matrix):
    df_mx = df.loc[:,('x','y')]
    df_mx['1'] = 1
    rotated_mx = (np.linalg.inv(final_matrix) @ df_mx.T).T
    rotated_df = df.copy()
    rotated_df['orig_x'] = df['x']
    rotated_df['orig_y'] = df['y']
    rotated_df['x'] = rotated.iloc[:,0]
    rotated_df['y'] = rotated.iloc[:,1]
    return rotated_df

  # rotating the image with the transform matrix based on the better linear fit
  df_in_mask_rotated = rotate_df(df_in_mask, final_matrix)
  datasave['dataframe'] = df_in_mask_rotated

  fig, ax = plt.subplots(4,1,figsize=(15,6),dpi = 600)
  fig.suptitle(f"{full_name_prefix} Final rotation and found spots")
  masked_GFP = np.multiply(final_mask,GFP)
  plot_styles={'s': 60, 'alpha':1, 'linewidths':1, 'facecolors': 'none', 'edgecolors': 'black' }
  ax[0].imshow(transform_and_crop(segmented_image,final_tform), cmap=color_map)
  annotate_spots(df_in_mask_rotated, transform_and_crop(segmented_image, final_tform),ax=ax[1], plot_styles=plot_styles)
  annotate_spots(df_in_mask_rotated, transform_and_crop(masked_GFP, final_tform), ax=ax[2], plot_styles=plot_styles)

  datasave['rotated_cropped_segmented_image'] = transform_and_crop(segmented_image, final_tform)
  datasave['rotated_cropped_masked_GFP'] = transform_and_crop(masked_GFP, final_tform)

  # get the narrower plot range that imshow is using
  narrow_bbox = ax[2].get_position()
  xmin,xmax = narrow_bbox.intervalx 
  prev_xmin, prev_xmax = ax[2].get_xlim()
  ax[3].set_xlim(prev_xmin, prev_xmax)

  bbox = ax[3].get_position()
  ymin,ymax = bbox.intervaly

  ax[3].set_position( transforms.Bbox([[xmin,ymin],[xmax,ymax]]))
  ax[3].stem(df_in_mask_rotated['x'],df_in_mask_rotated['mass'])
  _,ytop = ax[3].get_ylim()
  ax[3].set_ylim(0,ytop*1.1)

  plt.savefig(plotname + '.png')
  plt.close()

  # save a file
  spots_detected_dataframe = df_in_mask_rotated
  spots_detected_dataframe['Worm'] = wormnumber
  spots_detected_dataframe['Rep'] = repnum
  spots_detected_dataframe['RNAi'] = RNAi
  spots_detected_dataframe['Genotype'] = genotype
  spots_detected_dataframe.to_csv(f"{datestr}_{genotype}_{RNAi}_{repnum}_segmented_plot.csv")
  datasave['spots_detected_dataframe'] = spots_detected_dataframe


  number_of_detected_cells = len(spots_detected_dataframe)
  number_of_detected_cells

  ### Histograms
  print("Calculating total intensities. Sum of intensity in all pixels inside of a cell mask.")

  # Total intensity values
  plotname = f"{full_name_prefix}_histograms"
  print("Plotting Total intensity values", plotname)
  fig, ax = plt.subplots(1,2, figsize=(10, 5))
  fig.suptitle(f"{full_name_prefix}")
  # Total intensity values
  ax[0].hist(df_in_mask['mass'], bins=15, color = "orangered", ec="orangered")
  ax[0].set(xlabel='total intensity', ylabel='count', title="histogram of total intensities")
  # Nuclei size
  ax[1].hist(df_in_mask['size'], bins=15, color = "orangered", ec="orangered")
  ax[1].set(xlabel='nuclei size', ylabel='count', title="histogram of nuclei sizes")
  #plt.show()
  plt.savefig(plotname + '.png')
  plt.close()

  save_data(data)


if __name__ == '__main__': main()