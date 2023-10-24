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

import tifffile as tiff
import math
# # Loading libraries
import random                        # Library to generate random numbers

import numpy as np                   # Library for array manipulation
print("importing matplotlib...", file=sys.stderr, end="", flush=True)
import matplotlib.pyplot as plt      # Library used for plotting
from matplotlib import transforms
print("done", file=sys.stderr)

print("importing skimage...", file=sys.stderr, end="", flush=True)
import skimage                       # Library for image manipulation
from skimage.io import imread        # Module from skimage to read images as numpy arrays
from skimage import transform
from skimage.filters import gaussian # Module working with a gaussian filter
from skimage.measure import label, regionprops
from skimage.morphology import square, dilation
from skimage import measure
print("done", file=sys.stderr)

print("importing scipy...", file=sys.stderr, end="", flush=True)
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.spatial import distance
print("done", file=sys.stderr)

# for the cellpose step (calculating the mask), time the passes through the loop, save the data to skip the processing in
# future invocations
import time
import pickle

from utils import *
import json

from optparse import OptionParser
def PLOT_MASKED_IMAGES(full_name_prefix, segmented_image, masked_GFP, color_map):
  fig, ax  = plt.subplots(2,1)
  plotname = f"{full_name_prefix}_masked_images"
  fig.suptitle(f"{full_name_prefix} masked Brightfield/GFP")
  ax[0].imshow(segmented_image, cmap=color_map)
  ax[0].set(title='masked brightfield')
  ax[1].imshow(masked_GFP, cmap=color_map)
  ax[1].set(title='masked GFP')
  plt.savefig(plotname + '.png')
  plt.close()

def quickplot(*args, axis="on", do_columns = False, grid=False, filename = '', titles = [], save = True, **kwargs):
  ncols = len(args)
  nrows = 1
  if do_columns:
     nrows = ncols
     ncols = 1

  fig, ax = plt.subplots(nrows, ncols, **kwargs)
  if nrows == 1 and ncols == 1:
    ax = [ax]

  for i, img in enumerate(args):
    ax[i].imshow(img, cmap = 'Greys_r')
    # ax[i].axis(axis)
    # ax[i].grid(grid)
    if len(titles) > i:
      ax[i].set(title=titles[i])
  if save:      
    plt.savefig(filename)
    plt.close()
  else:
    return ax



def parse_provenance_from_path(path_dir):
  print("Parsing genotype, RNAi, stage, worm from path.")
  longname =  os.path.sep.join(path_dir.split(os.path.sep)[-4:])
  named_pat = "\d+_(?P<genotype>\S+)_elt-2_Promoter_Rep_(?P<repnum>\d)/(?P<RNAi>\S+)_RNAi/(?P<stage>\S+)/.+_worm_(?P<wormnum>\S+)$"
  genotype, repnum, RNAi, stage, wormnumber = re.match(named_pat, longname).groups()
  print(f"\t{genotype=}")
  print(f"\t{repnum=}")
  print(f"\t{RNAi=}")
  print(f"\t{stage=}")
  print(f"\t{wormnumber=}")
  return genotype, repnum, RNAi, stage, wormnumber

def make_fullname_prefix(genotype, repnum, RNAi, stage, wormnumber):
  return f"{genotype}_{RNAi}_{stage}_Rep{repnum}_Worm{wormnumber}"

def searchpathlist(pathlist, target):
  # search path list for target, return the first path that contains it or None
  for path in pathlist:
    if os.path.exists( path ) and os.path.exists( os.path.join(path, target ) ):
      return path

  return None

def main():
  USAGE = "[prog] image-set-path"

  # the script was called in...
  script_dir = os.getcwd()
  # defaults for options
  USE_CACHED_MASK = True
  FLIP_X = False
  SPOT_PARAMS = None #(388,19)
  parser = OptionParser(USAGE)
  parser.add_option("--ignore-flipx-setting", dest="ignore_flipx", 
                    default = False,
                    action="store_true",
                    help="if FLIP_X found in JSON, ignore it.")
  parser.add_option("-j", "--json-file", dest="json_path",
                    default="",
                    type="string",
                    help="manually provide the json settings file [default: based on parsed path name].")
  parser.add_option("--recalculate-mask", dest = "recalculate_mask",
                    default = False,
                    action="store_true",
                    help="Recalculate mask, even if it is stored in the pickle file.")
  parser.add_option("-p", "--pickle-file", dest="pickle_file",
                    type="string",
                    help="Specify location of pickle file [default: based on parsed path name].")
  parser.add_option("-x", "--get-exp-from-json", dest="get_exp_from_json",
                    default=False,
                    action="store_true",
                    help="Get info about the experiment (genotype,RNAi,Worm,Rep) from the json file. Requires, -j, --json-file)")
  
  (options, args) = parser.parse_args()

  if len(args) > 0:
    path_dir = args[0]
    print("argument provided:", path_dir)

  else:
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_1'
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201124_JM259_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM259_L1_ELT-2_worm_4'
    #path_dir = '/Users/david/work/MunskyColab/data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_9'
    path_dir = '/Users/david/work/MunskyColab/data/201002_JM149_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM149_L1_ELT-2_worm_1'
    print("no argument provided.\nDefaulting to", path_dir)

  if not os.path.exists(path_dir):
    print("input path not found:", path_dir, file=sys.stderr)
    sys.exit(1)

  # read manual settings from a file in the same directory as the images,
  # if they exist
  json_search_path = [path_dir, '.', script_dir]

  if options.json_path:
    json_filename = options.json_path
    json_file_found = searchpathlist(json_search_path, json_filename)
    if json_file_found is not None: # file already exists
      json_filename = pathlib.Path( os.path.join(json_file_found, json_filename)).absolute()
      print(f"Using {json_filename} to read and update settings.")

    else:  # file not found, make sure specified containing directories exist
      pathlib.Path(os.path.dirname(json_filename)).mkdir(parents=True, exist_ok=True)
      print(f"creating {json_filename} to store settings.")

  else:
      # get worm info from path and filename
    try:
      # print("Parsing genotype, RNAi, stage, worm from path.")
      # longname =  os.path.sep.join(path_dir.split(os.path.sep)[-4:])
      # named_pat = "\d+_(?P<genotype>\S+)_elt-2_Promoter_Rep_(?P<repnum>\d)/(?P<RNAi>\S+)_RNAi/(?P<stage>\S+)/.+_worm_(?P<wormnum>\S+)$"
      # genotype, repnum, RNAi, stage, wormnumber = re.match(named_pat, longname).groups()
      genotype, repnum, RNAi, stage, wormnumber = parse_provenance_from_path(path_dir)
      # print(f"\t{genotype=}")
      # print(f"\t{repnum=}")
      # print(f"\t{RNAi=}")
      # print(f"\t{stage=}")
      # print(f"\t{wormnumber=}")
      
    except:
      print("Error parsing: `%s`" % longname, file=sys.stderr)
      print(f"Expected last 4 directories of the path to follow the following pattern: \d+_(\\S+)_elt-2_Promoter_Rep_\\S+", file=sys.stderr)
      print("Example: 201124_JM259_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM259_L1_ELT-2_worm_1", file=sys.stderr)
      raise

    full_name_prefix = make_fullname_prefix(genotype, repnum, RNAi, stage, wormnumber)
    print(f"{full_name_prefix=}")
    json_filename = f"{full_name_prefix}_manual_params.json"

  # need to make sure this is still valid after we change directories
  json_filename = pathlib.Path(json_filename).absolute()

  # After resolving the path of the json filename (if specified)
  # or using the default, if not specified
  # read it in if it exists.
  user_params = None
  if os.path.exists(json_filename):
    print(f"Reading settings from json file: {os.path.basename(json_filename)}")
    with open(json_filename, "r") as json_in:
      user_params = json.load(json_in)
  else:
    print(f"Data and settings will be saved in {os.path.basename(json_filename)}")

  if options.get_exp_from_json and not options.json_path:
      print("option -x,--get-exp-from-json requires -j,--json-file to be specified.", file=sys.stderr)
      sys.exit(1)
  elif options.get_exp_from_json and options.json_path:
    if user_params is not None: 
        try:
          genotype =    user_params['experiment']['genotype']
          stage =       user_params['experiment']['stage']
          repnum =      user_params['experiment']['repnum']
          wormnumber =  user_params['experiment']['wormnum']
          RNAi =        user_params['experiment']['RNAi'] 
        except:
          print("JSON file requires genotype, RNAi, etc., with the -x,--get-exp-from-json but not all are specified.", file=sys.stderr)
          sys.exit(1)
    else:
      print(f"User argument -x,--get-exp-from-json requires data stored in a json file {options.json_path=} but it doesn't exist.")
      sys.exit(1)
  elif options.json_path and not options.get_exp_from_json:
      genotype, repnum, RNAi, stage, wormnumber = parse_provenance_from_path(path_dir)
      full_name_prefix = make_fullname_prefix(genotype, repnum, RNAi, stage, wormnumber)
      
  
  if options.recalculate_mask:
    USE_CACHED_MASK = False

  # change to the user-supplied directory now for the rest of input/output
  print("Parsing", path_dir)
  os.chdir(path_dir)
  current_dir = pathlib.Path().absolute()


  k, datasave, data = init_data(genotype, repnum, stage, RNAi, wormnumber)

  
  if user_params is not None:
    if 'SPOT_PARAMS' in user_params:
      SPOT_PARAMS = user_params['SPOT_PARAMS']['minmass'], user_params['SPOT_PARAMS']['particle_size']
    if 'FLIP_X' in user_params:
      FLIP_X = user_params['FLIP_X']

  if 'max_GFP' not in datasave or 'max_Brightfield' not in datasave or 'image_ZYXC' not in datasave:
    print("Reading and processing images")
    max_GFP, max_Brightfield, image_ZYXC = read_into_max_projections(current_dir)
    datasave['max_GFP'] = max_GFP
    datasave['max_Brightfield'] = max_Brightfield
    datasave['image_ZYXC'] = image_ZYXC
  else:
    max_GFP = datasave['max_GFP']
    max_Brightfield = datasave['max_Brightfield']
    image_ZYXC = datasave['image_ZYXC']

  print("Pixel value ranges:")
  print('\tRange in GFP: min', np.min(max_GFP), 'max',np.max(max_GFP))
  print('\tRange in Brightfield: min', np.min(max_Brightfield), 'max',np.max(max_Brightfield))

  datasave['max_GFP'] = max_GFP
  datasave['max_Brightfield'] = max_Brightfield

  #@title Plotting max projections
  plotname = f"{full_name_prefix}_max_projections"
  color_map = 'Greys_r'
  fig, ax = plt.subplots(1,2, figsize=(10, 3))
  fig.suptitle(f"{full_name_prefix} max. projections")
  # Plotting the heatmap of a section in the image - MISPLACED LABEL? - DK
  # print("Plotting the heatmap of a section in the image")
  ax[0].imshow(max_Brightfield,cmap=color_map)
  ax[1].imshow(max_GFP,cmap=color_map)
  ax[0].set(title='max_Brightfield'); ax[0].axis('on');ax[0].grid(False)
  ax[1].set(title='max_GFP'); ax[1].axis('on');ax[1].grid(False)
  print("Plotting max projections:", plotname + '.png')
  plt.savefig(plotname + '.png')
  plt.close()



  #@title Plotting all z-slices
  plotname = f"{full_name_prefix}_z-slices"
  number_z_slices = image_ZYXC.shape[0]
  fig, ax = plt.subplots(2,number_z_slices, figsize=(25, 5))
  fig.suptitle(f"{full_name_prefix} z slices")
  color_map = 'Greys_r'
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

  print(f"Plotting z-slices: {plotname}.png")
  plt.savefig(plotname + '.png')
  plt.close()

  color_map = 'Greys_r'

  total_time = 0
  pickled_mask_path = os.path.join(current_dir, f"{full_name_prefix}_mask.pickle")

  if USE_CACHED_MASK and os.path.exists( pickled_mask_path ):
    print("Reading cached mask from pickle...", end=" ")

    with open(pickled_mask_path, "rb") as inpickle:
      masks_total = pickle.load(inpickle)

    print("done")

  else:
    print("Using cellpose to segment image")
    print("importing cellpose...", flush=True, end="")
    from cellpose import models
    #from cellpose import plot
    print("done", flush=True)
    model = models.Cellpose(gpu=True, model_type='cyto2') # model_type='cyto', 'cyto2' or model_type='nuclei'
    list_ranges =np.linspace(20, 300, 10, dtype='int')  #[50,100,150,200,250]
    list_masks = []
    masks_total = np.zeros_like(max_Brightfield)
    for i,diameter in enumerate (list_ranges):
      begin_time = time.time()
      print("\tmodel.eval(max_Brightfield ...) %d/%d " % (i+1,len(list_ranges)), flush=True, end='')

      masks = model.eval(max_Brightfield, diameter=diameter, flow_threshold=1, channels=[0,0], net_avg=True, augment=True)[0]
      masks_total = masks_total+ masks

      end_time = time.time()
      total_time += end_time - begin_time
      print(round(end_time - begin_time), "seconds.", round(total_time), "total.")
    with open(pickled_mask_path,"wb") as outpickle:
      print("writing new mask to", pickled_mask_path, "...", end="")
      pickle.dump(masks_total, outpickle)
      print("done")

    print("Total time: %d seconds" % round(total_time))

  datasave['masks_total'] = masks_total

  # Binarization
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

  if True: #final_mask.max() == 0:
    #raise "everything masked"
    #final_mask = None
    segmented_image = max_Brightfield.copy()
    masked_GFP = max_GFP.copy()
  else:
    # Mask by image
    segmented_image = np.multiply(final_mask,max_Brightfield)
    masked_GFP = np.multiply(final_mask,max_GFP)
  datasave['final_mask'] = final_mask

  datasave['segmented_image'] = segmented_image
  datasave['masked_GFP'] = masked_GFP

  quickplot(segmented_image, max_GFP, do_columns = True, titles = ['masked brightfield','masked GFP'], filename="firehose.png")
  #PLOT_MASKED_IMAGES(full_name_prefix, segmented_image, masked_GFP, color_map)
  fig, ax  = plt.subplots(2,1)
  plotname = f"{full_name_prefix}_masked_images"
  fig.suptitle(f"{full_name_prefix} masked Brightfield/GFP")
  ax[0].imshow(segmented_image, cmap=color_map)
  ax[0].set(title='masked brightfield')
  ax[1].imshow(masked_GFP, cmap=color_map)
  ax[1].set(title='masked GFP')
  plt.savefig(plotname + '.png')
  plt.close()

  
  plot_center = np.array(segmented_image.T.shape)[:2]/2
  shift_to_plot_center = transform.EuclideanTransform(translation=-plot_center)
  cm = center_of_mass(segmented_image.T)
  pc = np.array(segmented_image.T.shape)[:2]/2

  """## Linear regression"""
  DO_REGRESSION_ON_MASK = False
  final_matrix = np.identity(len(segmented_image.T.shape)+1)
  final_tform  = transform.EuclideanTransform(final_matrix)
  if DO_REGRESSION_ON_MASK:
    minmax = lambda x: (min(x), max(x))
    x,y = segmented_image.T.nonzero() # if the whole image is masked due to background, this will be empty
    # horizontal line through worm
    A = np.vstack([x, np.ones(len(x))]).T
    horiz_fit = np.linalg.lstsq(A, y, rcond=None)
    horiz_slope, horiz_intercept = horiz_fit[0]

    # vertical line through worm
    A = np.vstack([y, np.ones(len(y))]).T
    vertical_fit = np.linalg.lstsq(A, x, rcond=None)
    vertical_slope, vertical_intercept = vertical_fit[0]

    horiz_line = lambda a: a*horiz_slope + horiz_intercept
    vertical_line = lambda a: a*vertical_slope + vertical_intercept

    horiz_line_inv = lambda a: (a-horiz_intercept)/horiz_slope
    vertical_line_inv = lambda a: (a-vertical_intercept)/vertical_slope

    m,b = vertical_fit[0]
    inverse_m = 1 / m # why isn't it -1 / m?
    horiz_angle_correction = math.atan(horiz_slope)
    inverse_vertical_angle_correction = math.atan(inverse_m)

    img_height, img_width = segmented_image.shape

    xmin = 0
    xmax = img_width
    ymin = 0
    ymax = img_height

    # horiz
    horiz_xbounds, horiz_ybounds = isect_line_box(horiz_fit[0], 0, 0, img_width, img_height)
    horiz_xseries = np.linspace(horiz_xbounds[0], horiz_xbounds[1],3)
    horiz_yseries = np.linspace(horiz_ybounds[0], horiz_ybounds[1],3)

    # vert
    vertical_ybounds, vertical_xbounds = isect_line_box(vertical_fit[0], 0, 0, img_height, img_width)
    vertical_xseries = np.linspace(vertical_xbounds[0], vertical_xbounds[1],3)
    vertical_yseries = np.linspace(vertical_ybounds[0], vertical_ybounds[1],3)

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

    # shift the origin back to the plot center
    shift_to_plot_center = transform.EuclideanTransform(translation= [plot_center[0],plot_center[1]])

    ## FLIP or nah??????
    flip = np.identity(3)
    if FLIP_X: flip[0,0] = -1

    # horizontal angle
    angleH = math.atan(horiz_slope)
    rotation = transform.EuclideanTransform(rotation=angleH)
    matrix_h = cf_horiz_shift.params @ rotation.params @ flip @ np.linalg.inv(shift_to_plot_center.params)
    tform_h = transform.EuclideanTransform(matrix_h)
    tf_img_h = transform.warp(segmented_image, tform_h)

    # vertical
    angleV = inverse_vertical_angle_correction

    rotation = transform.EuclideanTransform(rotation=angleV)
    matrix_v =  shift_to_plot_center.params @ flip @ np.linalg.inv(rotation.params)  @ cf_vertical_shift.params
    tform_v = transform.EuclideanTransform(matrix_v)
    tf_img_v = transform.warp(segmented_image, tform_v.inverse)

    ax[1,0].imshow(tf_img_h, cmap=color_map)
    ax[1,0].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
    ax[1,0].set(title='rotate %d degrees' % math.degrees(angleH))
    ax[1,1].imshow(tf_img_v, cmap=color_map)
    ax[1,1].scatter(pc[0], pc[1],  s=10, edgecolors='yellow', c='yellow')
    ax[1,1].set(title='rotate %d degrees' % math.degrees(angleV))

    # choose the transform coming from the better fit
    if horiz_fit[1] < vertical_fit[1]:
      matrix = matrix_h
      tform = tform_h
    else:
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
  spots_detected_dataframe_all = tp.locate(GFP, diameter=particle_size, minmass=0)

  # This section generates a histogram with the intensity of the detected particles in the image.
  fig, ax = plt.subplots(1,1, figsize=(4, 4))
  plotname = f"{full_name_prefix}_histogram_mass"
  print(f"\tPlotting {plotname}")
  fig.suptitle(f"{full_name_prefix}\nhistogram of intensities")
  ax.hist(spots_detected_dataframe_all['mass'], bins=50, color = "orangered", ec="orangered")
  ax.set(xlabel='mass', ylabel='count')
  plt.savefig(plotname + '.png')
  plt.close()


  # Plot GFP and the tp.annotate graph together
  plotname = f"{full_name_prefix}_comparison"
  print("\tPlotting GFP and segmentation together", plotname)
  color_map = 'Greys_r'

  if SPOT_PARAMS is None:
    print("Running optimization to find spot params:")
    # Optimization from Luis!
    # Creating vectors to test all conditions for nuclei detection.
    number_optimization_steps = 10
    particle_size_vector = [num for num in range(15, 27 + 1) if num % 2 != 0][:number_optimization_steps]
    print('\tparticle_size_vector: ', particle_size_vector)
    minmass_vector = np.linspace(250, 500, num=number_optimization_steps, endpoint=True,dtype=int)
    print('\tminmass_vector: ', minmass_vector)

    fig, ax = plt.subplots(len(minmass_vector),len(particle_size_vector)+1, 
                          figsize=(3*len(particle_size_vector),
                                   1*len(minmass_vector)), 
                                    dpi=1000)
    fig.suptitle(f"{full_name_prefix} parameter comparison")

    # Left column is a replot of previous steps, subsequent columns are 
    # spot finding at different params
    ax[0,0].imshow(transform_and_crop(segmented_image,final_tform),cmap=color_map)
    ax[1,0].imshow(transform_and_crop(final_mask,final_tform),cmap=color_map)
    ax[2,0].imshow(transform_and_crop(segmented_image, final_tform),cmap=color_map)
    ax[3,0].imshow(transform_and_crop(masked_GFP, final_tform),cmap=color_map)
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
    import warnings
    for i, mm in enumerate(minmass_vector):
      for j, particle_size in enumerate(particle_size_vector):
        outmsg = f"\ti: {i}, particle_size: {particle_size}; j: {j}, minmass: {mm}"
        print(outmsg, "\b" * len(outmsg), sep='', end='',flush=True)
        with warnings.catch_warnings(): # suppress warning for no spots found
          warnings.simplefilter("ignore")
          spots_detected_dataframe = tp.locate(GFP,diameter=particle_size, minmass=mm) 
        # Selecting only spots located inside mask
        df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)
        if len(df_in_mask) > 0:
          metric[i,j] = np.sum(df_in_mask['mass']) / len(df_in_mask) # maximizes the mean intensity in all spots
        else:
          metric[i,j] = 0
      
        annotate_spots(rotate_df(df_in_mask,final_tform), transform_and_crop(masked_GFP, final_tform), ax[i,j+1],
                       plot_styles={'s': 6, 'alpha':1, 'linewidths':.6, 'facecolors': 'none', 'edgecolors': 'r' })
        ax[i,j+1].axis('off')
        ax[i,j+1].grid(False)
        ax[i,j+1].set(title=f'{particle_size};{mm}; {len(df_in_mask)} spots')

        del spots_detected_dataframe, df_in_mask

    metric = metric.astype(int)

    # selecting indices that maximize metric
    selected_minmass_index, selected_particle_size_index = np.unravel_index(metric.argmax(), metric.shape)
    selected_minmass = minmass_vector[selected_minmass_index]
    selected_particle_size = particle_size_vector[selected_particle_size_index]
    spots_detected_dataframe = tp.locate(GFP,diameter=selected_particle_size, minmass=selected_minmass) 
    df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)
    print(f'Selected params: {selected_particle_size=}, {selected_minmass=}. {len(df_in_mask)} spots')
    # plot the "best" one underneath the brightfield/mask/GFP
    ax[4,0].set(title=f'Selected params: {selected_particle_size}, {selected_minmass}. {len(df_in_mask)} spots')
    # add plot with different parameters to grid    
    annotate_spots(df_in_mask, transform_and_crop(masked_GFP, final_tform), ax[4,0])
    # write the comparison figure
    plt.savefig(plotname + '.png')
    plt.close()
  else:
    selected_minmass, selected_particle_size = SPOT_PARAMS
    spots_detected_dataframe = tp.locate(GFP,diameter=selected_particle_size, minmass=selected_minmass) 
    df_in_mask = spots_in_mask(df=spots_detected_dataframe,masks=final_mask)

  datasave['selected_spots_params'] = {'selected_minmass':selected_minmass,
                                       'selected_particle_size':selected_particle_size}

  # rotating the image with the transform matrix based on the vertical fit
  df_mx = df_in_mask.loc[:,('x','y')]
  df_mx['1'] = 1

  DO_REGRESSION_ON_SPOTS = True
  if DO_REGRESSION_ON_SPOTS:
    spot_regression = numpy_regression(df_mx.loc[:,('x')], df_mx.loc[:,('y')])
    spot_regression_v = numpy_regression(df_mx.loc[:,('y')],  df_mx.loc[:,('x')])
    print(spot_regression)
    spot_m,spot_b = spot_regression[0]
    spot_m_v,spot_b_v = spot_regression_v[0]
    spot_SS = spot_regression[1]
    spot_SS_v = spot_regression_v[1]
    spot_inverse_m = 1 / spot_m_v # why isn't it -1 / m?
    spot_horiz_angle_correction = math.atan(spot_m)
    spot_inverse_vertical_angle_correction = math.atan(spot_inverse_m)
    selected_rotation = spot_horiz_angle_correction
    if spot_SS_v < spot_SS: selected_rotation = spot_inverse_vertical_angle_correction
    midpoint = lambda x: x.min() + ((x.max()-x.min())/2)
    cf_spot = (midpoint(df_mx['x']),midpoint(df_mx['y']))
    shift_to_cf_spot = transform.EuclideanTransform(translation= [cf_spot[0],cf_spot[1]])

    ## FLIP or nah??????
    flip = np.identity(3)
    if FLIP_X: flip[0,0] = -1

    rotation_et = transform.EuclideanTransform(rotation=selected_rotation)
    matrix_spot = shift_to_cf_spot.params @ rotation_et.params @ flip @ np.linalg.inv(shift_to_cf_spot.params)
    tform_spot = transform.EuclideanTransform(matrix_spot)
    tf_img_spot = transform.warp(segmented_image, tform_spot) 
    
    # used downstream
    final_matrix = matrix_spot
    final_tform = tform_spot
    

  rotated = (np.linalg.inv(final_matrix) @ df_mx.T).T
  df_in_mask_rotated = df_in_mask.copy()
  df_in_mask_rotated['x'] = rotated.iloc[:,0]
  df_in_mask_rotated['y'] = rotated.iloc[:,1]
  ymin = df_in_mask_rotated.loc[:,"y"].min()
  ymax = df_in_mask_rotated.loc[:,"y"].max()

  for amount in np.arange(6,0,-.1):
    new_ymin,new_ymax = widen(ymin,ymax,amount)
    if new_ymin >= 0 and new_ymax <= segmented_image.shape[0]:
      ymin = new_ymin
      ymax = new_ymax
      print("widened range by", amount)
      break


  segmented_image = transform_cropstrip_untransform(segmented_image, final_tform, (round(ymin), round(ymax)))
  masked_GFP = transform_cropstrip_untransform(masked_GFP, final_tform, (round(ymin), round(ymax)))
  
  # write the selected spot find
  plotname = f"{full_name_prefix}_spots_detected"
  print("\tPlotting", plotname)

  fig, ax = plt.subplots(1,5, figsize=(15, 5),dpi=1000)
  fig.suptitle(f"{full_name_prefix} spots detected")

  # rotating the image with the transform matrix based on the better linear fit
  #df_in_mask_rotated = rotate_df(df_in_mask, final_matrix)
  datasave['dataframe'] = df_in_mask_rotated

  fig, ax = plt.subplots(5,1,figsize=(15,6),dpi = 600)
  fig.suptitle(f"{full_name_prefix} Final rotation and found spots")
  
  plot_styles={'s': 60, 'alpha':1, 'linewidths':1, 'facecolors': 'none', 'edgecolors': 'black' }
  ax[0].imshow(transform_and_crop(segmented_image,final_tform), cmap=color_map)
  annotate_spots(df_in_mask_rotated, transform_and_crop(segmented_image, final_tform),ax=ax[1], plot_styles=plot_styles)
  plot_styles={'s': 60, 'alpha':1, 'linewidths':.75, 'facecolors': 'none', 'edgecolors': 'green' }
  annotate_spots(df_in_mask_rotated, transform_and_crop(masked_GFP, final_tform), ax=ax[2], plot_styles=plot_styles)

  df_in_mask_rotated['y'] = df_in_mask_rotated['y'] - ymin
  plot_styles={'s': 60, 'alpha':1, 'linewidths':1, 'facecolors': 'none', 'edgecolors': 'black' }
  annotate_spots(df_in_mask_rotated, transform_and_crop(segmented_image, final_tform),ax=ax[3], plot_styles=plot_styles)
  plot_styles={'s': 60, 'alpha':1, 'linewidths':.75, 'facecolors': 'none', 'edgecolors': 'green' }
  annotate_spots(df_in_mask_rotated, transform_and_crop(masked_GFP, final_tform), ax=ax[4], plot_styles=plot_styles)

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

  # save dataframe as csv
  spots_detected_dataframe = df_in_mask_rotated
  spots_detected_dataframe['Worm'] = wormnumber
  spots_detected_dataframe['Rep'] = repnum
  spots_detected_dataframe['RNAi'] = RNAi
  spots_detected_dataframe['Genotype'] = genotype
  spots_detected_dataframe.to_csv(f"{full_name_prefix}_segmented_plot.csv")
  datasave['spots_detected_dataframe'] = spots_detected_dataframe

  number_of_detected_cells = len(spots_detected_dataframe)
  number_of_detected_cells

  ### Histograms
  print("Histograms: Calculating total intensities. Sum of intensity in all pixels inside of a cell mask.")

  # Total intensity values
  plotname = f"{full_name_prefix}_histograms"
  print("\tPlotting Total intensity values", plotname)
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

  params = {}
  params['SPOT_PARAMS'] = {'minmass': int(selected_minmass), 'particle_size': int(selected_particle_size) }
  params['FLIP_X'] = FLIP_X
  params['experiment'] = {'genotype': genotype, 'stage': stage, 'repnum': repnum, 'wormnum': wormnumber, 'RNAi': RNAi}
  
  print(f"Saving data to:\n\t{json_filename}")
  with open(json_filename, "w") as json_out:
    json.dump(params, json_out, indent = 4)

  # end main()

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
  # integrate params into plot_styles obj
  plot_styles = plot_styles.copy()
  for k in scatter_args.keys():
    if k in plot_styles:
      scatter_args[k] = plot_styles[k]
      del plot_styles[k]
  # get coordinates
  x = list(df.loc[:,'x'])
  y = list(df.loc[:,'y'])
  markersizes = list(df.loc[:,'size'] * 1.5)
  # from trackpy.annotate...
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

def rotate_df(df, final_matrix):
    # prepare transform matrix
    df_mx = df.loc[:,('x','y')]
    df_mx['1'] = 1
    # perform transform
    rotated = (np.linalg.inv(final_matrix) @ df_mx.T).T
    # store result and retain original coordinates
    rotated_df = df.copy()
    rotated_df['orig_x'] = df['x']
    rotated_df['orig_y'] = df['y']
    rotated_df['x'] = rotated.iloc[:,0]
    rotated_df['y'] = rotated.iloc[:,1]
    return rotated_df

# from Luis: select only the spots in the mask using their coordinates (df)
def spots_in_mask(df,masks):
    if masks.max() == 0:
      return df
    # extracting the contours in the image
    coords = np.array([df.y, df.x]).T # These are the points detected by trackpy
    coords_int = np.round(coords).astype(int)  # or np.floor, depends
    values_at_coords = masks[tuple(coords_int.T)] # If 1 the value is in the mask
    df['In Mask']=values_at_coords # Check if pts are on/in polygon mask
    condition = df['In Mask'] == 1
    selected_rows = df[condition]
    
    return selected_rows.drop(columns=['In Mask'])
def make_strip_mask(image, tform, strip):
  tformed = transform.warp(image, tform)
  mask = np.ones_like(tformed)

def widen(lower, upper, amount):
  span = (upper - lower) * amount
  
  new_lower = math.floor((upper+lower)/2 - span/2)
  new_upper = math.ceil(lower + span)
  return (new_lower, new_upper)

def transform_cropstrip_untransform(image, tform, strip):
  tformed = transform.warp(image, tform)
  cropmin,cropmax = strip
  tformed[:cropmin,] = 0
  tformed[cropmax:,:] = 0
  
  inverse = transform.EuclideanTransform(np.linalg.inv(tform.params))
  untformed = transform.warp(tformed, inverse)
  return untformed
  
def transform_and_crop(image, tform):
  # auto-crop to the zeroed out coordinates (from mask) if they exist
  tformed = transform.warp(image, tform)
  x,y = tformed.nonzero()

  if len(x) == 0 or len(y) == 0:
    return tformed
  
  #shift_to_top = transform.EuclideanTransform(translation=(0,cropmin))
  #tformed = transform.warp(tformed, shift_to_top)
  return tformed[min(x):max(x),min(y):max(y)]

def numpy_regression(x,y, rcond=None):
  A = np.vstack([x, np.ones(len(x))]).T
  fit = np.linalg.lstsq(A, y, rcond=rcond)
  return fit

def check_pathdir(filepath, give_err_msg = False):
  # assume filepath specifies a path to a file that may not exist, 
  # but its containg directories must
  if os.path.exists(filepath):
    return True
  else:
    checkpath = os.path.dirname(filepath)
    if os.path.exists(checkpath):
      return True
    elif give_err_msg:
      fname = os.path.basename(filepath)
      print(f"Containing directory for specified file {fname} not found", file=sys.stderr)
  
  return False





if __name__ == '__main__': main()


