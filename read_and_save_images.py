#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""segmentation_c_elegans_3d.py
Read in a set of images in brightfield and GFP.
Create a mask around the worm.
Identify nuclei that are flourescing.
Measure the intensity of nuclei individually.
"""

import os
import matplotlib.pyplot as plt
from skimage.io import imsave
from utils import *
from optparse import OptionParser
DEFAULT_INPUT_DIR = '/Users/david/work/MunskyColab/data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_3'
USAGE = "%prog [options] dirname\nRead max projections from pickle, or calculate them if necessary/requested.\n\nIF no dirname is specified, defaults to:\n" + DEFAULT_INPUT_DIR

def main():
  parser = OptionParser(usage=USAGE)
  parser.add_option("-r", "--recalculate", 
                    action="store_true", 
                    dest="recalculate",
                    help="if not stored in pickle file, re-read in all z-stacks and compute max projections")
  parser.add_option("-p", "--do-plots",
                    action="store_true",
                    dest="do_plots",
                    help="Save max projections for Brightfield and GFP in a multipanel plot.")
  parser.add_option("-s", "--save-imgs",
                    action="store_true",
                    dest="save_imgs",
                    help="Write PNGs for calculated max projections")
  
  (options, args) = parser.parse_args()
  if options.recalculate is None: options.recalculate = False
  print(options)
  print(len(args), args)
  

  if len(args) > 0:
    path_dir = args[0]
  else:
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201002_JM149_elt-2_Promoter_Rep_1/L4440_RNAi/L1/JM149_L1_L4440_worm_1'
    #path_dir = '/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data/201124_JM259_elt-2_Promoter_Rep_1/ELT-2_RNAi/L1/JM259_L1_ELT-2_worm_4'
    path_dir = DEFAULT_INPUT_DIR
    


  print("reading", path_dir)
  
  os.chdir(path_dir)
  current_dir = pathlib.Path().absolute()

# get worm info from path and filename
  try:
    longname,RNAi,stage,shortname = path_dir.split(os.path.sep)[-4:]
    # i.e.  201124_JM259_elt-2_Promoter_Rep_1, ELT-2_RNAi, L1, JM259_L1_ELT-2_worm_1
    datestr, genotype, labl, _, _, repnum = longname.split('_')
  except ValueError:
    print("Error parsing: `%s`" % longname)
    raise

  wormnumber = shortname.split('_')[-1]
  full_name_prefix = f"{genotype}_{RNAi}_{stage}_Rep{repnum}_Worm{wormnumber}"

  # check for pickle file, initialize data if not present
  datakey, datasave, data = init_or_load_data(genotype, repnum, stage, RNAi, wormnumber)

  if options.recalculate or 'max_GFP' not in datasave or 'max_Brightfield' not in datasave or 'image_ZYXC' not in datasave:
    max_GFP, max_Brightfield, image_ZYXC = read_into_max_projections(path_dir)
    datasave['max_GFP'] = max_GFP
    datasave['max_Brightfield'] = max_Brightfield
    datasave['image_ZYXC'] = image_ZYXC
  else:
    max_GFP = datasave['max_GFP']
    max_Brightfield = datasave['max_Brightfield']
    image_ZYXC = datasave['image_ZYXC']
  
  print('Range in GFP: min', np.min(max_GFP), 'max',np.max(max_GFP))
  print('Range in Brightfield: min', np.min(max_Brightfield), 'max', np.max(max_Brightfield))

  if options.do_plots:
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
  
  if options.save_imgs:
    imsave(f"{full_name_prefix}_max_GFP.png", max_GFP)
    imsave(f"{full_name_prefix}_max_Brightfield.png", max_Brightfield)

  save_data(data) # write pickle file

if __name__ == '__main__': main()