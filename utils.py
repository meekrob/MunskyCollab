
import os, re
import pathlib                  # Library to work with file paths
from os import listdir
from os.path import isfile, join
import pickle 
from skimage.io import imread   # Module from skimage to read images as numpy arrays
import numpy as np
import matplotlib.pyplot as plt

def read_into_max_projections(path_dir):
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
  return max_GFP, max_Brightfield, image_ZYXC

def make_pickle_filename(key):
   genotype, rep, stage, RNAi, worm = key
   return f"{genotype}_{rep}_{stage}_{RNAi}_{worm}.pick"

def make_key(genotype, rep, stage, RNAi, worm):
   return genotype, rep, stage, RNAi, worm

def init_or_load_data(genotype, rep, stage, RNAi, worm):
  k = make_key(genotype, rep, stage, RNAi, worm)
  filename = make_pickle_filename(k)
  if os.path.exists(filename):
    return load_data(filename)
  else:
    return init_data(genotype, rep, stage, RNAi, worm)

def init_data(genotype, rep, stage, RNAi, worm):
  k = make_key(genotype, rep, stage, RNAi, worm)
  filename = make_pickle_filename(k)
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

def load_data(pathname):
  filename = os.path.basename(pathname)
  print(f"reading pickle from {filename}...", end=" ")
  with open(pathname, "rb") as inpickle:
    data = pickle.load(inpickle)
    print("done")
  key = list(data.keys())[0]
  return key, data[key], data

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

  x = list(df.loc[:,'x'])
  y = list(df.loc[:,'y'])
  markersizes = list(df.loc[:,'size'] * scatter_args['s'])

  # basically transferred this from trackpy.annotate, allowing for more control
  _imshow_style = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
  ax.imshow(GFP, **_imshow_style)
  ax.set_xlim(-0.5, GFP.shape[1] - 0.5)
  ax.set_ylim(-0.5, GFP.shape[0] - 0.5)
  ax.scatter(x, y, s = markersizes, edgecolors = scatter_args['edgecolors'], 
             linewidths = scatter_args['linewidths'], 
             alpha = scatter_args['alpha'],
             marker = scatter_args['marker'], facecolors = scatter_args['facecolors'],
             **plot_styles)
  
  bottom, top = ax.get_ylim()
  if top > bottom:
    ax.set_ylim(top, bottom, auto=None)

  return ax


