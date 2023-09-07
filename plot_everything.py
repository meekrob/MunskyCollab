#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
from utils import *

SELECTED_GENOTYPE = "JM259"
SELECTED_REP = "2"
SELECTED_RNAi = "ELT-2_RNAi"

with open("final_plot.pick", "rb") as inpickle:
  bigpick = pickle.load(inpickle)

to_plot = []
for k,v in bigpick.items():
  genotype, rep, stage, RNAi, worm = k
  if rep == SELECTED_REP and genotype == SELECTED_GENOTYPE and RNAi == SELECTED_RNAi:
    to_plot.append((k, v))

try:
  to_plot.sort(key=lambda k: int(k[0][4]))
except:
  to_plot.sort(key=lambda k: k[0][4])

fig, ax = plt.subplots(len(to_plot),4,figsize=(13,9),dpi = 600)
genotype, rep, stage, RNAi, worm = to_plot[0][0]
fig.suptitle(f"{genotype} {rep} {RNAi}")
brightfield_plot_styles={'s': 1.5, 'alpha':1, 'linewidths':.5, 'facecolors': 'g', 'edgecolors': 'black' }
gfp_plot_styles={'s': 1.5, 'alpha':.5, 'linewidths':.5, 'facecolors': 'none', 'edgecolors': 'r' }
color_map = 'Greys_r'

for i,worm in enumerate(to_plot):
  k,v = worm
  genotype, rep, stage, RNAi, worm = k
  df_in_mask_rotated = v['dataframe'] 
  rotated_cropped_segmented_image = v['rotated_cropped_segmented_image'] 
  rotated_cropped_masked_GFP = v['rotated_cropped_masked_GFP']
  dataframe = v['dataframe']

  for j in range(4):
    
    ax[i,j].axis('off') 
    ax[i,j].grid(False)

  ax[i,0].imshow(rotated_cropped_segmented_image, cmap=color_map)
  ax[i,0].set(title= f"{genotype}_{RNAi}_{stage}_Rep{rep}_Worm{worm}")
  annotate_spots(df_in_mask_rotated, rotated_cropped_segmented_image,ax=ax[i,1], plot_styles=brightfield_plot_styles)
  ax[i,2].imshow(rotated_cropped_masked_GFP, cmap=color_map)
  annotate_spots(df_in_mask_rotated, rotated_cropped_masked_GFP, ax=ax[i,3], plot_styles=gfp_plot_styles)

plt.savefig(f'{SELECTED_GENOTYPE}_{SELECTED_RNAi}_Rep{SELECTED_REP}.png')
plt.close()

