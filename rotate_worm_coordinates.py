#!/usr/bin/env python3
import sys,math
import pandas as pd
import numpy as np
from math import cos, sin

def grouped_eqn(df):
    x_max = max(df.loc[:,'x'])
    x_min = min(df.loc[:,'x'])
    x_range = (x_min,x_max)
    x_mag = abs(x_min-x_max)
   #print(x_range, x_mag)
    y_max = max(df.loc[:,'y'])
    y_min = min(df.loc[:,'y'])
    y_range = (y_min,y_max)
    y_mag = abs(y_min-y_max)
    
    #print(y_range, y_mag)
   
    # shift everything to origin
    df.x = df.x - x_min
    df.y = df.y - y_min

    # figure out how far to rotate it down to the x axis
    slope = y_max / x_max
    #print("slope is", slope)
    worm_angle = math.atan(slope)
    #print("theta is", worm_angle)

    # use a transformation matrix to convert to rotated coordinates
    tform = np.array([[cos(worm_angle),
                      sin(worm_angle)],
                      [-sin(worm_angle),
                      cos(worm_angle)]])
    #print(tform)
    df_xy = df.loc[:,['x','y'] ]
    xy = df_xy.to_numpy().transpose()
    #print("orig:")
    #print(xy)
    xy_rot = np.matmul(tform, xy)
    #print("rotated:")
    #print(xy_rot)
    df["x_rot"] = xy_rot[0,]
    df["y_rot"] = xy_rot[1,]
    return df.loc[:,['x_rot','y_rot']]

x = pd.read_csv("all_segmentation_results.csv")

rotated = x.groupby(['Worm','Rep','RNAi','Genotype']).apply(grouped_eqn)
rotated.index = x.index
x2=pd.concat([x, rotated.loc[:,["x_rot","y_rot"]]], axis=1)
x2.to_csv("rotated_all_segmentation_results.csv")