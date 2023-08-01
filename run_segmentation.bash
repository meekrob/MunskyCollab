#!/usr/bin/env bash
# Run my scriptification of Luis' segmentation.py (Google Colab notebook) on Izabella's data
# This produces a .csv in each directory data.
# If a *_segmentation.csv already exists, it prints out the path
MNTPNT=/Volumes/onishlab_shared/PROJECTS/32_David_Erin_Munskylab/Izabella_data/Keyence_data
echo "------- $(date) -------" 1>&2;

# find directories with a name like "JM259_L1_L4440_worm_9"
DIRS=$(find $MNTPNT/*/*/L1 -name '*_worm_*' -type d)
for f in $DIRS
do

    ls $f/*segmented.csv > /dev/null 2> /dev/null #&& echo 'got data' || echo 'no data yet'
    if [ $? -eq 1 ]
    then
        time ~/work/MunskyColab/segmentation_c_elegans_3d.py $f
    else
        echo $f/*segmented.csv
    fi
done

echo "------- $(date) -------" 1>&2;