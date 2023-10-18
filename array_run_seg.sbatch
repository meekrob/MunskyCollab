#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0:20:00
#SBATCH --qos=normal
#SBATCH --partition=amilan
#SBATCH --job-name=spots
#SBATCH --output=%x.%A_%a.log # like: spots.ID_1.log
# ------------------------------------------------------
echo "[$0] $SLURM_JOB_NAME $@" # log the command line
date # timestamp
# ------------------------------------------------------
# set temporary directories
export TMPDIR=$SLURM_SCRATCH
export TMP=$TMPDIR
# ------------------------------------------------------
# conda environment
module load anaconda/2022.10
conda activate segmentation
# ------------------------------------------------------
# check argument assigned through sbatch --array
which_line=$SLURM_ARRAY_TASK_ID
if [ -z "$which_line" ]
then
    echo "no array index argument given" >&2
    exit 1
fi
# ------------------------------------------------------
# loop through input until reaching the entry specified by the array index
i=0
while read fname
do
    if [ $i -eq $which_line ]
    then
        # --------------------------------------------------
        # run script
        cmd="python segmentation_c_elegans_3d.py  $fname"
        echo $cmd
        time eval $cmd
        break # only process the requested line
    fi
    i=$((i+1))
done < <(cat *dirlist.txt | grep -v '^$' )
# end of script  ------------------------------------------------------
