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
genotype=$1 # one of JM149/JM259
Rep=$2 # Rep_1, Rep_2, Rep_3
RNAi=$3 # L4440 or ELT-2
Worm=$SLURM_ARRAY_TASK_ID

#sbatch --array=1-3,5-7 array_run_seg.sbatch JM259 Rep_3 ELT-2
#sbatch --array=1-10 array_run_seg.sbatch JM259 Rep_1 L4440

# to run a non-numeric worm number (such as 7-8), run individually (without --array) and give 7-8 as the 4th argument
if [ -z "$Worm" ]
then
    Worm=$4
fi

fname="segmentation/${genotype}_elt-2_Promoter_${Rep}/${RNAi}_RNAi/L1/${genotype}_L1_${RNAi}_worm_${Worm}"
echo "Parsed SLURM_ARRAY_TASK_ID into $fname"

if [ ! -e "$fname" ]
then
    echo "$fname does not exist" >&2
    exit 2
fi

cmd="python segmentation_c_elegans_3d.py $(readlink -f $fname)"
echo $cmd
time eval $cmd
