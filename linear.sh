#!/bin/bash

#SBATCH --time=150:00:00
#SBATCH --account=PAS1957
#SBATCH --gpus-per-node=1
#SBATCH --output=output/%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --ntasks=28

echo $SLURM_JOB_ID


module load miniconda3/4.10.3-py37
module load cuda/11.2.2

source activate
conda activate nlp
conda env list

set -x
mkdir output/$SLURM_JOB_ID
mkdir output/$SLURM_JOB_ID/code/
cp *.py output/$SLURM_JOB_ID/code/
cp *.sh output/$SLURM_JOB_ID/code/


python linear_regression.py --latent_v_dir=/users/PAS2062/delijingyic/project/org_wavegan/output/17524888/latent_v --s_dir=/users/PAS2062/delijingyic/project/org_wavegan/FriDNN/output/17524955/s_output/s_output.csv --job_id=$SLURM_JOB_ID --output_dir=output > "output/${SLURM_JOB_ID}/stdout.log"

mv output/$SLURM_JOB_ID.log output/$SLURM_JOB_ID/sbatch.log
