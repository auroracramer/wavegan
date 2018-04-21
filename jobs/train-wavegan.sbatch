#!/usr/bin/env bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=rain-wavegan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=7-0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nl1115@nyu.edu
#SBATCH --output="train-wavegan-%j.out"
#SBATCH --err="train-wavegan-%j.err"

source ~/.bashrc
source activate aml


SRCDIR=/home/nl1115/advml/wavegan
AUDIO_DIR=/home/nl1115/gc
OUTPUT_DIR=/home/nl1115/advml/wvgan_out

module purge
module load cuda/8.0.44
module load cudnn/8.0v6.0
module load ffmpeg/intel/3.2.2

python $SRCDIR/train_wavegan.py \
    --model-size 64 \
    --phase-shuffle-shift-factor 2 \
    --post-proc-filt-len 512 \
    --lrelu-alpha 0.2 \
    --valid-ratio 0.1 \
    --test-ratio 0.1 \
    --batch-size 64 \
    --num-epochs 600 \
    --batches-per-epoch 10 \
    --ngpus 1 \
    --discriminator-updates 5 \
    --latent-dim 100 \
    --epochs-per-sample 1 \
    --sample-size 20 \
    --learning-rate 0.000001 \
    --beta-one 0.5 \
    --beta-two 0.9 \
    --regularization-factor 10.0 \
    --verbose \
    $AUDIO_DIR \
    $OUTPUT_DIR \
