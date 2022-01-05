import os
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np


def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Save output samples to disk
    """
    sample_dir = os.path.join(output_dir, str(epoch))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for idx, samp in enumerate(epoch_samples):
        output_path = os.path.join(sample_dir, "{}.wav".format(idx + 1))
        samp = samp[0]
        samp = (samp - np.mean(samp)) / np.abs(samp).max()
        plt.figure()
        plt.plot(samp)
        plt.savefig(Path(sample_dir) / f"{idx+1}.png")
        plt.close()
        sf.write(output_path, samp, fs)
