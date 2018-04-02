import os
import librosa


def save_samples(samples, output_dir, fs=16000):
    """
    Save output samples to disk
    """
    keys = sorted(samples.keys())
    for epoch in keys:
        epoch_samples = samples[epoch]
        for idx, samp in enumerate(epoch_samples):
            output_path = os.path.join(output_dir, str(epoch), "{}.wav".format(idx+1))
            samp = samp[0]
            librosa.output.write_wav(output_path, samp, fs)