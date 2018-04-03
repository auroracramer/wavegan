import os
import librosa


def save_samples(epoch_samples, epoch, output_dir, fs=16000):
    """
    Save output samples to disk
    """
    for idx, samp in enumerate(epoch_samples):
        output_path = os.path.join(output_dir, str(epoch), "{}.wav".format(idx+1))
        samp = samp[0]
        librosa.output.write_wav(output_path, samp, fs)
