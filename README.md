# WaveGAN

A PyTorch reimplementation of WaveGAN
[(Donahue, et al. 2018)](https://arxiv.org/abs/1802.04208).


### Setup
This code requires Python 3 and `ffmpeg` (which can be installed with `conda`),
along with the following packages (which can be installed with `conda`  or `pip`):
* `pytorch`
* `numpy`
* `librosa`
* `pescador` (for sampling)


### Running
You can train the WaveGAN with audio from a given directory by using
`train_wavegan.py`. To run with default parameters, run:

`python train_wavegan.py <audio_dir> <output_dir>`

To see the full list of arguments, run `python train_wavegan.py -h`. Note that
the data split for validation and testing is done on a filewise basis. For SLURM
users, an example `sbatch` script is provided in `jobs/train-wavegan.sbatch`.
