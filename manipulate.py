from wgan import get_random_z
import argparse
from wavegan import WaveGANDiscriminator, WaveGANGenerator, WaveGANQ
import os
from pathlib import Path
import torch
from utils import save_samples
from tqdm import trange
import pickle


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Analyze a fiwGAN on a given latent code c')

    parser.add_argument('--model-size',
                        dest='model_size',
                        type=int,
                        default=64,
                        help='Model size parameter used in WaveGAN')
    parser.add_argument(
        '-ppfl',
        '--post-proc-filt-len',
        dest='post_proc_filt_len',
        type=int,
        default=512,
        help=
        'Length of post processing filter used by generator. Set to 0 to disable.'
    )
    parser.add_argument('--batch-size',
                        dest='batch_size',
                        type=int,
                        default=64,
                        help='Batch size used for training')
    parser.add_argument('--ngpus',
                        dest='ngpus',
                        type=int,
                        default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--latent-dim',
                        dest='latent_dim',
                        type=int,
                        default=100,
                        help='Size of latent dimension used by generator')
    parser.add_argument('--verbose',
                        dest='verbose',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to directory where model files will be output')
    parser.add_argument('--num_categ',
                        dest='num_categ',
                        type=int,
                        default=3,
                        help='Number of categorical variables')
    parser.add_argument('--model_path',
                        dest='model_path',
                        type=str,
                        help="the path of the model")
    parser.add_argument('--random_range',
                        dest='random_range',
                        type=int,
                        help="latent variable range")
    parser.add_argument('--num-epochs',
                        dest='num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs')

    parser.add_argument('--job_id', type=str)
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    batch_size = args['batch_size']
    latent_dim = args['latent_dim']
    ngpus = args['ngpus']
    model_size = args['model_size']
    model_dir = os.path.join(args['output_dir'], args["job_id"])
    args['model_dir'] = Path(model_dir)
    model_dir = args['model_dir']
    Q_num_categ = args['num_categ']
    model_path = Path(args['model_path'])
    random_range = args['random_range']
    output_dir = Path(args['output_dir'])
    num_epochs = args['num_epochs']
    #load model
    model_gen = WaveGANGenerator(model_size=model_size,
                                 ngpus=ngpus,
                                 latent_dim=latent_dim,
                                 post_proc_filt_len=args['post_proc_filt_len'],
                                 upsample=True,
                                 verbose=args["verbose"])
    model_gen.load_state_dict(torch.load(model_path / "Gen.pkl"))

    #Starting: analyze the model
    samples = {}
    for epoch in trange(num_epochs):
        noise_v = get_random_z(Q_num_categ,
                               batch_size,
                               latent_dim,
                               use_cuda=ngpus >= 1,
                               random_range=random_range)

        if (epoch + 1) % epochs_per_sample == 0:
            # Generate outputs for fixed latent samples
            samp_output = model_gen.forward(sample_noise_v)
            if ngpus >= 1:
                samp_output = samp_output.cpu()

            samples[epoch + 1] = samp_output.data.numpy()
            if output_dir:
                save_samples(samples[epoch + 1], epoch + 1,
                             output_dir / "Audio")
