import argparse
import logging
import os
import json
import datetime
import torch
import pprint
import pickle as pk
from sample import get_all_audio_filepaths, create_data_split
from wavegan import WaveGANDiscriminator, WaveGANGenerator
from wgan import train_wgan
from log import init_console_logger
from utils import save_samples


LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def parse_arguments():
    """
    Get command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a WaveGAN on a given set of audio')

    parser.add_argument('-ms', '--model-size', dest='model_size', type=int, default=64, help='Model size parameter used in WaveGAN')
    parser.add_argument('-pssf', '--phase-shuffle-shift-factor', dest='shift_factor', type=int, default=2, help='Maximum shift used by phase shuffle')
    parser.add_argument('-psb', '--phase-shuffle-batchwise', dest='batch_shuffle', action='store_true', help='If true, apply phase shuffle to entire batches rather than individual samples')
    parser.add_argument('-ppfl', '--post-proc-filt-len', dest='post_proc_filt_len', type=int, default=512, help='Length of post processing filter used by generator. Set to 0 to disable.')
    parser.add_argument('-lra', '--lrelu-alpha', dest='alpha', type=float, default=0.2, help='Slope of negative part of LReLU used by discriminator')
    parser.add_argument('-vr', '--valid-ratio', dest='valid_ratio', type=float, default=0.1, help='Ratio of audio files used for validation')
    parser.add_argument('-tr', '--test-ratio', dest='test_ratio', type=float, default=0.1, help='Ratio of audio files used for testing')
    parser.add_argument('-bs', '--batch-size', dest='batch_size', type=int, default=64, help='Batch size used for training')
    parser.add_argument('-ne', '--num-epochs', dest='num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-bpe', '--batches-per-epoch', dest='batches_per_epoch', type=int, default=10, help='Batches per training epoch')
    parser.add_argument('-ng', '--ngpus', dest='ngpus', type=int, default=1, help='Number of GPUs to use for training')
    parser.add_argument('-du', '--discriminator-updates', dest='discriminator_updates', type=int, default=5, help='Number of discriminator updates per training iteration')
    parser.add_argument('-ld', '--latent-dim', dest='latent_dim', type=int, default=100, help='Size of latent dimension used by generator')
    parser.add_argument('-eps', '--epochs-per-sample', dest='epochs_per_sample', type=int, default=1, help='How many epochs between every set of samples generated for inspection')
    parser.add_argument('-ss', '--sample-size', dest='sample_size', type=int, default=20, help='Number of inspection samples generated')
    parser.add_argument('-rf', '--regularization-factor', dest='lmbda', type=float, default=10.0, help='Gradient penalty regularization factor')
    parser.add_argument('-lr', '--learning-rate', dest='learning_rate', type=float, default=1e-4, help='Initial ADAM learning rate')
    parser.add_argument('-bo', '--beta-one', dest='beta1', type=float, default=0.5, help='beta_1 ADAM parameter')
    parser.add_argument('-bt', '--beta-two', dest='beta2', type=float, default=0.9, help='beta_2 ADAM parameter')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('audio_dir', type=str, help='Path to directory containing audio files')
    parser.add_argument('output_dir', type=str, help='Path to directory where model files will be output')
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_arguments()

    init_console_logger(LOGGER, args['verbose'])

    LOGGER.info('Initialized logger.')

    batch_size = args['batch_size']
    latent_dim = args['latent_dim']
    ngpus = args['ngpus']
    model_size = args['model_size']
    model_dir = os.path.join(args['output_dir'],
                              datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    args['model_dir'] = model_dir

    # Create output directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    LOGGER.info('Saving configurations...')
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args, f)

    # Try on some training data
    LOGGER.info('Loading audio data...')
    audio_filepaths = get_all_audio_filepaths(args['audio_dir'])
    train_gen, valid_data, test_data \
        = create_data_split(audio_filepaths, args['valid_ratio'], args['test_ratio'],
                            batch_size, batch_size, batch_size)

    LOGGER.info('Creating models...')
    model_gen = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim,
                                 post_proc_filt_len=args['post_proc_filt_len'], upsample=True)
    model_dis = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus,
                                     alpha=args['alpha'], shift_factor=args['shift_factor'],
                                     batch_shuffle=args['batch_shuffle'])

    LOGGER.info('Starting training...')
    model_gen, model_dis, history, final_discr_metrics, samples = train_wgan(
        model_gen=model_gen,
        model_dis=model_dis,
        train_gen=train_gen,
        valid_data=valid_data,
        test_data=test_data,
        num_epochs=args['num_epochs'],
        batches_per_epoch=args['batches_per_epoch'],
        batch_size=batch_size,
        output_dir=model_dir,
        lr=args['learning_rate'],
        beta_1=args['beta1'],
        beta_2=args['beta2'],
        lmbda=args['lmbda'],
        use_cuda=ngpus>=1,
        discriminator_updates=args['discriminator_updates'],
        latent_dim=latent_dim,
        epochs_per_sample=args['epochs_per_sample'],
        sample_size=args['sample_size'])

    LOGGER.info('Finished training.')

    LOGGER.info('Final discriminator loss on validation and test:')
    LOGGER.info(pprint.pformat(final_discr_metrics))

    LOGGER.info('Saving models...')
    model_gen_output_path = os.path.join(model_dir, "model_gen.pkl")
    model_dis_output_path = os.path.join(model_dir, "model_dis.pkl")
    torch.save(model_gen.state_dict(), model_gen_output_path,
               pickle_protocol=pk.HIGHEST_PROTOCOL)
    torch.save(model_dis.state_dict(), model_dis_output_path,
               pickle_protocol=pk.HIGHEST_PROTOCOL)

    LOGGER.info('Saving metrics...')
    history_output_path = os.path.join(model_dir, "history.pkl")
    final_discr_metrics_output_path = os.path.join(model_dir, "final_discr_metrics.pkl")
    with open(history_output_path, 'wb') as f:
        pk.dump(history, f)
    with open(final_discr_metrics_output_path, 'wb') as f:
        pk.dump(final_discr_metrics, f)

    LOGGER.info('Done!')
