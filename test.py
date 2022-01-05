import torch
from torch import autograd


def get_random_z(wavegan_cate: int,
                 batch_size,
                 latent_dim,
                 use_cuda: bool = True):
    noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
    sec = noise[:, :wavegan_cate]
    sec[sec > 0] = 1
    sec[sec <= 0] = 0

    if use_cuda:
        noise = noise.cuda()
    noise_v = autograd.Variable(noise)

    return noise_v
