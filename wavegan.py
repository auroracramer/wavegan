import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data


class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary
    """

    def __init__(self, shift_factor):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor

    def forward(self, x):
        # Make sure to use PyTorch to generate number RNG state is all shared
        k = int(torch.Tensor(1).random_(0, self.shift_factor + 1)) - 5

        # Return if no phase shift
        if k == 0:
            return x

        # Slice feature dimension
        if k > 0:
            x_trunc = x[:, :, :-k]
            pad = (0, k)
        else:
            x_trunc = x[:, :, -k:]
            pad = (-k, 0)

        # Reflection padding
        x_shuffle = F.pad(x_trunc, pad, mode='reflect')
        assert x_shuffle.shape == x.shape, "{}, {}".format(x_shuffle.shape,
                                                           x.shape)
        return x_shuffle


class WaveGANGenerator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, latent_dim=100, verbose=False):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size # d
        self.num_channels = num_channels # c
        self.latent_dim = latent_dim
        self.verbose = verbose

        self.fc1 = nn.DataParallel(nn.Linear(latent_dim, 256 * model_size))

        self.tconv1 = nn.DataParallel(
            nn.ConvTranspose1d(16 * model_size, 8 * model_size, 25, stride=4, padding=11,
                               output_padding=1))
        self.tconv2 = nn.DataParallel(
            nn.ConvTranspose1d(8 * model_size, 4 * model_size, 25, stride=4, padding=11,
                               output_padding=1))
        self.tconv3 = nn.DataParallel(
            nn.ConvTranspose1d(4 * model_size, 2 * model_size, 25, stride=4, padding=11,
                               output_padding=1))
        self.tconv4 = nn.DataParallel(
            nn.ConvTranspose1d(2 * model_size, model_size, 25, stride=4, padding=11,
                               output_padding=1))
        self.tconv5 = nn.DataParallel(
            nn.ConvTranspose1d(model_size, num_channels, 25, stride=4, padding=11,
                               output_padding=1))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):

        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        if self.verbose:
            print(x.shape)

        x = F.relu(self.tconv1(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.tconv2(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.tconv3(x))
        if self.verbose:
            print(x.shape)

        x = F.relu(self.tconv4(x))
        if self.verbose:
            print(x.shape)

        output = F.tanh(self.tconv5(x))
        if self.verbose:
            print(output.shape)

        return output


class WaveGANDiscriminator(nn.Module):
    def __init__(self, model_size=64, ngpus=1, num_channels=1, shift_factor=2, alpha=0.2, verbose=False):
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size # d
        self.ngpus = ngpus
        self.num_channels = num_channels # c
        self.shift_factor = shift_factor # n
        self.alpha = alpha
        self.verbose = verbose
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.DataParallel(nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11))
        self.conv2 = nn.DataParallel(
            nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11))
        self.conv3 = nn.DataParallel(
            nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4, padding=11))
        self.conv4 = nn.DataParallel(
            nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4, padding=11))
        self.conv5 = nn.DataParallel(
            nn.Conv1d(8 * model_size, 16 * model_size, 25, stride=4, padding=11))
        self.ps1 = PhaseShuffle(shift_factor)
        self.ps2 = PhaseShuffle(shift_factor)
        self.ps3 = PhaseShuffle(shift_factor)
        self.ps4 = PhaseShuffle(shift_factor)
        self.fc1 = nn.DataParallel(nn.Linear(256 * model_size, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps1(x)

        x = F.leaky_relu(self.conv2(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps2(x)

        x = F.leaky_relu(self.conv3(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps3(x)

        x = F.leaky_relu(self.conv4(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)
        x = self.ps4(x)

        x = F.leaky_relu(self.conv5(x), negative_slope=self.alpha)
        if self.verbose:
            print(x.shape)

        x = x.view(-1, 256 * self.model_size)
        if self.verbose:
            print(x.shape)

        return F.sigmoid(self.fc1(x))


def load_wavegan_generator(filepath, model_size=64, ngpus=1, num_channels=1,
                           latent_dim=100, **kwargs):
    model = WaveGANGenerator(model_size=model_size, ngpus=ngpus,
                             num_channels=num_channels, latent_dim=latent_dim)
    model.load_state_dict(torch.load(filepath))

    return model


def load_wavegan_discriminator(filepath, model_size=64, ngpus=1, num_channels=1,
                               shift_factor=2, alpha=0.2, **kwargs):
    model = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus,
                                 num_channels=num_channels,
                                 shift_factor=shift_factor, alpha=alpha)
    model.load_state_dict(torch.load(filepath))

    return model
