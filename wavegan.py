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

    If batch shuffle is enabled, only a single shuffle is applied to the entire
    batch, rather than each sample in the batch.
    """
    def __init__(self, shift_factor, batch_shuffle=False):
        super(PhaseShuffle, self).__init__()
        self.shift_factor = shift_factor
        self.batch_shuffle = batch_shuffle

    def forward(self, x):
        # Return x if phase shift is disabled
        if self.shift_factor == 0:
            return x

        if self.batch_shuffle:
            # Make sure to use PyTorcTrueh to generate number RNG state is all shared
            k = int(torch.Tensor(1).random_(
                0, 2 * self.shift_factor + 1)) - self.shift_factor

            # Return if no phase shift
            if k == 0:
                return x

            # Slice feature dimension
            if k > 0:
                x_trunc = x[:, :, :-k]
                pad = (k, 0)
            else:
                x_trunc = x[:, :, -k:]
                pad = (0, -k)

            # Reflection padding
            x_shuffle = F.pad(x_trunc, pad, mode='reflect')

        else:
            # Generate shifts for each sample in the batch
            k_list = torch.Tensor(x.shape[0]).random_(0, 2*self.shift_factor+1)\
                - self.shift_factor
            k_list = k_list.numpy().astype(int)

            # Combine sample indices into lists so that less shuffle operations
            # need to be performed
            k_map = {}
            for idx, k in enumerate(k_list):
                k = int(k)
                if k not in k_map:
                    k_map[k] = []
                k_map[k].append(idx)

            # Make a copy of x for our output
            x_shuffle = x.clone()

            # Apply shuffle to each sample
            for k, idxs in k_map.items():
                if k > 0:
                    x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0),
                                            mode='reflect')
                else:
                    x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k),
                                            mode='reflect')

        assert x_shuffle.shape == x.shape, "{}, {}".format(
            x_shuffle.shape, x.shape)
        return x_shuffle


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
            reflection_padding = kernel_size // 2
            self.reflection_pad = torch.nn.ConstantPad1d(reflection_padding,
                                                         value=0)
            #             self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
            self.conv1d = torch.nn.Conv1d(in_channels, out_channels,
                                          kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv1d(out)
        return out


class WaveGANGenerator(nn.Module):
    def __init__(self,
                 model_size=64,
                 ngpus=1,
                 num_channels=1,
                 latent_dim=100,
                 post_proc_filt_len=512,
                 verbose=False,
                 upsample=True):
        super(WaveGANGenerator, self).__init__()
        self.ngpus = ngpus
        self.model_size = model_size  # d
        self.num_channels = num_channels  # c
        self.latent_dim = latent_dim
        self.post_proc_filt_len = post_proc_filt_len
        self.verbose = verbose

        self.fc1 = nn.DataParallel(nn.Linear(latent_dim, 256 * model_size))

        self.tconv1 = None
        self.tconv2 = None
        self.tconv3 = None
        self.tconv4 = None
        self.tconv5 = None

        self.upSampConv1 = None
        self.upSampConv2 = None
        self.upSampConv3 = None
        self.upSampConv4 = None
        self.upSampConv5 = None

        self.upsample = upsample

        if self.upsample:
            self.upSampConv1 = nn.DataParallel(
                UpsampleConvLayer(16 * model_size,
                                  8 * model_size,
                                  25,
                                  stride=1,
                                  upsample=4))
            self.upSampConv2 = nn.DataParallel(
                UpsampleConvLayer(8 * model_size,
                                  4 * model_size,
                                  25,
                                  stride=1,
                                  upsample=4))
            self.upSampConv3 = nn.DataParallel(
                UpsampleConvLayer(4 * model_size,
                                  2 * model_size,
                                  25,
                                  stride=1,
                                  upsample=4))
            self.upSampConv4 = nn.DataParallel(
                UpsampleConvLayer(2 * model_size,
                                  model_size,
                                  25,
                                  stride=1,
                                  upsample=4))
            self.upSampConv5 = nn.DataParallel(
                UpsampleConvLayer(model_size,
                                  num_channels,
                                  25,
                                  stride=1,
                                  upsample=4))

        else:
            self.tconv1 = nn.DataParallel(
                nn.ConvTranspose1d(16 * model_size,
                                   8 * model_size,
                                   25,
                                   stride=4,
                                   padding=11,
                                   output_padding=1))
            self.tconv2 = nn.DataParallel(
                nn.ConvTranspose1d(8 * model_size,
                                   4 * model_size,
                                   25,
                                   stride=4,
                                   padding=11,
                                   output_padding=1))
            self.tconv3 = nn.DataParallel(
                nn.ConvTranspose1d(4 * model_size,
                                   2 * model_size,
                                   25,
                                   stride=4,
                                   padding=11,
                                   output_padding=1))
            self.tconv4 = nn.DataParallel(
                nn.ConvTranspose1d(2 * model_size,
                                   model_size,
                                   25,
                                   stride=4,
                                   padding=11,
                                   output_padding=1))
            self.tconv5 = nn.DataParallel(
                nn.ConvTranspose1d(model_size,
                                   num_channels,
                                   25,
                                   stride=4,
                                   padding=11,
                                   output_padding=1))

        if post_proc_filt_len:
            self.ppfilter1 = nn.DataParallel(
                nn.Conv1d(num_channels, num_channels, post_proc_filt_len))

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):

        x = self.fc1(x).view(-1, 16 * self.model_size, 16)
        x = F.relu(x)
        output = None
        if self.verbose:
            print(x.shape)

        if self.upsample:
            x = F.relu(self.upSampConv1(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv2(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv3(x))
            if self.verbose:
                print(x.shape)

            x = F.relu(self.upSampConv4(x))
            if self.verbose:
                print(x.shape)

            output = torch.tanh(self.upSampConv5(x))
        else:
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

            output = torch.tanh(self.tconv5(x))

        if self.verbose:
            print(output.shape)

        if self.post_proc_filt_len:
            # Pad for "same" filtering
            if (self.post_proc_filt_len % 2) == 0:
                pad_left = self.post_proc_filt_len // 2
                pad_right = pad_left - 1
            else:
                pad_left = (self.post_proc_filt_len - 1) // 2
                pad_right = pad_left
            output = self.ppfilter1(F.pad(output, (pad_left, pad_right)))
            if self.verbose:
                print(output.shape)

        return output


class WaveGANDiscriminator(nn.Module):
    def __init__(self,
                 model_size=64,
                 ngpus=1,
                 num_channels=1,
                 shift_factor=2,
                 alpha=0.2,
                 batch_shuffle=False,
                 verbose=False):
        super(WaveGANDiscriminator, self).__init__()
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.DataParallel(
            nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11))
        self.conv2 = nn.DataParallel(
            nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11))
        self.conv3 = nn.DataParallel(
            nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4,
                      padding=11))
        self.conv4 = nn.DataParallel(
            nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4,
                      padding=11))
        self.conv5 = nn.DataParallel(
            nn.Conv1d(8 * model_size,
                      16 * model_size,
                      25,
                      stride=4,
                      padding=11))
        self.ps1 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps2 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps3 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps4 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.fc1 = nn.DataParallel(nn.Linear(256 * model_size, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

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

        return torch.sigmoid(self.fc1(x))


def load_wavegan_generator(filepath,
                           model_size=64,
                           ngpus=1,
                           num_channels=1,
                           latent_dim=100,
                           post_proc_filt_len=512,
                           **kwargs):
    model = WaveGANGenerator(model_size=model_size,
                             ngpus=ngpus,
                             num_channels=num_channels,
                             latent_dim=latent_dim,
                             post_proc_filt_len=post_proc_filt_len)
    model.load_state_dict(torch.load(filepath))

    return model


def load_wavegan_discriminator(filepath,
                               model_size=64,
                               ngpus=1,
                               num_channels=1,
                               shift_factor=2,
                               alpha=0.2,
                               **kwargs):
    model = WaveGANDiscriminator(model_size=model_size,
                                 ngpus=ngpus,
                                 num_channels=num_channels,
                                 shift_factor=shift_factor,
                                 alpha=alpha)
    model.load_state_dict(torch.load(filepath))

    return model


class WaveGANQ(nn.Module):
    def __init__(self,
                 model_size=64,
                 ngpus=1,
                 num_channels=1,
                 shift_factor=2,
                 alpha=0.2,
                 batch_shuffle=False,
                 verbose=False,
                 num_categ=10):
        super(WaveGANQ, self).__init__()
        self.model_size = model_size  # d
        self.ngpus = ngpus
        self.num_channels = num_channels  # c
        self.shift_factor = shift_factor  # n
        self.alpha = alpha
        self.verbose = verbose
        self.num_categ = num_categ
        # Conv2d(in_channels, out_channels, kernel_size, stride=1, etc.)
        self.conv1 = nn.DataParallel(
            nn.Conv1d(num_channels, model_size, 25, stride=4, padding=11))
        self.conv2 = nn.DataParallel(
            nn.Conv1d(model_size, 2 * model_size, 25, stride=4, padding=11))
        self.conv3 = nn.DataParallel(
            nn.Conv1d(2 * model_size, 4 * model_size, 25, stride=4,
                      padding=11))
        self.conv4 = nn.DataParallel(
            nn.Conv1d(4 * model_size, 8 * model_size, 25, stride=4,
                      padding=11))
        self.conv5 = nn.DataParallel(
            nn.Conv1d(8 * model_size,
                      16 * model_size,
                      25,
                      stride=4,
                      padding=11))
        self.ps1 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps2 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps3 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.ps4 = PhaseShuffle(shift_factor, batch_shuffle=batch_shuffle)
        self.fc1 = nn.DataParallel(nn.Linear(256 * model_size, num_categ))

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

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

        x = self.fc1(x)

        if self.verbose:
            print(x.shape)

        x = torch.sigmoid(x)
        return x