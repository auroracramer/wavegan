import logging
import torch
from torch import autograd
from torch import optim
from utils import save_samples
import numpy as np
import pprint
from wavegan import WaveGANDiscriminator, WaveGANGenerator, WaveGANQ
from tqdm import trange
from pathlib import Path
import pickle as pk

LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)


def get_random_z(
    wavegan_cate: int,
    batch_size,
    latent_dim,
    with_grad: bool = True,
    use_cuda: bool = True,
    random_range=1,
):
    noise = torch.Tensor(batch_size,
                         latent_dim).uniform_(-random_range, random_range)
    sec = noise[:, :wavegan_cate]
    sec[sec > 0] = 1
    sec[sec <= 0] = 0
    #noise[:,:wavegan_cate] =sec

    if use_cuda:
        noise = noise.cuda()
    if (with_grad):
        noise_v = autograd.Variable(noise)
    else:
        with torch.no_grad():
            noise_v = autograd.Variable(noise)
    return noise_v


def get_manipulate_z(
    wavegan_cate: int,
    batch_size,
    latent_dim,
    with_grad: bool = True,
    use_cuda: bool = True,
    random_range=1,
):
    noise = torch.Tensor(batch_size,
                         latent_dim).uniform_(-random_range, random_range)
    noise[:, 40] = 11
    sec = noise[:, :wavegan_cate]
    sec[sec > 0] = 1
    sec[sec <= 0] = 0
    #noise[:,:wavegan_cate] =sec

    if use_cuda:
        noise = noise.cuda()
    if (with_grad):
        noise_v = autograd.Variable(noise)
    else:
        with torch.no_grad():
            noise_v = autograd.Variable(noise)
    return noise_v


def compute_discr_loss_terms(model_dis: WaveGANDiscriminator,
                             model_gen: WaveGANGenerator,
                             real_data_v,
                             noise_v,
                             batch_size,
                             latent_dim,
                             lmbda,
                             use_cuda,
                             compute_grads=False):
    # Convenient values for
    one = torch.FloatTensor([1])
    neg_one = one * -1
    if use_cuda:
        one = one.cuda()
        neg_one = neg_one.cuda()

    # Reset gradients
    model_dis.zero_grad()

    # a) Compute loss contribution from real training data and backprop
    # (negative of the empirical mean, w.r.t. the data distribution, of the discr. output)

    D_real = model_dis.forward(real_data_v)

    D_real = D_real.mean(axis=0)
    # Negate since we want to _maximize_ this quantity
    if compute_grads:
        D_real.backward(neg_one)

    # b) Compute loss contribution from generated data and backprop
    # (empirical mean, w.r.t. the generator distribution, of the discr. output)
    # Generate noise in latent space

    # Generate data by passing noise through the generator
    fake = autograd.Variable(model_gen.forward(noise_v).data)
    inputv = fake
    D_fake = model_dis.forward(inputv)
    D_fake = D_fake.mean(axis=0)
    if compute_grads:
        D_fake.backward(one)

    # c) Compute gradient penalty and backprop
    gradient_penalty = calc_gradient_penalty(model_dis,
                                             real_data_v.data,
                                             fake.data,
                                             batch_size,
                                             lmbda,
                                             use_cuda=use_cuda)

    if compute_grads:
        gradient_penalty.backward(one)

    # Compute metrics and record in batch history
    D_cost = D_fake - D_real + gradient_penalty
    Wasserstein_D = D_real - D_fake

    return D_cost, Wasserstein_D


def compute_q_loss_terms(model_gen: WaveGANGenerator,
                         wavegan: WaveGANQ,
                         criterion: torch.nn.BCEWithLogitsLoss,
                         batch_size,
                         latent_dim,
                         use_cuda,
                         wavegan_cate: int = 3,
                         compute_grads=False):
    one = torch.FloatTensor([1])
    neg_one = one * -1
    if use_cuda:
        one = one.cuda()
        neg_one = neg_one.cuda()
    model_gen.zero_grad()
    wavegan.zero_grad()

    z = get_random_z(wavegan_cate, batch_size, latent_dim, use_cuda)

    Q = wavegan.forward(model_gen.forward(z))
    """
    sig_Q = torch.sigmoid(Q)
    sig_Q[sig_Q > 0.5] = 1
    sig_Q[sig_Q <= 0.5] = 0
    print(torch.abs(sig_Q - z[:, :wavegan_cate]).sum())
    """

    #print(Q.shape, z[:, :wavegan_cate].shape)
    Q_loss = criterion.forward(Q, z[:, :wavegan_cate]).unsqueeze(dim=0)
    #print(Q_loss, Q_loss.shape)

    #print(one.shape)
    #input()
    #Q_loss = Q_loss.mean(axis=0)
    if compute_grads:
        Q_loss.backward(one)
    #Q = torch.sigmoid_cross
    #pass
    return Q_loss


def compute_batch_fft_difference(real_data, fake_data):
    #print(int(real_data.shape[-1] / 2))
    fft_real = torch.fft.fft(real_data)[:, :, :int(real_data.shape[-1] / 2)]
    fft_fake = torch.fft.fft(fake_data)[:, :, :int(real_data.shape[-1] / 2)]

    mean_fft_real = torch.mean(fft_real, dim=0)
    mean_fft_fake = torch.mean(fft_fake, dim=0)
    #print(f"fft_real {fft_real.shape}")
    #print(f"fft_fake {fft_fake.shape}")
    #print(f"mean_fft_real {mean_fft_real.shape}")
    #print(f"mean_fft_fake {mean_fft_fake.shape}")

    diff = torch.sum((mean_fft_real - mean_fft_fake)**2).real
    print(f"diff {diff}")
    #input()


def compute_gener_loss_terms(model_dis: WaveGANDiscriminator,
                             model_gen: WaveGANGenerator,
                             batch_size,
                             latent_dim,
                             use_cuda,
                             wavegan_cate: int = 3,
                             compute_grads=False):
    # Convenient values for
    one = torch.FloatTensor([1])
    neg_one = one * -1
    if use_cuda:
        one = one.cuda()
        neg_one = neg_one.cuda()

    # Reset generator gradients
    model_gen.zero_grad()

    # Sample from the generator
    noise_v = get_random_z(wavegan_cate, batch_size, latent_dim, use_cuda)
    fake = model_gen.forward(noise_v)

    # Compute generator loss and backprop
    # (negative of empirical mean (w.r.t generator distribution) of discriminator output)
    G = model_dis.forward(fake)
    G = G.mean(dim=0)
    if compute_grads:
        G.backward(neg_one)
    G_cost = -G

    return G_cost


def np_to_input_var(data, use_cuda):
    data = data[:, np.newaxis, :]
    data = torch.Tensor(data)
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data)


def dataset_to_input_var(data, use_cuda):
    if use_cuda:
        data = data.cuda()
    return autograd.Variable(data)


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(model_dis: WaveGANDiscriminator,
                          real_data,
                          fake_data,
                          batch_size,
                          lmbda,
                          use_cuda=True):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = model_dis.forward(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda()
        if use_cuda else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to encourage discriminator
    # to be a 1-Lipschitz function
    gp = (gradients.norm(2, dim=1) - 1)**2

    gp = torch.mean(gp, dim=0, keepdim=True)

    gp = gp * lmbda
    return gp


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def train_wgan(model_gen: WaveGANGenerator,
               model_dis: WaveGANDiscriminator,
               train_gen,
               valid_data,
               test_data,
               num_epochs,
               batches_per_epoch,
               batch_size,
               output_dir: Path = None,
               lmbda=0.1,
               use_cuda=True,
               discriminator_updates=5,
               epochs_per_sample=10,
               sample_size=20,
               lr=1e-4,
               beta_1=0.5,
               beta_2=0.9,
               latent_dim=100):

    if use_cuda:
        model_gen = model_gen.cuda()
        model_dis = model_dis.cuda()

    # Initialize optimizers for each model
    optimizer_gen = optim.Adam(model_gen.parameters(),
                               lr=lr,
                               betas=(beta_1, beta_2))
    optimizer_dis = optim.Adam(model_dis.parameters(),
                               lr=lr,
                               betas=(beta_1, beta_2))

    # Sample noise used for seeing the evolution of generated output samples throughout training
    sample_noise = torch.Tensor(sample_size, latent_dim).uniform_(-1, 1)
    if use_cuda:
        sample_noise = sample_noise.cuda()
    sample_noise_v = autograd.Variable(sample_noise)

    samples = {}
    history = []

    train_iter = iter(train_gen)
    valid_iter = iter(valid_data)
    test_iter = iter(test_data)

    valid_data_v = dataset_to_input_var(next(valid_iter), use_cuda)
    test_data_v = dataset_to_input_var(next(test_iter), use_cuda)

    model_output_path = output_dir / "model"
    model_output_path.mkdir(parents=True, exist_ok=True)

    save_samples(valid_data_v.cpu().data.numpy(), 0, output_dir / "Sample")
    # Loop over the dataset multiple times
    for epoch in trange(num_epochs):
        LOGGER.info("Epoch: {}/{}".format(epoch + 1, num_epochs))

        epoch_history = []

        for batch_idx in range(batches_per_epoch):

            # Set model parameters to require gradients to be computed and stored
            for p in model_dis.parameters():
                p.requires_grad = True
            for p in model_gen.parameters():
                p.requires_grad = False
            # Initialize the metrics for this batch
            batch_history = {'discriminator': [], 'generator': {}}

            # Discriminator Training Phase:
            # -> Train discriminator k times
            for iter_d in range(discriminator_updates):
                # Get real examples
                try:
                    real_data_v = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_gen)
                    real_data_v = next(train_iter)

                real_data_v = dataset_to_input_var(real_data_v, use_cuda)

                # Get noise
                noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
                if use_cuda:
                    noise = noise.cuda()
                with torch.no_grad():
                    noise_v = autograd.Variable(noise)
                # totally freeze model_gen

                # Get new batch of real training data
                D_cost_train, D_wass_train = compute_discr_loss_terms(
                    model_dis,
                    model_gen,
                    real_data_v,
                    noise_v,
                    batch_size,
                    latent_dim,
                    lmbda,
                    use_cuda,
                    compute_grads=True)

                # Update the discriminator
                optimizer_dis.step()

            D_cost_valid, D_wass_valid = compute_discr_loss_terms(
                model_dis,
                model_gen,
                valid_data_v,
                noise_v,
                batch_size,
                latent_dim,
                lmbda,
                use_cuda,
                compute_grads=False)

            if use_cuda:
                D_cost_train = D_cost_train.cpu()
                D_cost_valid = D_cost_valid.cpu()
                D_wass_train = D_wass_train.cpu()
                D_wass_valid = D_wass_valid.cpu()

            batch_history['discriminator'].append({
                'cost':
                D_cost_train.data.numpy()[0],
                'wasserstein_cost':
                D_wass_train.data.numpy()[0],
                'cost_validation':
                D_cost_valid.data.numpy()[0],
                'wasserstein_cost_validation':
                D_wass_valid.data.numpy()[0]
            })

            ############################
            # (2) Update G network
            ###########################

            # Prevent discriminator from computing gradients, since
            # we are only updating the generator
            for p in model_dis.parameters():
                p.requires_grad = False
            for p in model_gen.parameters():
                p.requires_grad = True

            G_cost = compute_gener_loss_terms(model_dis,
                                              model_gen,
                                              batch_size,
                                              latent_dim,
                                              use_cuda,
                                              compute_grads=True)

            optimizer_gen.step()

            if use_cuda:
                G_cost = G_cost.cpu()

            # Record generator loss
            batch_history['generator']['cost'] = G_cost.data.numpy()[0]

            # Record batch metrics
            epoch_history.append(batch_history)

        # Record epoch metrics
        history.append(epoch_history)

        LOGGER.info(pprint.pformat(epoch_history[-1]))

        if (epoch + 1) % epochs_per_sample == 0:
            # Generate outputs for fixed latent samples
            LOGGER.info('Generating samples...')
            samp_output = model_gen.forward(sample_noise_v)
            if use_cuda:
                samp_output = samp_output.cpu()

            samples[epoch + 1] = samp_output.data.numpy()
            if output_dir:
                LOGGER.info('Saving samples...')
                save_samples(samples[epoch + 1], epoch + 1,
                             output_dir / "Audio")

        #save model
        model_epoch_output_path = model_output_path / f"{epoch+1}"
        model_epoch_output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_gen.state_dict(),
                   model_epoch_output_path / f"Gen.pkl",
                   pickle_protocol=pk.HIGHEST_PROTOCOL)
        torch.save(model_dis.state_dict(),
                   model_epoch_output_path / f"Disc.pkl",
                   pickle_protocol=pk.HIGHEST_PROTOCOL)

    ## Get final discriminator loss
    # Get noise
    noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
    if use_cuda:
        noise = noise.cuda()
    with torch.no_grad():
        noise_v = autograd.Variable(noise)  # totally freeze model_gen

    # Get new batch of real training data
    D_cost_test, D_wass_test = compute_discr_loss_terms(model_dis,
                                                        model_gen,
                                                        test_data_v,
                                                        noise_v,
                                                        batch_size,
                                                        latent_dim,
                                                        lmbda,
                                                        use_cuda,
                                                        compute_grads=False)

    D_cost_valid, D_wass_valid = compute_discr_loss_terms(model_dis,
                                                          model_gen,
                                                          valid_data_v,
                                                          noise_v,
                                                          batch_size,
                                                          latent_dim,
                                                          lmbda,
                                                          use_cuda,
                                                          compute_grads=False)

    if use_cuda:
        D_cost_test = D_cost_test.cpu()
        D_cost_valid = D_cost_valid.cpu()
        D_wass_test = D_wass_test.cpu()
        D_wass_valid = D_wass_valid.cpu()

    final_discr_metrics = {
        'cost_validation': D_cost_valid.data.numpy()[0],
        'wasserstein_cost_validation': D_wass_valid.data.numpy()[0],
        'cost_test': D_cost_test.data.numpy()[0],
        'wasserstein_cost_test': D_wass_test.data.numpy()[0],
    }

    return model_gen, model_dis, history, final_discr_metrics, samples


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def train_wganQ(model_gen: WaveGANGenerator,
                model_dis: WaveGANDiscriminator,
                model_Q: WaveGANQ,
                train_gen,
                valid_data,
                test_data,
                num_epochs,
                batches_per_epoch,
                batch_size,
                Q_num_categ,
                output_dir: Path = None,
                lmbda=0.1,
                use_cuda=True,
                discriminator_updates=5,
                epochs_per_sample=10,
                sample_size=20,
                lr=1e-4,
                beta_1=0.5,
                beta_2=0.9,
                latent_dim=100):

    if use_cuda:
        model_gen = model_gen.cuda()
        model_dis = model_dis.cuda()
        model_Q = model_Q.cuda()

    # Initialize optimizers for each model
    optimizer_gen = optim.Adam(model_gen.parameters(),
                               lr=lr,
                               betas=(beta_1, beta_2))
    optimizer_dis = optim.Adam(model_dis.parameters(),
                               lr=lr,
                               betas=(beta_1, beta_2))
    optimizer_wgan = optim.RMSprop(model_Q.parameters(), lr=lr)
    # Sample noise used for seeing the evolution of generated output samples throughout training
    sample_noise_v = get_random_z(Q_num_categ,
                                  batch_size,
                                  latent_dim,
                                  use_cuda=use_cuda)

    samples = {}
    history = []

    train_iter = iter(train_gen)
    valid_iter = iter(valid_data)
    test_iter = iter(test_data)

    valid_data_v = dataset_to_input_var(next(valid_iter), use_cuda)
    test_data_v = dataset_to_input_var(next(test_iter), use_cuda)

    model_output_path = output_dir / "model"
    model_output_path.mkdir(parents=True, exist_ok=True)

    save_samples(valid_data_v.cpu().data.numpy(), 0, output_dir / "Sample")
    wavegan_criterion = torch.nn.BCEWithLogitsLoss()
    #wavegan_criterion.eval()
    # Loop over the dataset multiple times
    for epoch in trange(num_epochs):
        LOGGER.info("Epoch: {}/{}".format(epoch + 1, num_epochs))

        epoch_history = []

        for batch_idx in range(batches_per_epoch):

            # Set model parameters to require gradients to be computed and stored
            for p in model_dis.parameters():
                p.requires_grad = True
            for p in model_gen.parameters():
                p.requires_grad = False
            for p in model_Q.parameters():
                p.requires_grad = False
            # Initialize the metrics for this batch
            batch_history = {'discriminator': [], 'generator': {}, "Q": {}}

            # Discriminator Training Phase:
            # -> Train discriminator k times
            for iter_d in range(discriminator_updates):
                # Get real examples
                try:
                    real_data_v = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_gen)
                    real_data_v = next(train_iter)

                real_data_v = dataset_to_input_var(real_data_v, use_cuda)

                # Get noise
                noise_v = get_random_z(Q_num_categ,
                                       batch_size,
                                       latent_dim,
                                       with_grad=False,
                                       use_cuda=use_cuda)
                # totally freeze model_gen

                # Get new batch of real training data
                D_cost_train, D_wass_train = compute_discr_loss_terms(
                    model_dis,
                    model_gen,
                    real_data_v,
                    noise_v,
                    batch_size,
                    latent_dim,
                    lmbda,
                    use_cuda,
                    compute_grads=True)

                # Update the discriminator
                optimizer_dis.step()

            D_cost_valid, D_wass_valid = compute_discr_loss_terms(
                model_dis,
                model_gen,
                valid_data_v,
                noise_v,
                batch_size,
                latent_dim,
                lmbda,
                use_cuda,
                compute_grads=False)

            if use_cuda:
                D_cost_train = D_cost_train.cpu()
                D_cost_valid = D_cost_valid.cpu()
                D_wass_train = D_wass_train.cpu()
                D_wass_valid = D_wass_valid.cpu()

            batch_history['discriminator'].append({
                'cost':
                D_cost_train.data.numpy()[0],
                'wasserstein_cost':
                D_wass_train.data.numpy()[0],
                'cost_validation':
                D_cost_valid.data.numpy()[0],
                'wasserstein_cost_validation':
                D_wass_valid.data.numpy()[0]
            })

            ############################
            # (2) Update G network
            ###########################

            # Prevent discriminator from computing gradients, since
            # we are only updating the generator
            for p in model_dis.parameters():
                p.requires_grad = False
            for p in model_gen.parameters():
                p.requires_grad = True
            for p in model_Q.parameters():
                p.requires_grad = True

            G_cost = compute_gener_loss_terms(
                model_dis,
                model_gen,
                batch_size,
                latent_dim,
                use_cuda,
                wavegan_cate=Q_num_categ,
                compute_grads=True,
            )
            optimizer_gen.step()

            if use_cuda:
                G_cost = G_cost.cpu()
            compute_batch_fft_difference(
                real_data_v,
                model_gen.forward(noise_v),
            )
            # Record generator loss
            batch_history['generator']['cost'] = G_cost.data.numpy()[0]
            Q_cost = compute_q_loss_terms(model_gen,
                                          model_Q,
                                          wavegan_criterion,
                                          batch_size,
                                          latent_dim,
                                          use_cuda,
                                          wavegan_cate=Q_num_categ,
                                          compute_grads=True)
            # Update generator
            optimizer_gen.step()
            optimizer_wgan.step()
            if use_cuda:
                Q_cost = Q_cost.cpu()
            batch_history['Q']['cost'] = Q_cost.data.numpy()[0]
            # Record batch metrics
            epoch_history.append(batch_history)

        # Record epoch metrics
        history.append(epoch_history)

        LOGGER.info(pprint.pformat(epoch_history[-1]))

        if (epoch + 1) % epochs_per_sample == 0:
            # Generate outputs for fixed latent samples
            LOGGER.info('Generating samples...')
            samp_output = model_gen.forward(sample_noise_v)
            if use_cuda:
                samp_output = samp_output.cpu()

            samples[epoch + 1] = samp_output.data.numpy()[:sample_size]
            if output_dir:
                LOGGER.info('Saving samples...')
                save_samples(samples[epoch + 1], epoch + 1,
                             output_dir / "Audio")

        #save model
        model_epoch_output_path = model_output_path / f"{epoch+1}"
        model_epoch_output_path.mkdir(parents=True, exist_ok=True)
        torch.save(model_gen.state_dict(),
                   model_epoch_output_path / f"Gen.pkl",
                   pickle_protocol=pk.HIGHEST_PROTOCOL)
        torch.save(model_dis.state_dict(),
                   model_epoch_output_path / f"Disc.pkl",
                   pickle_protocol=pk.HIGHEST_PROTOCOL)
        torch.save(model_Q.state_dict(),
                   model_epoch_output_path / f"Q.pkl",
                   pickle_protocol=pk.HIGHEST_PROTOCOL)

    ## Get final discriminator loss
    # Get noise
    noise = torch.Tensor(batch_size, latent_dim).uniform_(-1, 1)
    if use_cuda:
        noise = noise.cuda()
    noise_v = autograd.Variable(noise,
                                volatile=True)  # totally freeze model_gen

    # Get new batch of real training data
    D_cost_test, D_wass_test = compute_discr_loss_terms(model_dis,
                                                        model_gen,
                                                        test_data_v,
                                                        noise_v,
                                                        batch_size,
                                                        latent_dim,
                                                        lmbda,
                                                        use_cuda,
                                                        compute_grads=False)

    D_cost_valid, D_wass_valid = compute_discr_loss_terms(model_dis,
                                                          model_gen,
                                                          valid_data_v,
                                                          noise_v,
                                                          batch_size,
                                                          latent_dim,
                                                          lmbda,
                                                          use_cuda,
                                                          compute_grads=False)

    if use_cuda:
        D_cost_test = D_cost_test.cpu()
        D_cost_valid = D_cost_valid.cpu()
        D_wass_test = D_wass_test.cpu()
        D_wass_valid = D_wass_valid.cpu()

    final_discr_metrics = {
        'cost_validation': D_cost_valid.data.numpy()[0],
        'wasserstein_cost_validation': D_wass_valid.data.numpy()[0],
        'cost_test': D_cost_test.data.numpy()[0],
        'wasserstein_cost_test': D_wass_test.data.numpy()[0],
    }

    return model_gen, model_dis, model_Q, history, final_discr_metrics, samples
