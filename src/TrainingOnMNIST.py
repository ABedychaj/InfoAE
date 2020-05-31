import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from src.Decoders.Decoder import FCDecoder
from src.Encoders.Encoder import FCEncoder
from src.Loss.Loss import MutualInformationLoss


def save_images_from_epoch(imgs, epoch, folder, save_every, train=True):
    if (epoch + 1) % save_every == 0:
        if train is True:
            name = 'train_image_{}.png'
        else:
            name = 'test_image_{}.png'
        save_train = imgs.cpu().data.clamp(0, 1)
        save_image(save_train,
                   os.path.join(os.path.join(folder, 'images'), name.format(epoch)))


def report(epoch_, num_epochs_, start_, end_, loss_, save_every=2, train=True):
    if epoch_ % save_every == 0:
        if train:
            print("Training...")
        else:
            print("Testing...")
        print("epoch : {}/{}, loss = {:.6f}".format(epoch_ + 1, num_epochs_, loss_))
        print("Time of epoch {}: {}".format(epoch_ + 1, end_ - start_))


class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        if isinstance(x, tuple):
            return (x[0].view(*self.shape), *x[1:])
        return x.view(*self.shape)


results = {
    "latent_size": [],
    "hidden_dims": [],
    "mse": []
}

# parameters
latent_dim = [8]
hidden_dim = [1]
neurons = 64
num_epochs = 51
lr = 1E-3
batch_size = 256
beta = 2
criterion = nn.MSELoss()
MI_loss = MutualInformationLoss()

train_dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                              download=True,
                                                              train=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.1307,), (0.3081,))
                                                              ])),
                                               batch_size=batch_size,
                                               shuffle=True)

test_dataloader = torch.utils.data.DataLoader(datasets.MNIST('../data',
                                                             download=True,
                                                             train=False,
                                                             transform=transforms.Compose([
                                                                 transforms.ToTensor(),
                                                                 transforms.Normalize((0.1307,), (0.3081,))
                                                             ])),
                                              batch_size=batch_size,
                                              shuffle=False)

for _ in range(1):
    for l in latent_dim:
        for h in hidden_dim:
            # hidden_dims=[neurons // (_ + 1) for _ in range(h)]
            hidden_dims_e = [neurons for _ in range(h)]
            encoder_activations = [nn.Sequential(nn.ReLU()) for _ in hidden_dims_e]
            encoder_activations.extend([nn.Sequential(nn.ReLU())])

            encoder = FCEncoder(
                input_dim=28 * 28,
                latent_dim=l,
                activations=encoder_activations,
                hidden_dims=hidden_dims_e
            )

            encoder = nn.Sequential(View((-1, 784)), encoder)

            # hidden_dims=[neurons // _ for _ in range(h, 0, -1)],
            hidden_dim_d = hidden_dims_e[::-1]
            decoder_activations = [nn.Sequential(nn.ReLU()) for _ in hidden_dim_d]
            decoder_activations.extend([nn.Sequential(nn.ReLU())])  # dim of output

            decoder = FCDecoder(
                latent_dim=l,
                activations=decoder_activations,
                hidden_dims=hidden_dim_d,
                output_dim=28 * 28
            )

            decoder = nn.Sequential(decoder, View((-1, 1, 28, 28)))

            optimizer_E = optim.Adam(encoder.parameters(), lr)
            optimizer_D = optim.Adam(decoder.parameters(), lr)

            print("Beginning training...")
            print("Encoder")
            print(encoder)
            print("Decoder")
            print(decoder)

            for epoch in range(num_epochs):
                encoder.train()
                decoder.train()
                train_loss = 0
                start = time.time()

                for batch_idx, (data, target) in enumerate(train_dataloader):
                    optimizer_E.zero_grad()
                    optimizer_D.zero_grad()

                    encoded_latent = encoder(data)
                    recon = decoder(encoded_latent)

                    loss = criterion(recon, data) + beta * MI_loss.loss(recon.detach(), data.detach())
                    loss.backward()

                    optimizer_E.step()
                    optimizer_D.step()

                    train_loss += loss.data

                end = time.time()
                loss = train_loss / len(train_dataloader)

                report(epoch, num_epochs, start, end, loss, save_every=2, train=True)
                save_images_from_epoch(imgs=recon,
                                       epoch=epoch,
                                       folder="../",
                                       save_every=2)

                test_loss = 0
                start_t = time.time()
                encoder.eval()
                decoder.eval()

                for batch_idx, (data_t, target_t) in enumerate(test_dataloader):
                    encoded_latent = encoder(data_t)
                    recon = decoder(encoded_latent)

                    loss = criterion(recon, data_t) + beta * MI_loss.loss(recon.detach(), data_t.detach())

                    test_loss += loss.data

                end_t = time.time()

                loss_t = test_loss / len(test_dataloader)

                report(epoch, num_epochs, start_t, end_t, loss_t, save_every=2, train=False)
                save_images_from_epoch(imgs=recon,
                                       epoch=epoch,
                                       folder="../",
                                       save_every=2,
                                       train=False)

            print("Training complete")
