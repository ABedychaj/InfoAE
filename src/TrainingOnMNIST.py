import os
import time

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.utils import save_image

from src.Decoders.Decoder import FCDecoder
from src.Encoders.Encoder import FCEncoder


def to_img(x, normalized):
    if normalized:
        x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x


def save_images_from_epoch(train_imgs, test_imgs, epoch, normalize_img, folder, save_every):
    if (epoch + 1) % save_every == 0:
        save_train = to_img(train_imgs.cpu().data, normalize_img)
        save_test = to_img(test_imgs.cpu().data, normalize_img)
        save_image(save_train,
                   os.path.join(os.path.join(folder, 'images'), 'train_image_{}.png'.format(epoch)))
        save_image(save_test,
                   os.path.join(os.path.join(folder, 'images'), 'test_image_{}.png'.format(epoch)))


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
latent_dim = [5, 10]
hidden_dim = [1, 2, 3]
neurons = 64
num_epochs = 50
lr = 1E-5
batch_size = 32
criterion = nn.MSELoss()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5)),
                                ])

train_dataset = MNIST("../data", transform=transform, download=True, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for _ in range(10):
    for l in latent_dim:
        for h in hidden_dim:
            encoder_activations = [nn.Sequential(nn.ReLU()) for _ in range(h)]
            encoder_activations.extend([nn.ReLU()])

            encoder = FCEncoder(
                input_dim=28 * 28,
                latent_dim=l,
                activations=encoder_activations,
                hidden_dims=[neurons for _ in range(h)]
            )

            decoder = FCDecoder(
                latent_dim=l,
                activations=[nn.Sequential(nn.ReLU()) for _ in range(h + 1)],
                hidden_dims=[neurons for _ in range(h)],
                output_dim=28 * 28
            )

            model = AutoEncoder(nn.Sequential(View((-1, 784)), encoder), nn.Sequential(decoder, View((-1, 1, 28, 28))))
            optimizer = optim.Adam(model.parameters(), lr)

            print("Beginning training...")
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0
                start = time.time()

                for batch_idx, (data, _) in enumerate(train_dataloader):
                    optimizer.zero_grad()
                    data = data.float()
                    recon, encoded = model(data)

                    loss = criterion(recon, data)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.data

                end = time.time()
                loss = train_loss / len(train_dataloader)

                if epoch % 2 == 0:
                    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, num_epochs, loss))
                    print("Time of epoch {}: {}".format(epoch + 1, end - start))

                if epoch == num_epochs - 1:
                    results["latent_size"].extend([l])
                    results["hidden_dims"].extend([h])
                    results["mse"].extend([float(loss.data)])

                save_images_from_epoch(train_imgs=recon,
                                       test_imgs=encoded,
                                       epoch=epoch,
                                       normalize_img=True,
                                       folder="../",
                                       save_every=10
                                       )

            print("Training complete")

# pd.DataFrame.from_dict(results).to_csv("./results.csv")
