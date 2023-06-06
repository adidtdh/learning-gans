import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


"""
    Discriminator Class calculates the "fakeness" of the output of the Generator

    __init__
    :param in_features  : the number of inputs from the MNIST dataset or the image dimetnions

    forward calculates the layers
    :param x            : Output of Generator
    :return             : Returns value between 0 and 1
"""
class Discriminator(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.1), # leaky relu is usually the go to in GAN networks. You can play with the parameter.
            nn.Linear(128,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

"""
    Generator generates images

    __init__
    :param noise_dims   : the dimentions of the noise image fed to the Generator
    :param img_dims     : the flattened dimentions of the images fed to the generator( also the output size of the geneartor)

    forward
    :param x            : random generated image of noise
    :return             : generated image
"""
class Generator(nn.Module):
    def __init__(self, noise_dims, img_dims) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(noise_dims, 256),
            nn.LeakyReLU(.1),
            nn.Linear(256, img_dims), 
            nn.Tanh(), # output is between -1 and 1 to match normalized input data
        )

    def forward(self, x):
        return self.layers(x)


# Hyperparamerters. These are very sensitive and will casue random stuff to happen if changed or so I was told.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # best learning rate for adam apperently
noise_dims = 64 # try 128, 256
image_dim = 28 * 28 * 1 # image dimentions of images in MNIST dataset is 28 x 28 x 1
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(noise_dims, image_dim)
fixed_noise = torch.randn((batch_size, noise_dims)).to(device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))] # .1307 and .3081 are the mean and stddev of the mnist dataset
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# optimizers: 2 optimizers are required for both the generator and discrimiator models
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake") # writer for tensorboard fake images
writer_real = SummaryWriter(f"runs/GAN_MNIST/real") # writer for tensorboard real images
step = 0

for epoch in range(num_epochs):
    for batch_index, (real, _) in enumerate(loader): # labels are not required because this is unsupervised learning :0
        real = real.view(-1, image_dim) # keep number of images in batch and flatten images
        # real is the real images
        # fake is the generated images
        batch_size = real.shape[0]

        # Training Discriminator
        noise = torch.randn(batch_size, noise_dims).to(device)
        fake = gen(noise)

        ### Loss is max log(D(real)) + log(1-D(G(z))

        # this is the log(D(real)) part
        disc_real = disc(real).view(-1) # flattens everything
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) # loss of disc_real in comparison to 

        # this is the log(1-D(G(z))) part
        disc_fake = disc(fake.detach()).view(-1) # detach so it can be used for training too
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # this combines the two
        lossD = (lossD_fake + lossD_real)/2

        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator

        ### Loss is min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_index == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_index}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            ) 

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

