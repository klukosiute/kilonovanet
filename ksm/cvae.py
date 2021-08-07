import torch
import torch.nn as nn


class CVAE(nn.Module):
    """
        Base pytorch cVAE class
    """

    def __init__(self, image_size=1629, hidden_dim=500, z_dim=20, c=4):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space
        :param c: Dimension of conditioning variables
        """
        super(CVAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(image_size, hidden_dim, z_dim, c)
        self.decoder = Decoder(image_size, hidden_dim, z_dim, c)

    def forward(self, x, y):
        """
        Compute one single pass through decoder and encoder

        :param x: Conditioning variables corresponding to images/spectra
        :param y: Images/spectra
        :return: Mean returned by decoder, mean returned by encoder, log variance returned by encoder
        """
        y = torch.cat((y, x), dim=1)
        mean, logvar = self.encoder(y)

        # re-parametrize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        sample = mean + eps * std

        z = torch.cat((sample, x), dim=1)
        mean_dec = self.decoder(z)

        return mean_dec, mean, logvar


class Encoder(nn.Module):
    """
    Encoder of the cVAE
    """

    def __init__(self, image_size=1629, hidden_dim=500, z_dim=20, c=4):
        """
        :param image_size: Size of 1D "images" of data set i.e. spectrum size
        :param hidden_dim: Dimension of hidden layer
        :param z_dim: Dimension of latent space
        :param c: Dimension of conditioning variables

        """
        super().__init__()

        self.layers_mu = nn.Sequential(
            nn.Linear(image_size + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.layers_logvar = nn.Sequential(
            nn.Linear(image_size + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, x):
        """
        Compute single pass through the encoder

        :param x: Concatenated images and corresponding conditioning variables
        :return: Mean and log variance of the encoder's distribution
        """
        mean = self.layers_mu(x)
        logvar = self.layers_logvar(x)
        return mean, logvar


class Decoder(nn.Module):
    """
    Decoder of cVAE
    """

    def __init__(self, image_size=1629, hidden_dim=500, z_dim=20, c=4):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim + c, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, image_size),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Compute single pass through the decoder

        :param z: Concatenated sample of hidden variables and the originally inputted conditioning variables
        :return: Mean of decoder's distirbution
        """
        mean = self.layers(z)
        return mean
