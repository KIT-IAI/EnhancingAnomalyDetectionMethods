from abc import ABC
from typing import Dict

import numpy as np
import torch
import xarray as xr
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from torch import nn

from anomalyINN import GeneratorBase


class CVAE(nn.Module):
    def __init__(self, input_size, latent_size, condition_size, num_layers):
        super(CVAE, self).__init__()

        encoded_cond_dim = 4
        self.cond_encoding = nn.Sequential(nn.Linear(condition_size,  8),
                                       nn.Tanh(),
                                       nn.Linear(8, encoded_cond_dim),
                                       )

        # encode
        l_size = input_size + encoded_cond_dim
        encode_layers = []
        for i in range(num_layers, 1, -1):
            encode_layers.append(
                nn.Linear(l_size, latent_size * 2**(i-1))
            )
            l_size = latent_size * 2**(i-1)
        self.fc21 = nn.Linear(l_size, latent_size)
        self.fc22 = nn.Linear(l_size, latent_size)


        l_size = latent_size + encoded_cond_dim
        decode_layers = []
        for i in range(1, num_layers):
            decode_layers.append(
                nn.Linear(l_size, latent_size * 2**(i))
            )
            l_size = latent_size * 2**(i)
        decode_layers.append(
                nn.Linear(l_size, input_size)
        )
        self.encode_layers = nn.ModuleList(encode_layers)
        self.decode_layers = nn.ModuleList(decode_layers)

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.trainable_parameters = self.parameters()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        c = c.reshape(len(c), -1)
        inputs = torch.cat([x, self.cond_encoding(c)], 1) # (bs, feature_size+class_size)
        h = inputs
        for layer in self.encode_layers:
            h = self.tanh(layer(h))

        z_mu = self.fc21(h)
        z_var = self.fc22(h)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        c = c.reshape(len(c), -1)
        inputs = torch.cat([z, self.cond_encoding(c)], 1)
        h = inputs
        for layer in self.decode_layers[:-1]:
            h = self.tanh(layer(h))
        return self.decode_layers[-1](h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


class cVAEModule(GeneratorBase, ABC):

    def __init__(self, name: str = "VAE", bottleneck_size=16, num_layers=3, **kwargs):
        super().__init__(name, **kwargs)
        self.num_layers= num_layers
        self.bottleneck_size = bottleneck_size


    def get_params(self) -> Dict[str, object]:

        return {
            "epochs": self.epochs,
            "num_layers": self.num_layers,
            "bottleneck_size": self.bottleneck_size
        }

    def set_params(self, epochs=None, num_layers=None, bottleneck_size=None):
        if epochs is not None:
            self.epoch = epochs
        if num_layers is not None:
            self.num_layers = num_layers
        if bottleneck_size is not None:
            self.bottleneck_size = bottleneck_size

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
        """
        json_module = super().save(fm)
        return json_module


    def _transform(self, input_data: xr.DataArray, get_log_prob=False, reverse=False,
                   **kwargs: xr.DataArray) -> np.array:
        x = input_data.values
        conds = self._get_conditions(kwargs)

        if reverse:
            return self.generator.decode(torch.from_numpy(x.astype("float32")), torch.from_numpy(conds.astype("float32"))).detach().numpy()
        else:
            return self.generator.encode(torch.from_numpy(x.astype("float32")), torch.from_numpy(conds.astype("float32")))[0].detach().numpy()

    def get_generator(self, x_features, cond_features):
        return CVAE(x_features, self.bottleneck_size, cond_features, num_layers=self.num_layers)

    def _run_epoch(self, data_loader, epoch, cond_val, x_val):
        self.generator.train()
        for batch_idx, (data, conds) in enumerate(data_loader):
            recon_batch, mu, logvar = self.generator(data, conds)
            loss = self.loss_function(recon_batch, data, mu, logvar)[0]
            self._apply_backprop(loss)

            if not batch_idx % 50:
                with torch.no_grad():
                    recon_batch, mu, logvar = self.generator(torch.from_numpy(x_val.astype("float32")),
                                                        torch.from_numpy(cond_val.astype("float32")))
                    loss_test = self.loss_function(recon_batch, x_val, mu, logvar)
                    print(f"{epoch}, {batch_idx}, {len(data_loader.dataset)}, {loss.item()}, {loss_test[0].item()}, {loss_test[1].item()}, {loss_test[2].item()}")

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        if self.supervised:
            mse = torch.mean(torch.square(recon_x - x))
            kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
            return mse + kld, mse, kld
        else:
            mse = torch.mean(torch.square(recon_x - x), dim=-1)
            kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)
            loss = mse + kld
            q = torch.quantile(loss, self.contamination)
            return loss[loss < q].mean(), mse[loss < q].mean(), kld[loss < q].mean()


    def transform(self, input_data: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> Dict[str, xr.DataArray]:
        result  = self._transform(input_data=input_data, **kwargs)
        res = {
            "latent_space": numpy_to_xarray(result, input_data),
        }
        return res

