import pickle
from abc import ABC
from typing import Dict

import numpy as np
import torch
import xarray as xr
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from torch import distributions

from generative_models.generator_base import GeneratorBase
from generative_models.inn_base_functions import AnomalyINN


class INNWrapperBase(GeneratorBase, ABC):

    def __init__(self, name: str = "INN", logprob=False, **kwargs):
        super().__init__(name, **kwargs)
        self.logprob = logprob


    def get_params(self) -> Dict[str, object]:

        return {
            "epochs": self.epoch,
            "horizon": self.horizon,
            "cond_features": self.cond_features,
        }

    def set_params(self, epochs=None, horizon=None, cond_features=None):
        if epochs is not None:
            self.epoch = epochs
        if horizon is not None:
            self.horizon = horizon
        if cond_features is not None:
            self.cond_features = cond_features

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
        """
        json_module = super().save(fm)
        path = fm.get_path(f"module_{self.name}.pickle")
        with open(path, 'wb') as outfile:
            pickle.dump(self.generator, outfile)
        json_module["module"] = path
        return json_module

    def _transform(self, input_data: xr.DataArray, logprob=False, reverse=False,
                   **kwargs: xr.DataArray) -> np.array:
        x = input_data.values
        conds = self._get_conditions(kwargs)
        if logprob:
            z, jac = self.generator.forward(
                torch.from_numpy(x.astype("float32")),
                torch.from_numpy(conds.astype("float32")), rev=reverse)
            prior = distributions.MultivariateNormal(torch.zeros(96), torch.eye(96))
            log_z = prior.log_prob(z)
            log_jac = jac.log()
            return (log_z + log_jac).detach().numpy()

        return self.generator.forward(torch.from_numpy(x.astype("float32")),
                                      torch.from_numpy(conds.astype("float32")),
                                      rev=reverse)[0].detach().numpy()

    def get_generator(self, x_features, cond_features):
        return AnomalyINN(5e-4, horizon=x_features, cond_features=cond_features, n_layers_cond=10)


class INNWrapper(INNWrapperBase):

    def loss_function(self, z, log_j):
        if self.supervised:
            loss = torch.mean(z ** 2) / 2 - torch.mean(log_j) / z.shape[-1]
        else:
            loss = torch.mean(z ** 2, dim=-1) / 2 - torch.mean(log_j, dim=-1) / z.shape[-1]
            q = torch.quantile(loss, self.contamination)
            loss = loss[loss < q]
            loss = loss.mean()
        return loss

    def _run_epoch(self, data_loader, epoch, conds_val, x_val):
        self.generator.train()
        for batch_idx, (data, conds) in enumerate(data_loader):

            z, log_j = self.generator(data, conds)

            loss = self.loss_function(z, log_j)

            self._apply_backprop(loss)

            if not batch_idx % 50:
                with torch.no_grad():
                    z, log_j = self.generator(torch.from_numpy(x_val.astype("float32")),
                                              torch.from_numpy(conds_val.astype("float32")))
                    loss_test = self.loss_function(z, log_j)
                    print(f"{epoch}, {batch_idx}, {len(data_loader.dataset)}, {loss.item()}, {loss_test.item()}")

    def transform(self, input_data: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        result = self._transform(input_data=input_data, logprob=self.logprob, **kwargs)
        return numpy_to_xarray(result, input_data)
