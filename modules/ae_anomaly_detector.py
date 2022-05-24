import tensorflow as tf
from pywatts.core.base import BaseEstimator
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from typing import Dict
import xarray as xr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K

class KLDivergenceLayer(tf.keras.layers.Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def get_ae(input_dim):
    input_data = Input(shape=(input_dim,))
    encoded = Dense(12, activation='tanh')(input_data)
    latent = Dense(6, activation='tanh')(encoded)
    decoded = Dense(12, activation='tanh')(latent)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencode = Model(inputs=input_data, outputs=decoded)
    autoencode.compile(optimizer='adam', loss='mse')

    return autoencode

def get_vae(input_dim):
    input_data = Input(shape=(input_dim,))
    encoded = Dense(12, activation='tanh')(input_data)
    mu_z = tf.keras.layers.Dense(6)(encoded)
    sigma_z = tf.keras.layers.Dense(6)(encoded)
    mu_z, sigma_z = KLDivergenceLayer()([mu_z, sigma_z])
    eps = tf.random.normal(shape=(1, 6), mean=0, stddev=1)
    latent = tf.keras.layers.Multiply()([eps, tf.math.exp(tf.math.divide(sigma_z, 2))])
    decoded = Dense(12, activation='tanh')(latent)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencode = Model(inputs=input_data, outputs=decoded)
    autoencode.compile(optimizer='adam', loss='mse')

    return autoencode

class AutoencoderDetection(BaseEstimator):

    def __init__(self, name: str = "ForecastAnomalyDetection", threshold=0.95, method="ae"):
        super().__init__(name)
        self.threshold = threshold
        self.method = method

    def fit(self, x: xr.DataArray):
        # Split into input and output variables
        if self.method == "ae":
            self.ae = get_ae(input_dim=x.shape[-1])
        elif self.method == "vae":
            self.ae = get_vae(input_dim=x.shape[-1])
        data_train = x.values
        # Fit model
        self.ae.fit(data_train, data_train, epochs=500, validation_split=0.2,
                    callbacks=[EarlyStopping(patience=10)],
                    batch_size=128, shuffle=True)
        self.is_fitted = True

    def transform(self, x: xr.DataArray):
        # Split into input and output variables

        # Predict output variables based on input

        # Evaluate prediction quality

        # Use threshold to decide if it is an outlier or not.

        pr = self.ae.predict(x.values)
        score_ae = np.sqrt(np.mean((pr - x.values) ** 2, axis=1))
        quantile = np.quantile(score_ae, self.threshold)
        result = np.ones(score_ae.shape)
        result[score_ae > quantile] = -1

        return numpy_to_xarray(result.reshape((-1, 1)), x)

    def get_params(self) -> Dict[str, object]:
        return super().get_params()

    def set_params(self, **kwargs):
        return super().set_params(**kwargs)
