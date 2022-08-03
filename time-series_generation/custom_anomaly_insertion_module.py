from typing import List, Optional
import numpy as np
import xarray as xr


from pywatts.modules.generation.anomaly_generation_module import AnomalyGeneration

class CustomAnomalyInsertion(AnomalyGeneration):
    """
    Module to define more individual anomalies to be inserted
    """

    def _anomaly_type1(self, target, indices, lengths):
        """
        Anomaly (type 1) that drops the time series values to a negative value and following zero values before adding
        the missed sum to the end of the anomaly
        """
        mean = target.mean()
        std = np.std(target.values)
        for idx, length in zip(indices, lengths):
            if length <= 1:
                raise Exception("Type 1 anomalies must be longer than 1.")
            else:
                random_limit = 2 + np.random.rand() * 3
                accumulated_values = target[idx:idx + length - 1].sum()
                target[idx:idx + length - 1] = 0
                # Replace first value by a negative value
                target[idx] = -1 * (mean + random_limit * std)
                target[idx + length - 1] += accumulated_values
        return target

    def _anomaly_type2(self, target, indices, lengths):
        """
        Anomaly (type 2) that drops the time series values to zero and adds the missed sum to the end of the anomaly
        """
        for idx, length in zip(indices, lengths):
            if length <= 1:
                raise Exception("Type 2 anomalies must be longer than 1.")
            else:
                accumulated_values = target[idx:idx + length - 1].sum()
                target[idx:idx + length - 1] = 0
                target[idx + length - 1] += accumulated_values
        return target


    def _anomaly_type3(self, target, indices, lengths, limits):
        """
        Anomaly (type 3) is an outlier with a negative sign
        """
        return self._outlier(target=target, indices=indices, lengths=lengths, outlier_sign="negative", limits=limits)

    def _anomaly_type4(self, target, indices, lengths, limits):
        """
        Anomaly (type 4) is an outlier with a positive sign
        """
        return self._outlier(target=target, indices=indices, lengths=lengths, outlier_sign="positive", limits=limits)


    def _outlier(self, target: xr.DataArray, indices: List, lengths: List, outlier_sign: Optional[str] = None,
                         random: bool = True, limits: List = [2, 4]):
        """
        Insert anomalies that randomly multiply and, if desired, negate the target values.

        :param target: Array into which anomalies should be inserted.
        :type target: xr.DataArray
        :param indices: List of positions where the anomalies should start.
        :type indices: List
        :param lengths: List of lengths of the anomalies and their positions (indices).
        :type lengths: List
        :param outlier_sign: Sign of the outliers to have only 'positive' or 'negative' outliers.
                             If None, randomly choose positive or negative sign.
        :type outlier_sign: Optional[str]
        :param random: If True, shuffle new random value for every anomaly (default True).
        :type random: bool
        :param limits: Upper and lower limits of the random value (default [2, 4]).
        :type limits: List
        :raises WrongParameterException: If type of outlier is unknown.
        :return: Data with newly inserted anomalies.
        :rtype: xr.DataArray
        """
        # get target data stats needed for outlier setting
        mean = target.mean()
        std = np.std(target.values)
        random_limit = lambda: limits[0] + np.random.rand() * limits[1]


        # outlier definition depending only on mean (Class 3 anomaly from Moritz Weber's thesis)
        high_outlier = lambda x: mean * random_limit()
        low_outlier = lambda x: -1 * mean * random_limit()
        

        # check whether a new outlier should be calculated for each idx
        # or whether there should be a 'global' outlier variable
        if not random:
            high = high_outlier(target[indices[0]])
            high_outlier = lambda x: high
            low = low_outlier(target[indices[0]])
            low_outlier = lambda x: low

        # check whether only negative or positive outlier should be used
        if outlier_sign == 'negative':
            high_outlier = low_outlier
        elif outlier_sign == 'positive':
            low_outlier = high_outlier

        # generate outlier
        for idx, length in zip(indices, lengths):
            # NOTE: Outlier with length > 1 will be constant over the interval
            #       because of target[idx:idx + length] = single_value.
            #       Solution: length = 1 and more anomalies to generate
            if np.random.rand() > 0.5:
                target[idx:idx + length] = high_outlier(target[idx])
            else:
                target[idx:idx + length] = low_outlier(target[idx])

        return target
