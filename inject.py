#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copy from tsad-model-selection-master/src/tsadams/model-selection
# Modification: (1)AnomalyBERT (2)Perturbation in Frequency Domain

import numpy as np
import inspect
from typing import Union

from numpy.lib.function_base import average


def _moving_average(x: np.ndarray, 
                    w: int =2):
    """Function to move average over a window.
        
    Parameters
    ----------
    x: np.ndarray [n_times, n_features]
        Time series for moving average.
    
    w: int
        Window of moving average.
    """
    # padding:keep length same
    padding_size = (w - 1) // 2
    padded = np.pad(x, ((padding_size, w - 1 - padding_size), (0, 0)), mode='edge')

    # weight kernel
    window = np.ones(w) / w

    # moving average along time
    averaged = np.apply_along_axis(lambda m: np.convolve(m, window, mode='valid'), axis=0, arr=padded)

    return averaged
'''
def _perturbation_spectrum(s: np.ndarray, 
                           r: float = 0.1, 
                           K: int = 3, 
                           r_amp: float = 1.5, 
                           theta: float = 1):
    """Function used for perturbing in the frequency domain to generate anomalies.
        
    Parameters
    ----------
    s: numpy.ndarray [n_times_s, n_cols]
        
    r: float
        Proportion of individual anomalous segments to the entire sequence
        
    K: int
        The number of abnormal frequency segments
        
    r_map: 
        Multiplier: The variance of amplitude noise, relative to the multiple of the variance of the original amplitude noise of the entire sequence
        
    theta:
       The variance of Gaussian noise added in phase/angle.
    """

    # In order to maintain conjugate symmetry during disturbances, rfft and irfft are used here
    s_fft = np.fft.rfft(s, axis=0)
    n_times_s, n_cols = s_fft.shape

    # Extracting fragments, in order to simplify the code, overlapping fragments were not excluded. Papers require that the overlap between fragments should not exceed half of the length.
    segment_length = max(int(r * n_times_s), 1)
    indices = self.rng.choice(np.arange(n_times_s - segment_length + 1), size=min(K, n_times_s - segment_length + 1), replace=False)
    all_segments = np.sort(np.unique(np.concatenate([np.arange(start, start+segment_length) for start in indices])))

    # Calculate amplitude, angle/phase
    s_angle = np.angle(s_fft)
    s_amplitude = np.abs(s_fft)
    mean_amplitude = s_amplitude.mean(axis=0, keepdims=True)
    std_amplitude = s_amplitude.std(axis=0, keepdims=True)
    # Disturbance of amplitude and angle separately
    s_amplitude[all_segments, :] = mean_amplitude + r_amp * self.rng.randn(len(all_segments), n_cols) * std_amplitude 
    s_angle[all_segments, :] = s_angle[all_segments, :] + self.rng.randn(len(all_segments), n_cols) * theta

    # Calculating complex numerical representations from amplitude and angle
    s_fft_perturbation = s_amplitude * np.exp(1j * s_angle)
    # IFFT
    s_perturbation = np.fft.irfft(s_fft_perturbation, s.shape[0], axis=0)

    return s_perturbation
'''

class InjectAnomalies:

    def __init__(self,
                 ano_col_rate: float = 0.3,
                 ano_time_rate_max: float = 1,
                 ano_time_rate_min: float = 0.5,
                 random_state: int = 0,
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
        random_state: int
            Random state
        
        verbose: bool
            Controls verbosity
        """
        self.random_state = random_state
        self.rng = np.random.RandomState(self.random_state) # 只设置了np随机种子

        self.ano_col_rate = ano_col_rate
        # 按维度添加异常的比例
        self.ano_time_rate_max = ano_time_rate_max
        self.ano_time_rate_min = ano_time_rate_min
        self.verbose = verbose
        
        if self.verbose:
            print(f'Max rate of a anomalous segment to entire window:{self.ano_time_rate_max}')
            print(f'Proportion of anomalous columns: {self.ano_col_rate}')

    def __str__(self):
        InjectAnomaliesObject = {
            'random_state': self.random_state,
            'verbosity': self.verbose, 
            'ano_col_rate': self.ano_col_rate,
            'ano_time_rate_max': self.ano_time_rate_max,    
            'ano_time_rate_min': self.ano_time_rate_min,   
        }
        return f'InjectAnomaliesObject: {InjectAnomaliesObject}'


    def set_random_anomaly_parameters(self):
        noise_std = self.rng.choice([0.05, 0.1, 0.15, 0.2])
        specturm_amplitude_scale = self.rng.choice([0.25, 0.3, 0.5, 2, 3, 4])
        spectrum_angle_std = self.rng.choice([0.1, 0.3, 0.4, 0.5, 0.7, 0.9, 1, 1.2, 1.5])
        flip_direction = self.rng.choice(['horizontal', 'vertical', 'both'])
        length_adjusting_scale = self.rng.randint(2, 5)
        amplitude_scale = self.rng.choice([0.25, 0.3, 0.5, 2, 3, 4])
        soft_rep_direction = self.rng.choice(['horizontal', 'vertical', 'both', 'none'])
        soft_rep_weight = self.rng.choice([0.5, 0.6, 0.7, 0.75])
        ma_window = self.rng.choice([2, 3, 4, 5])
        constant_type = self.rng.choice(
            ['quantile', 'noisy_0', 'noisy_1', '0', '1'])
        constant_quantile = self.rng.choice([0.1, 0.25, 0.5, 0.7, 0.9])
        baseline = self.rng.choice([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3])
        contextual_scale = self.rng.choice([0.5, 1, 1.5, 2])

        params = {"noise_std":noise_std, 
                  "specturm_amplitude_scale":specturm_amplitude_scale, "spectrum_angle_std":spectrum_angle_std, "flip_direction":flip_direction, "length_adjusting_scale":length_adjusting_scale, "amplitude_scale":amplitude_scale, "soft_rep_direction":soft_rep_direction, "soft_rep_weight":soft_rep_weight, 
                  "ma_window":ma_window, 
                  "constant_type":constant_type, "constant_quantile":constant_quantile, "baseline":baseline, 
                  "contextual_scale":contextual_scale}
        return params
        
    def replacing_weights(self, 
                          interval_len, 
                          replacing_weight):
        """Function to generate weight list for soft replacement.
        """                 
        warmup_len = interval_len // 10
        return np.concatenate((np.linspace(0, replacing_weight, num=warmup_len),
                               np.full(interval_len - 2 * warmup_len, replacing_weight),
                               np.linspace(replacing_weight, 0, num=warmup_len)), axis=None)
    
    def _gen_label(self, 
                   T: np.ndarray,
                   ano_start: int,
                   ano_end: int):
        """Function to generate anomaly label for methods except peaks.
        """
        n_times = T.shape[0]
        ano_labels = np.zeros(n_times)
        ano_labels[ano_start:ano_end] = 1

        return ano_labels



    def inject_anomalies(self,
                         T: np.ndarray,
                         cat_cols: np.ndarray = np.array([]),
                         anomaly_type: str = 'flip', 
                         use_global_config: bool = False,
                         **kwargs
                         ):
        """Function to inject different kinds of anomalies.
        
        Parameters
        ----------
        T: numpy.ndarray [n_times, n_features]
            Timeseries for injecting anomalies.
            In the following functions, the explanations for (T, num_cols, cat_cols, ano_num_cols, ano_cat_cols, ano_start, ano_end, ano_times) are the same, so they are not repeated.
        
        cat_cols:
            Categorical columns in the data
        
        anomaly_type: str
            Type of anomaly to inject. Choices=['random_and_batch_fixed', 'random', 'peaks', 'timenoise', 'spectralnoise', 'flip', 'length_adjusting', 'amplitude_scaling', 'soft_replacing', 'uniform_replacing', 'average', 'cutoff', 'wander', 'contextual'] 

        use_global_config: bool
            Whether to randomly initialize anomaly injection parameters
        """
        n_times, n_features = T.shape
        timeseries_with_anomalies = T.copy()

        num_cols = np.array([i for i in range(n_features) if i not in cat_cols])

        # Randomly extract features and inject anomalies onto these features

        # Extract numeric columns to inject exceptions
        ano_num_cols = self.rng.uniform(0., 1., size=len(num_cols))
        ano_num_cols = ano_num_cols - np.max([ano_num_cols.min(), self.ano_col_rate]) <= 0.001 
        # 比右侧大的都设为false
        # To reduce the situation where no columns are modified

        ano_num_cols = num_cols[ano_num_cols]

        if cat_cols.size > 0:
            ano_cat_cols = self.rng.uniform(0., 1., size=len(cat_cols))
            ano_cat_cols = ano_cat_cols - np.max([ano_cat_cols.min(), self.ano_col_rate]) <= 0.001
            ano_cat_cols = cat_cols[ano_cat_cols]
        else:
            cat_cols = []
            ano_cat_cols = []
        
        # Randomly extract the length and start index of anomalies
        ano_times = self.rng.randint(int(n_times * self.ano_time_rate_min), int(n_times * self.ano_time_rate_max))
        ano_start = self.rng.randint(0, n_times - ano_times)
        ano_end = ano_start + ano_times

        if self.verbose:
            print(f'Anomalous numerical cols: {ano_num_cols}')
            print(f'Anomalous categorical cols: {ano_cat_cols}')
            print(f'Anomalous interval: [{ano_start}, {ano_end}) with length={ano_times}')
            print(f'Anomaly type: {anomaly_type}')

        func = getattr(self, anomaly_type, None)
        if func is not None and callable(func):
            # Get function parameters
            params = inspect.signature(func).parameters
            # Only pass parameters that exist in function parameters to the corresponding function
            kwargs['ano_num_cols'] = ano_num_cols
            kwargs['ano_cat_cols'] = ano_cat_cols
            kwargs['ano_times'] = ano_times
            kwargs['ano_start'] = ano_start
            kwargs['ano_end'] = ano_end
            kwargs['num_cols'] = num_cols
            kwargs['cat_cols'] = cat_cols
            if use_global_config:
                params_random = self.set_random_anomaly_parameters()
                kwargs.update(params_random)
            func_args = {k: v for k, v in kwargs.items() if k in params}
            return func(T=timeseries_with_anomalies, 
                        **func_args)
        else:
            raise AttributeError(f"{anomaly_type} not found or not callable in class {self.__class__.__name__}")
    
    def peaks(self, 
              T: np.ndarray, 

              ano_num_cols: np.ndarray,

              peaks_num: int = 5):
        """Function to generate peak anomalies.
        
        Parameters
        ----------
        peaks_num: int
            The number of peak anomalies
        """
        if self.verbose:
            print(f'Num of peaks: {peaks_num}')

        n_times = T.shape[0]
        peaks_inds = self.rng.randint(0, n_times, size=peaks_num)
        peaks_value = T[peaks_inds,:][:, ano_num_cols] < 0.5
        peaks_value = peaks_value + 0.1 * (1 - 2 * peaks_value) * self.rng.uniform(low=0, high=1, size=len(ano_num_cols))

        T[peaks_inds,:][:, ano_num_cols] = peaks_value
        ano_labels = np.zeros(n_times)
        ano_labels[peaks_inds] = 1

        return T, ano_labels

    def time_noise(self, 
                   T: np.ndarray,

                   ano_num_cols: np.ndarray,
                   ano_times: int,
                   ano_start: int,
                   ano_end: int,

                   noise_std: float = 0.05
                   ):
        """Function to inject time noise anomalies.
        
        Parameters
        ----------
        Anomalies exist in interval [ano_start, ano_end) with length of ano_times in all types of anomalies except peaks.

        noise_std: float 
            Std of gaussian noise in time domain.
        """
        if self.verbose:
            print(f'Std of time noise: {noise_std}')

        T[ano_start:ano_end, ano_num_cols] = np.clip(T[ano_start:ano_end, ano_num_cols] + self.rng.normal(loc=0, scale=noise_std, size=(ano_times, len(ano_num_cols))), a_min=0, a_max=1)

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels
    
    def spectral_noise(self, 
                       T: np.ndarray,

                       ano_num_cols: np.ndarray,
                       ano_start: int,
                       ano_end: int,

                       spectrum_rate: float = 0.1,
                       spectrum_segments_num: int = 3,
                       spectrum_amplitude_scale: float = 1.5,
                       spectrum_angle_std: float = 1,
                       ):
        """Function to inject spectral noise anomalies.
        
        Parameters
        ----------
        spectrum_rate: float
            Proportion of an amomalous segment to entire window.
        
        spectrum_segments_num: int
            Num of anomalous segments in frequency domain.
                       
        spectrum_amplitude_scale: float
            Std(noise)/Std(input) in amplitude of frequencies.
        
        spectrum_angle_std: float
            Std of noise added to angle of input.
        """
        if self.verbose:
            print(f'Proportion of a single segment to entire window: {spectrum_rate}')
            print(f'Num of segments: {spectrum_segments_num}')
            print(f'Times of std of amplitude disturbance: {spectrum_amplitude_scale}')
            print(f'Std of angle: {spectrum_angle_std}')

        T[ano_start:ano_end, ano_num_cols] = _perturbation_spectrum(T[ano_start:ano_end, ano_num_cols], r=spectrum_rate, K=spectrum_segments_num, r_amp=spectrum_amplitude_scale, theta=spectrum_angle_std)
        
        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def flip(self,
             T: np.ndarray, 

             ano_num_cols: np.ndarray,
             ano_cat_cols: np.ndarray,
             ano_start: int,
             ano_end: int,

             flip_direction: str = 'horizontal',
             ):
        """Function to inject flip anomalies.
        
        Parameters
        ----------
        T: np.ndarray [n_features, n_times]
            Time series.
        """
        if self.verbose:
            print(f'Flip direction: {flip_direction}')

        ano_cols = np.concatenate([ano_num_cols, ano_cat_cols])
        ano_cols = ano_cols.astype(int)

        if flip_direction == 'horizontal':
            T[ano_start:ano_end, ano_cols] = np.flip(T[ano_start:ano_end, ano_cols], axis=1)
        elif flip_direction == 'vertical':
            T[ano_start:ano_end, ano_cols] = 1 - T[ano_start:ano_end, ano_cols]
        elif flip_direction == 'both':
            T[ano_start:ano_end, ano_cols] = 1 - np.flip(T[ano_start:ano_end, ano_cols], axis=1)
        else:
            raise ValueError(f'Flip anomalies can only created in horizontal, vertical or both directions, but got {flip_direction}')

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels
          
    def length_adjusting(self, 
                         T: np.ndarray, 

                         ano_num_cols: np.ndarray,
                         ano_cat_cols: np.ndarray,
                         ano_times: int,
                         ano_start: int,
                         ano_end: int,

                         length_adjusting_scale: int = 2,
                         T_more: np.ndarray = np.zeros([0,0]), 
                         ):
        """Function to inject length adjusting anomalies.
        
        Parameters
        ----------
        length_adjusting_scale: int
            Times/Multiple of length adjustment

        T_more: np.ndarray
            Supplement the time steps that are left empty after shortening the abnormal fragments
        """
        if self.verbose:
            print(f'Length adjusting scale: {length_adjusting_scale}')

        ano_cols = np.concatenate([ano_num_cols, ano_cat_cols])
        n_times = T.shape[0]

        # Lengthenth
        if ano_times > 0.5 * n_times * self.ano_time_rate_max:
            if self.verbose:
                print('Lengthening')

            # The window's divisible scale length.
            ano_times = ano_times - ano_times % length_adjusting_scale  
            
            ano_end = ano_start + ano_times
            # The length after dividing by scale, which is the length before being stretched.
            repeat_segments = ano_times // length_adjusting_scale 

            T[ano_end:, ano_cols] = T[ano_start + repeat_segments: -(ano_times - repeat_segments), ano_cols].copy()
            T[ano_start:ano_end, ano_cols] =np.repeat(T[ano_start: ano_start+repeat_segments, ano_cols], length_adjusting_scale, axis=0)

        # Shortening, according to the approach in AnomalyBERT, should be to downsample a segment of length ano_time * scale to a length of ano_time, but at this point, the sequence length will decrease ano_time * (scale-1). If the data loader is convenient, add this length to the end of the data; Otherwise, copy the last value.
        else:  
            if self.verbose:
                print('Shortening')

            add_len = ano_times * (length_adjusting_scale - 1)
            if T_more.shape != (0, 0):
                T = np.concatenate((T, T_more[:add_len, :]), axis=0)
            else:
                T = np.concatenate((T, np.repeat(T[[-1], :], add_len, axis=0)), axis=0)
                T[ano_start:ano_end, ano_cols] = T[ano_start:ano_start + ano_times * length_adjusting_scale : length_adjusting_scale, ano_cols]
                T[ano_end:-add_len, ano_cols] = T[ano_start + ano_times * length_adjusting_scale:, ano_cols]
                T = T[:n_times, :]
        
        # If there is no external data and the index before shortening exceeds n_ times, then calculate the actual length of carboxylic acid.
        if T_more.shape == (0, 0) and ano_start + ano_times * length_adjusting_scale > n_times:
            ano_times = (n_times - ano_times) // length_adjusting_scale
        
        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def amplitude_scaling(self, 
                        T: np.ndarray, 

                        ano_num_cols: np.ndarray,
                        ano_start: int,
                        ano_end: int,

                        amplitude_scale: float = 4,
                        ):
        """Function to inject amplitude scaling anomalies.
        
        Parameters
        ----------
        amplitude_scale: float
            Times/Multiple of amplitude of anomaly to input in time domain.
        """
        if self.verbose:
            print(f'Times of amplitude scale: {amplitude_scale}')

        T[ano_start:ano_end, ano_num_cols] = amplitude_scale * T[ano_start:ano_end, ano_num_cols]

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def soft_replacing(self, 
                       T: np.ndarray, 

                       ano_num_cols: np.ndarray,
                       ano_cat_cols: np.ndarray,
                       ano_times: int,
                       ano_start: int,
                       ano_end: int,
                       num_cols: np.ndarray,
                       cat_cols: np.ndarray,

                       rep_data: np.ndarray = np.zeros([0, 0]),

                       soft_rep_weight: float = 0.7,
                       soft_rep_direction: str = 'horizontal',
                       ):
        """Function to inject soft replacing anomalies.
        
        Parameters
        ----------
        soft_rep_weight: float
            The weight of soft replacement, the maximum weight of replacement sequence in abnormal sequence

        soft_rep_direction: str
            The direction of soft replacement, refers to the flip direction of numerical columns. By default, there are options for horizontal, vertical, both, and none, where horizontal refers to flipping along the time dimension and vertical refers to the reversal of values (1-input).
        """
        if self.verbose:
            print(f'Weight of soft replacing: {soft_rep_weight}')
            print(f'Flip direction of soft replacing: {soft_rep_direction}')

        rep_data_len = rep_data.shape[0] 
        # rep_data, np.ndarray (n_features, rep_data_len)
        rep_num_cols = self.rng.choice(num_cols, size=len(ano_num_cols))
        rep_num_inds = self.rng.randint(0, rep_data_len - ano_times + 1, size=len(ano_num_cols))

        rep_num_timeseries = []
        for col, ind in zip(rep_num_cols, rep_num_inds):
            rep_num_timeseries.append(rep_data[ind:ind + ano_times, col])
        rep_num_timeseries = np.array(rep_num_timeseries).T

        if soft_rep_direction == 'horizontal':
        # soft_rep_direction可以选择 none， both， horizontal， vertical
            rep_num_timeseries = np.flip(rep_num_timeseries, axis=0)
        elif soft_rep_direction == 'vertical':
            rep_num_timeseries = 1 - rep_num_timeseries
        elif soft_rep_direction == 'both':
            rep_num_timeseries = 1 - np.flip(rep_num_timeseries, axis=0)
        elif soft_rep_direction == 'none':
            pass
        else:
            raise ValueError(f'Flip direction of soft replacing can only chosen in horizontal, vertical, both or none directions, but got {soft_rep_direction}')
            
        weights = self.replacing_weights(interval_len=ano_times , replacing_weight=soft_rep_weight)
        weights = np.tile(weights.reshape(ano_times, -1), (1, len(ano_num_cols)))
        T[ano_start:ano_end, ano_num_cols] = T[ano_start:ano_end, ano_num_cols] * (1 - weights) + rep_num_timeseries * weights

        # The categorical column adopts a direct substitution method instead of a weighted substitution method for the numerical column.
        if ano_cat_cols.size > 0:
            rep_cat_cols = self.rng.choice(cat_cols, size=len(ano_cat_cols))
            rep_cat_inds = self.rng.randint(0, rep_data_len - ano_times + 1, size=len(ano_cat_cols))

            rep_cat_timeseries = []
            for col, ind in zip(rep_cat_cols, rep_cat_inds):
                rep_cat_timeseries.append(rep_data[ind:ind + ano_times, col])
            rep_cat_timeseries = np.array(rep_cat_timeseries).T

            T[ano_start:ano_end, ano_cat_cols] = rep_cat_timeseries
        
        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def uniform_replacing(self, 
                          T: np.ndarray, 
                          ano_num_cols: np.ndarray,
                          ano_times: int,
                          ano_start: int,
                          ano_end: int,
                          ):
        """Function to inject uniform replacing anomalies.
        
        Parameters
        ----------
        """
        values = self.rng.rand(1, len(ano_num_cols))
        T[ano_start:ano_end, ano_num_cols] = np.tile(values, (ano_times, 1))

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def average(self, 
                T: np.ndarray, 
                ano_num_cols: np.ndarray,
                ano_start: int,
                ano_end: int,
                ma_window: int = 4,
                ):
        """Function to inject moving average anomalies.
        
        Parameters
        ----------
        ma_window: int
            Window length for sliding average of time series

        """
        if self.verbose:
            print(f'Window of average anomalies: {ma_window}')

        T[ano_start:ano_end, ano_num_cols] = _moving_average(T[ano_start:ano_end, ano_num_cols], w=ma_window)

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels

    def cutoff(self, 
               T: np.ndarray, 
               ano_num_cols: np.ndarray,
               ano_times: int,
               ano_start: int,
               ano_end: int,
               constant_type: str = 'noisy_0',
               constant_quantile: Union[int, float] = 75,
               ):
        """Function to inject cutoff anomalies.
        
        Parameters
        ----------
        constant_type: str
            Cutoff refers to replacing the values of all feature dimensions on the abnormal time step with the same one. There are three constant replacement methods, 0,1, percentile; 2. Random replacement method, Gaussian noise centered on 0 and 1. The parameter "constant_type" refers to the categories of the above 5 categories, using 0,1, quantity, noise_ 0, noise_ 1 to represent separately.
        
        constant_quantile: Union[int, float]
            When the cutoff type is quantile, it is necessary to specify the percentile, which can be represented by integers or floating-point numbers. The floating-point number needs to be a decimal between 0 and 1.
        """
        constant_quantile = constant_quantile if isinstance(constant_quantile, int) else 100 * constant_quantile

        if self.verbose:
            print(f'Constant type of cutoff: {constant_type}')
            if constant_type=='quantile':
                print(f'Percentile of quantile cutoff: {constant_quantile}')

        constant_map = {'0':0, '1':1}
        constant_map['quantile'] = np.percentile(T[ano_start:ano_end, ano_num_cols], q=constant_quantile)
        constant_map['noisy_0'] = self.rng.normal(size=(ano_times, 1), scale=0.01)
        constant_map['noisy_0'] = np.tile(constant_map['noisy_0'], (1, len(ano_num_cols)))
        constant_map['noisy_1'] = 1 + constant_map['noisy_0']
        T[ano_start:ano_end, ano_num_cols] = constant_map[constant_type]
        
        ano_labels = self._gen_label(T, ano_start, ano_end)
        
        return T, ano_labels

    def wander(self, 
               T: np.ndarray, 
               ano_num_cols: np.ndarray,
               ano_times: int,
               ano_start: int,
               ano_end: int,
               baseline: float = 0.2,
               ):
        """Function to add a linear trend from 0 to "baseline", where the length is the length of the anomalous segment.
        """
        if self.verbose:
            print(f'Ending value of trend in wander anomaly: {baseline}')

        T[ano_start:ano_end, ano_num_cols] = np.tile(np.linspace(0, baseline, ano_times).reshape(-1, 1), (1, len(ano_num_cols))) + T[ano_start:ano_end, ano_num_cols]
        T[ano_end:, ano_num_cols] = baseline + T[ano_end:, ano_num_cols]

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels
        
    
    def contextual(self, 
                   T: np.ndarray, 
                   ano_num_cols: np.ndarray,
                   ano_start: int,
                   ano_end: int,
                   contextual_scale: float = 2, 
                   ):
        """Function to inject contextual anomalies.
        
        Parameters
        ----------
        contextual_scale: float
            Standard deviation of Gaussian distribution, used to generate slope and intercept parameters for linear transformation
        """
        if self.verbose:
            print(f'Std of parameters of contextual: {contextual_scale}')

        a = self.rng.normal(loc=1, scale=contextual_scale, size=1)
        b = self.rng.normal(loc=0, scale=contextual_scale, size=1)
        T[ano_end:, ano_num_cols] = a * T[ano_end:, ano_num_cols] + b

        ano_labels = self._gen_label(T, ano_start, ano_end)

        return T, ano_labels



if __name__ == "__main__":
    anomalies = [
                'peaks', 
                'time_noise', 'spectral_noise', 
                'flip', 'length_adjusting', 'amplitude_scaling',  
                'soft_replacing', 'uniform_replacing', 'average', 'cutoff', 
                'wander', 'contextual'
            ] 
    test_data1 = np.arange(80).reshape(16, 5).astype(float)
    test_data2 = np.arange(80).reshape(16, 5).astype(float)
    test_data = np.stack((test_data1,test_data2),axis=0)
    print(test_data.shape)


    inj = InjectAnomalies(ano_col_rate=0.3, ano_time_rate_max=0.3,random_state=42)
    label = []
    for i in range(test_data.shape[0]):
        test_data[i],la = inj.inject_anomalies(test_data[i], 
                                  anomaly_type=anomalies[1], 
                                  use_global_config=True)
    print(test_data)
        
    









'''
import pandas as pd
x = pd.read_csv('to/dataset/synthetic/unidataset/01234.csv')

x = x['value'].values
y = np.array([x[100*i:100*(i+1)] for i in range(10)], dtype=float)
y = np.tile(y, (5,2))
num_cols = np.sort(self.rng.choice(np.arange(50), 30, replace=False))
cat_cols = np.array([i for i in np.arange(50) if i not in num_cols])
y[cat_cols, :] = y[cat_cols, :] < y[cat_cols, :].mean(axis=1, keepdims=True) 

inj = InjectAnomalies()
anomalies = [
            'peaks', 
            'time_noise', 'spectral_noise', 
            'flip', 'length_adjusting', 'amplitude_scaling',  
            'soft_replacing', 'uniform_replacing', 'average', 'cutoff', 
            'wander', 'contextual'
        ] * 30
for j in range(30):
    i = j*12 + 11
    print(f'\n第{i}个异常：{anomalies[i]}')
    y1, l1 = inj.inject_anomalies(y, 
                                  num_cols=num_cols, 
                                  cat_cols=cat_cols,
                                  anomaly_type=anomalies[i], 
                                  use_global_config=True, 
                                  rep_data=y)
    print(y1.shape, l1.shape)
                        
self = inj
T = y.copy()
patch_size = 1
ano_col_rate = 0.3
ano_time_rate_max =  0.15


a = np.arange(80).reshape(16, 5).astype(float)
a_s = _moving_average(a, 3)
a_spectral = _perturbation_spectrum(a)
print(np.round(np.concatenate((a_spectral, a),axis=1),2))
print(a)
'''
