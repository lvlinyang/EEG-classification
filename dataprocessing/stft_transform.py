import pywt
import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget
import torch
from torch import nn
import pywt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

n_fft = 136
hop_length = 22
x = torch.randn(230,22,1125)
for i,data in enumerate(x):
    stft_data = []
    stft = torch.stft(data, n_fft, hop_length=hop_length, win_length=64, window=None, center=True,
                                                   pad_mode='reflect', normalized=True)















# stft transfrom
"""
BCI_data: c3:7 ; c4:9 ; cz:11.
HGM_data: c3:4 ; c4:5.

"""

# def data_stft(train_set, test_set):
#     """
#
#     :param train_set:
#     :param test_set:
#     :return: all channle wave transform
#     """
#     wavename = 'morl'
#     n_fft = 136
#     hop_length = 22
#     # channel_cwt_data = []
#     train_trial_data = []
#     # cwt for train_set
#     for i, t in enumerate(train_set.X):
#         stft = torch.stft(t, n_fft, hop_length=hop_length, win_length=64, window=None, center=True,
#                              pad_mode='reflect', normalized=True)
#
#
#
#
#         train_channel_stft_data = []
#         for j in [7,9]:
#             out = torch.stft(input_torch, n_fft, hop_length=hop_length, win_length=64, window=None, center=True,
#                              pad_mode='reflect', normalized=True)
#             cwtmatr = cwtmatr[0:22, :].astype(np.float32)
#             train_channel_stft_data.append(cwtmatr)
#         train_trial_data.append(np.array(train_channel_cwt_data))
#
#     train_cwt_signal = np.array(train_trial_data)
#     train_cwt_targal = train_set.y
#     train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)
#
#     test_trial_data = []
#     for i, t in enumerate(test_set.X):
#         test_channel_cwt_data = []
#         for j in range(test_set.X.shape[1]):
#             cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
#             cwtmatr = cwtmatr[0:22, :].astype(np.float32)
#             test_channel_cwt_data.append(cwtmatr)
#         test_trial_data.append(np.array(test_channel_cwt_data))
#
#     test_cwt_signal = np.array(test_trial_data)
#     test_cwt_targal = test_set.y
#     test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)
#
#     return train_cwt_dataset, test_cwt_dataset


if __name__ == '__main__':
    input = torch.randn(58,3,1125,22)



    print("hello  lly")













