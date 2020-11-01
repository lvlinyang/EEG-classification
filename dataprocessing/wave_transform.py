import pywt
import numpy as np
from braindecode.datautil.signal_target import SignalAndTarget

# wavelet transfrom

def nomal(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data

def data_all_chan_cwtandraw(test_set):
    """
    :param test_set:
    :return:  all channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # cwt for test_set
    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in range(test_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)
            
        # 对小波变换之后的数据进行归一化
        test_channel_cwt_data = nomal(np.array(test_channel_cwt_data))
        test_channel_cwt_data = test_channel_cwt_data.tolist()
        
        test_channel_cwt_data.append(t)
        test_trial_data.append(np.array(test_channel_cwt_data))

    test_cwt_signal = np.array(test_trial_data).astype(np.float32)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return test_cwt_dataset


def Continuous_Wavelt_Transform(data, wavename, totalscal, sampling_rate):
    fc = pywt.central_frequency(wavename)  # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(1, totalscal + 1)
    [coef, freqs] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    return coef, freqs
