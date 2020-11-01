
import pywt
import numpy as np

from braindecode.datautil.signal_target import SignalAndTarget

# wavelet transfrom
"""
BCI_data: c3:7 ; c4:9 ; cz:11.
HGM_data: c3:4 ; c4:5.
"""

def nomal_example(data):
    for i in range(data.shape(0)):
        max = np.max(data[i])
        min = np.min(data[i])
        data[i] = (data[i] - min) / (max - min)
    return data

def nomal(data):
    max = np.max(data)
    min = np.min(data)
    data = (data - min) / (max - min)
    return data

def nomal_score(data):
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data-mean_data)/std_data
    return data

def data_cwt(train_set, test_set):
    """

    :param train_set:
    :param test_set:
    :return: all channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in range(train_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in range(test_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)
        test_trial_data.append(np.array(test_channel_cwt_data))

    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset

def data_all_chan_cwtandraw(train_set, test_set):
    """
    :param train_set:
    :param test_set:
    :return:  all channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set

    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in range(train_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)

        # 对小波变换之后的数据进行归一化
        train_channel_cwt_data = nomal(np.array(train_channel_cwt_data))
        train_channel_cwt_data = train_channel_cwt_data.tolist()

        train_channel_cwt_data.append(t)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data).astype(np.float32)
    
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

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

    return train_cwt_dataset, test_cwt_dataset


def data_all_chan_cwtandraw_WithoutRawdata(train_set, test_set):
    """
    :param train_set:
    :param test_set:
    :return:  all channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set

    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in range(train_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)

        # train_channel_cwt_data.append(t)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data).astype(np.float32)

    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in range(test_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)

        # test_channel_cwt_data.append(t)
        test_trial_data.append(np.array(test_channel_cwt_data))

    test_cwt_signal = np.array(test_trial_data).astype(np.float32)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset

def data_c3c4_cwt(train_set, test_set):
    """

    :param train_set:
    :param test_set:
    :return: ac3  c4  channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in [7,11]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in [7,11]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)
        test_trial_data.append(np.array(test_channel_cwt_data))

    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset
def data_c3c4cz_cwt(train_set, test_set):
    """

    :param train_set:
    :param test_set:
    :return:  c3 c4 cz channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in [7,9,11]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in [7,9,11]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)
        test_trial_data.append(np.array(test_channel_cwt_data))

    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset
def data_c3c4_cwt_and_rowdata(train_set, test_set):
    """

    :param train_set:
    :param test_set:
    :return: c3 c4 channle wave transform and raw data
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in [7,9]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        # train_channel_cwt_data = nomal(train_channel_cwt_data)   # let wave_data nomal
        train_channel_cwt_data.append(t)   # t :原始数据
        train_trial_data.append(np.array(train_channel_cwt_data))
        # train_trial_data.append(t)   # 增加原始数据

    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in [7,9]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)
        # test_channel_cwt_data = nomal(test_channel_cwt_data)   # let wave_data nomal
        test_channel_cwt_data.append(t)   # t :原始数据
        test_trial_data.append(np.array(test_channel_cwt_data))
        # train_trial_data.append(t)  # 增加原始数据

    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset

def data_c3c4cz_cwt_and_rowdata(train_set, test_set):
    """
    :param train_set:
    :param test_set:
    :return:  all channle wave transform
    """
    wavename = 'morl'
    totalscal = 64
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set

    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in [7,9,11]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:22, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)

        # 对小波变换之后的数据进行归一化
        train_channel_cwt_data = nomal(np.array(train_channel_cwt_data))
        train_channel_cwt_data = train_channel_cwt_data.tolist()

        train_channel_cwt_data.append(t)
        train_trial_data.append(np.array(train_channel_cwt_data))

    train_cwt_signal = np.array(train_trial_data).astype(np.float32)

    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in [7,9,11]:
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

    return train_cwt_dataset, test_cwt_dataset

def hgm_data_all_chan(train_set, test_set):
    """
    :param train_set:
    :param test_set:
    :return: ac3  c4  channle wave transform
    """
    wavename = 'morl'
    totalscal = 125
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in range(train_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:44, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        train_channel_cwt_data.append(t)
        train_trial_data.append(np.array(train_channel_cwt_data))


    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in range(train_set.X.shape[1]):
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:44, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)

        test_channel_cwt_data.append(t)
        test_trial_data.append(np.array(test_channel_cwt_data))



    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset


def hgm_data_c3c4_cwt(train_set, test_set):
    """
    :param train_set:
    :param test_set:
    :return: ac3  c4  channle wave transform
    """
    wavename = 'morl'
    totalscal = 125
    sampling_rate = 250
    # channel_cwt_data = []
    train_trial_data = []
    # cwt for train_set
    for i, t in enumerate(train_set.X):
        train_channel_cwt_data = []
        for j in [4,5]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:44, :].astype(np.float32)
            train_channel_cwt_data.append(cwtmatr)
        # 对小波变换之后的数据进行归一化


        train_channel_cwt_data.append(t)
        train_trial_data.append(np.array(train_channel_cwt_data))


    train_cwt_signal = np.array(train_trial_data)
    train_cwt_targal = train_set.y
    train_cwt_dataset = SignalAndTarget(train_cwt_signal, train_cwt_targal)

    test_trial_data = []
    for i, t in enumerate(test_set.X):
        test_channel_cwt_data = []
        for j in [4,5]:
            cwtmatr, frequencies = Continuous_Wavelt_Transform(np.squeeze(t[j, :]), wavename, totalscal, sampling_rate)
            cwtmatr = cwtmatr[0:44, :].astype(np.float32)
            test_channel_cwt_data.append(cwtmatr)

        test_channel_cwt_data.append(t)
        test_trial_data.append(np.array(test_channel_cwt_data))



    test_cwt_signal = np.array(test_trial_data)
    test_cwt_targal = test_set.y
    test_cwt_dataset = SignalAndTarget(test_cwt_signal, test_cwt_targal)

    return train_cwt_dataset, test_cwt_dataset



def Continuous_Wavelt_Transform(data, wavename, totalscal, sampling_rate):
    fc = pywt.central_frequency(wavename)  # central frequency
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(1, totalscal + 1)
    [coef, freqs] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    return coef, freqs
