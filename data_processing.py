import numpy as np
from scipy.signal import butter, lfilter, filtfilt, convolve
import pickle
from time import time

try:
    from pyomeca import Analogs

    pyomec_module = True
except ModuleNotFoundError:
    pyomec_module = False


def process_EMG(data, frequency, bpf_lcut=10, bpf_hcut=425, lpf_lcut=5, order=4, MA_win=200, pyomec=False, MA=False):
    if pyomec is True:
        if pyomec_module is False:
            raise RuntimeError("Pyomeca module not found.")
        if MA is True:
            raise RuntimeError("Moving average not available with pyomeca.")
        emg = Analogs(data)
        EMG_processed = (
            emg.meca.band_pass(order=order, cutoff=[bpf_lcut, bpf_hcut], freq=frequency)
            .meca.abs()
            .meca.low_pass(order=order, cutoff=lpf_lcut, freq=frequency)
        )
        EMG_processed = EMG_processed.values

    else:
        EMG_processed = abs(butter_bandpass_filter(data, bpf_lcut, bpf_hcut, frequency))
        if MA is True:
            w = np.repeat(1, MA_win) / MA_win
            empty_ma = np.ndarray((data.shape[0], data.shape[1]))
            EMG_processed = moving_average(EMG_processed, w, empty_ma)
        else:
            EMG_processed = butter_lowpass_filter(EMG_processed, lpf_lcut, frequency, order=order)
    return EMG_processed


def normalize_EMG(EMG, MVC_list):
    if len(MVC_list) == 0:
        raise RuntimeError("Please give a list of MVC to normalize the EMG signal.")
    norm_EMG = np.zeros((EMG.shape[0], EMG.shape[1]))
    for emg in range(EMG.shape[0]):
        norm_EMG[emg, :] = EMG[emg, :] / MVC_list[emg]
    return norm_EMG


# TODO: add framework to accelerometer signal processing
def process_IMU(IM_proc, raw_IM, IM_tmp, IM_win, IM_sample, ma_win, accel=False):

    if len(raw_IM) == 0:
        IM_proc = np.zeros((IM_tmp.shape[0], 1))
        raw_IM = IM_tmp

    elif raw_IM.shape[2] < ma_win:
        IM_proc = np.zeros((IM_tmp.shape[0], int(raw_IM.shape[2] / IM_sample)))
        raw_IM = np.append(raw_IM[:, :, -IM_win + IM_sample :], IM_tmp, axis=2)

    else:
        raw_IM = np.append(raw_IM[:, :, -IM_win + IM_sample :], IM_tmp, axis=2)
        IM_proc_tmp = raw_IM
        average = np.mean(IM_proc_tmp[:, :, -ma_win:], axis=2).reshape(-1, 3, 1)
        if accel:
            average = abs(np.linalg.norm(average, axis=1)-1)
        else:
            average = np.linalg.norm(average, axis=1)
        IM_proc = np.append(IM_proc[:, 1:], average, axis=1)


    return raw_IM, IM_proc


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype="low")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, lowcut, fs, order=4):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def moving_average(data, w, empty_ma):
    for i in range(data.shape[0]):
        empty_ma[i, :] = convolve(data[i, :], w, mode="same", method="fft")
    return empty_ma


def process_EMG_RT(raw_EMG, EMG_proc, EMG_tmp, MVC_list, ma_win, EMG_win=2000, EMG_freq=2000, norm_EMG=True, lpf=False):
    if ma_win > EMG_win:
        raise RuntimeError(f"Moving average windows ({ma_win}) higher than EMG windows ({EMG_win}).")
    EMG_sample = EMG_tmp.shape[1]
    if norm_EMG is True:
        if isinstance(MVC_list, np.ndarray) is True:
            if len(MVC_list.shape) == 1:
                quot = MVC_list.reshape(-1, 1)
            else:
                quot = MVC_list
        else:
            quot = np.array(MVC_list).reshape(-1, 1)
    else:
        quot = [1]

    if len(raw_EMG) == 0:
        EMG_proc = np.zeros((EMG_tmp.shape[0], 1))
        raw_EMG = EMG_tmp

    elif raw_EMG.shape[1] < EMG_win:
        EMG_proc = np.zeros((EMG_tmp.shape[0], int(raw_EMG.shape[1] / EMG_sample)))
        raw_EMG = np.append(raw_EMG, EMG_tmp, axis=1)

    else:
        raw_EMG = np.append(raw_EMG[:, -EMG_win + EMG_sample :], EMG_tmp, axis=1)
        EMG_proc_tmp = abs(butter_bandpass_filter(raw_EMG, 10, 425, EMG_freq))

        if lpf is True:
            EMG_lpf_tmp = butter_lowpass_filter(EMG_proc_tmp, 5, EMG_freq, order=4)
            EMG_lpf_tmp = EMG_lpf_tmp[:, ::EMG_sample]
            EMG_proc = np.append(EMG_proc[:, EMG_sample:], EMG_lpf_tmp[:, -EMG_sample:], axis=1)

        else:
            average = np.mean(EMG_proc_tmp[:, -ma_win:], axis=1).reshape(-1, 1)
            EMG_proc = np.append(EMG_proc[:, 1:], average / quot, axis=1)
    return raw_EMG, EMG_proc


def add_data_to_pickle(data_dict, data_path):
    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


def read_data(filename):
    data = {}
    with open(filename, "rb") as inf:
        count = 0
        while True:
            try:
                data_tmp = pickle.load(inf)
                for key in data_tmp.keys():
                    if count == 0:
                        if isinstance(data_tmp[key], (int, float, str)) is True:
                            data[key] = [data_tmp[key]]
                        else:
                            data[key] = data_tmp[key]
                    else:
                        if isinstance(data_tmp[key], (int, float, str)) is True:
                            if key == "time" or key == "Time":
                                data[key].append(data_tmp[key])
                        elif len(data_tmp[key].shape) == 2:
                            data[key] = np.append(data[key], data_tmp[key], axis=1)
                        elif len(data_tmp[key].shape) == 3:
                            data[key] = np.append(data[key], data_tmp[key], axis=2)
                if count == 0:
                    count += 1
            except EOFError:
                break
    return data


def add_to_timeline(init_time, timeline):
    return timeline.append(time() - init_time)
