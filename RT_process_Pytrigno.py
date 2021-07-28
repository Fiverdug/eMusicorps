import pytrigno
from data_plot import init_plot_EMG, update_plot_EMG
from data_processing import process_EMG_RT, process_accel, process_gyro, add_data_to_pickle
from time import sleep, time, strftime
import os
import numpy as np
import scipy.io as sio

from pythonosc.udp_client import SimpleUDPClient


def run(
    muscles_range,
    get_EMG=True,
    get_accel=True,
    get_gyro=True,
    read_freq=100,
    MVC_list=(),
    host_ip=None,
    EMG_freq=2000,
    IM_freq=148.1,
    EMG_windows=2000,
    IM_windows=100,
    accept_delay=0.005,
    save_data=True,
    output_dir=None,
    output_file=None,
    show_data=None,
    print_data=False,
    server="pytrigno",
    norm_EMG=True,
    muscle_names=(),
    test_with_connection=True,
    OSC_stream=False,
    OSC_ip="127.0.0.1",
    OSC_port=51337,
):
    """
        Run streaming of delsys sensor data with real time processing and plotting.
        ----------
        muscles_range: tuple
            list of sensor to stream, note that last values is included.
        get_EMG: bool
            True to stream EMG data
        get_accel: bool
            True to stream accelerometer data
        get_gyro: bool
            True to stream gyroscope data
        read_freq: int
            frequency at which the system will read sensor data.
        MVC_list: list
            list of MVC value length must be the same than sensor number.
        host_ip: str
            IP adress of the device which run the trigno software default is 'localhost'.
        EMG_freq: float
            frequency of EMG data.
        IM_freq: float
            frequency of inertial measurement data.
        EMG_windows: float
            size of the sliding window for EMG processing.
        IM_windows: float
            size of the sliding window for IM processing.
        accept_delay: float
            acceptable delay between real time and data streaming.
        save_data: bool
            True for save data during streaming, this can impact the realtime streaming.
        output_dir: str
            name of output directory.
        output_file: str
            name of output file.
        show_data: list
            list of name of data to plot. Can be: 'emg' or 'raw_emg' (gyro and accel not implemented yet).
        print_data: bool
            True to print data in the console
        server: str
            method to stream data. Can be 'pytrigno'.
        norm_EMG: bool
            True for normalize EMG in real time. Note that you need a MVC list.
        muscle_names: list
            list of muscle names. Length must be the same than the number of delsys sensors.
        OSC_stream: bool
            Stream OSC (open sound control) value to destination
        OSC_port: int
            OSC output port (must be over 1024 and under 65000), default : 51337 
        OSC_ip: str
            OSC output ip address, default : 127.0.0.1        
        Returns
            -------
     """
    if test_with_connection is not True:
        print("[WARNING] Please note that you are in 'no connection' mode for debug.")

    if get_EMG is False and get_accel is False and get_gyro is False:
        raise RuntimeError("Please define at least one data to read (emg/gyro/accel).")
    if get_gyro is True:
        print("[WARNING] Please note that only avanti sensor have gyroscope data available.")

    if show_data:
        for data in show_data:
            if data == "accelerometer" or data == "accel":
                raise RuntimeError("Plot accelerometer data not implemented yet.")
            elif data == "gyroscope" or data == "gyro":
                raise RuntimeError("Plot gyroscope data not implemented yet.")

    if server == "vicon" and get_accel is True or server == "vicon" and get_gyro is True:
        raise RuntimeError("Read IM data with vicon not implemented yet")

    current_time = strftime("%Y%m%d-%H%M")
    output_file = output_file if output_file else f"trigno_streaming_{current_time}"

    output_dir = output_dir if output_dir else "live_data"

    if os.path.isdir(output_dir) is not True:
        os.mkdir(output_dir)

    if os.path.isfile(f"{output_dir}/{output_file}"):
        os.remove(f"{output_dir}/{output_file}")

    data_path = f"{output_dir}/{output_file}"

    if get_accel is not True and get_EMG is not True and get_gyro is not True:
        raise RuntimeError("Please define at least one data to read.")

    dev_EMG = []
    dev_IM = []
    if isinstance(muscles_range, tuple) is not True:
        raise RuntimeError("muscles_range must be a tuple.")
    n_muscles = muscles_range[1] - muscles_range[0] + 1

    EMG_sample = int(EMG_freq / read_freq)
    IM_sample = int(IM_freq / read_freq)
    IM_range = (muscles_range[0], muscles_range[0] + (n_muscles * 9))

    host_ip = "localhost" if None else host_ip
    if test_with_connection is True:
        if server == "pytrigno":
            if get_EMG is True:
                dev_EMG = pytrigno.TrignoEMG(channel_range=muscles_range, samples_per_read=EMG_sample, host=host_ip)
            if get_accel is True:
                dev_IM = pytrigno.TrignoIM(channel_range=IM_range, samples_per_read=IM_sample, host=host_ip)

    IM = np.zeros((n_muscles, 9, IM_sample))
    data_EMG_tmp = []
    data_IM_tmp = []
    raw_EMG = []
    EMG_proc = []
    EMG_to_plot = []

    if show_data:
        p, win_EMG, app, box = init_plot_EMG(n_muscles, muscle_names)

    if len(muscle_names) == 0:
        muscle_names = []
        for i in range(n_muscles):
            muscle_names.append("muscle_" + f"{i}")

    print("Streaming data.....")
    ma_win = 200
    if test_with_connection is not True:
        EMG_exp = sio.loadmat("EMG_test.mat")["EMG"][:, :1500]

    c = 0
    initial_time = time()
    if OSC_stream is True:
        OSC_client = SimpleUDPClient(OSC_ip, OSC_port)
        print("Streaming OSC activated") 
    while True:
        if test_with_connection:
            if server == "pytrigno":
                if get_EMG is True:
                    dev_EMG.reset()
                    dev_EMG.start()
                    data_EMG_tmp = dev_EMG.read()

                if get_accel is True:
                    dev_IM.reset()
                    dev_IM.start()
                    data_IM_tmp = dev_IM.read()
                    data_IM_tmp = data_IM_tmp.reshape(n_muscles, 9, IM_sample)
            else:
                raise RuntimeError(f"Server '{server}' not valid, please use 'pytrigno' server.")
        else:
            if get_EMG is True:
                if c < EMG_exp.shape[1]:
                    data_EMG_tmp = EMG_exp[:n_muscles, c : c + EMG_sample]
                    c += EMG_sample
                else:
                    c = 0
            if get_accel is True:
                data_IM_tmp = np.random.random((n_muscles, 9, IM_sample))
        tic = time()

        if get_EMG is True:
            raw_EMG, EMG_proc = process_EMG_RT(
                raw_EMG,
                EMG_proc,
                data_EMG_tmp,
                MVC_list=MVC_list,
                ma_win=ma_win,
                EMG_win=EMG_windows,
                EMG_freq=EMG_freq,
                norm_EMG=norm_EMG,
                lpf=False,
            )

            if show_data:
                for data in show_data:
                    if data == "raw_emg":
                        if raw_EMG.shape[1] < EMG_windows:
                            EMG_to_plot = np.append(
                                np.zeros((raw_EMG.shape[0], EMG_windows - raw_EMG.shape[1])), raw_EMG, axis=1
                            )
                        else:
                            EMG_to_plot = raw_EMG
                    elif data == "emg":
                        if EMG_proc.shape[1] < read_freq:
                            EMG_to_plot = np.append(
                                np.zeros((EMG_proc.shape[0], read_freq - EMG_proc.shape[1])), EMG_proc, axis=1
                            )
                        else:
                            EMG_to_plot = EMG_proc

            # print EMG data
            if print_data is True:
                print(f"EMG processed data :\n {EMG_proc[:, -1:]}")
            if OSC_stream is True:
                OSC_client.send_message("/EMG/processed/", np.mean(EMG_proc[:, -EMG_sample:],axis=1))
        
        if get_accel is True or get_gyro is True:
            IM = np.concatenate((IM[:, :, -IM_windows + IM_sample :], data_IM_tmp), axis=2)
            accel = IM[:, :3, -IM_windows:]
            gyro = IM[:, 3:6, -IM_windows:]
            accel_proc = process_accel(accel, IM_freq)  # return raw data for now
            gyro_proc = process_gyro(gyro, IM_freq)  # return raw data for now

            # Print IM data
            if print_data is True:
                print(f"Accel data :\n {accel_proc[:, :, -IM_sample:]}")
                print(f"Gyro data :\n {gyro_proc[:, :, -IM_sample:]}")
            if OSC_stream is True: 
                if get_accel is True:
                    i=0
                    for x in np.mean(accel_proc[:, :, -IM_sample:],axis=2):
                        j=0
                        for y in x:
                            OSC_client.send_message("/accel/"+str(i)+"/"+str(j),  y)
                            j=j+1
                        i=i+1
                    pass
                if get_gyro is True:
                    i=0
                    for x in np.mean(gyro_proc[:, :, -IM_sample:],axis=2):
                        j=0
                        for y in x:
                            OSC_client.send_message("/gyro/"+str(i)+"/"+str(j),  y)
                            j=j+1
                        i=i+1
                    pass
        # Save data
        if save_data is True:
            data_to_save = {
                "Time": time() - initial_time,
                "EMG_freq": EMG_freq,
                "IM_freq": IM_freq,
                "read_freq": read_freq,
            }
            if get_EMG is True:
                data_to_save["EMG_proc"] = EMG_proc[:, -1:]
                data_to_save["raw_EMG"] = data_EMG_tmp
            if get_gyro is True:
                data_to_save["raw_accel"] = data_IM_tmp[:, 0:3, :]
            if get_accel is True:
                data_to_save["raw_gyro"] = data_IM_tmp[:, 3:6, :]

            add_data_to_pickle(data_to_save, data_path)

        # Plot data real time
        if show_data:
            for data in show_data:
                if data == "raw_emg" or data == "emg":
                    update_plot_EMG(EMG_to_plot, p, app, box)

                else:
                    raise RuntimeError(
                        f"{data} is unknown. Please define a valid type of data to plot ('emg' or raw_emg')."
                    )

        t = time() - tic
        time_to_sleep = 1 / read_freq - t
        if time_to_sleep > 0:
            sleep(time_to_sleep)
        elif float(abs(time_to_sleep)) < float(accept_delay):
            pass
        else:
            print(
                f"[Warning] Processing need to much time and delay ({abs(time_to_sleep)}) exceeds "
                f"the threshold ({accept_delay}). Try to reduce the read frequency or EMG frequency."
            )
