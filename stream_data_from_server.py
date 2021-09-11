import scipy.io as sio
from client import Client
import numpy as np
import os
from time import time
from pythonosc.udp_client import SimpleUDPClient
from data_processing import add_data_to_pickle

if __name__ == '__main__':
    # Set program variables
    read_freq = 100  # Be sure that it's the same than server read frequency
    show_data = ["raw_emg"]  # can be ["emg"] to show process EMG
    device_host = "192.168.1.211"  # IP address of computer which run trigno device
    n_electrode = 10
    n_elec_pianist_1 = 5
    n_elec_pianist_2 = n_electrode - n_elec_pianist_1

    # load MVC data from previous trials.
    file_name = "MVC_xxxx.mat"
    file_dir = "MVC_01_08_2021"
    list_mvc = sio.loadmat(f"{file_dir}/{file_name}")["MVC_list_max"]
    list_mvc = list_mvc[:, :n_electrode].T
    list_mvc = list_mvc.tolist()

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_accel"

    # Run streaming data
    muscles_idx_pianist = [(0, n_elec_pianist_1 - 1), (n_elec_pianist_1 - 1, n_electrode - 1)]
    host_ip = 'localhost'
    host_port = 50000
    get_accel = True
    get_gyro = True
    get_emg = True
    print_data = True
    OSC_ip = "127.0.0.1"
    OSC_port = 51337
    OSC_stream = False
    save_data = True
    data_path = 'streamed_data'
    if os.path.isfile(data_path):
        os.remove(data_path)
    if OSC_stream is True:
        OSC_client = SimpleUDPClient(OSC_ip, OSC_port)
        print("Streaming OSC activated")
    count = 0
    pianist = input("Please choose a pianist (1 or 2), then press enter.")
    pianist = int(pianist)
    muscles_idx = slice(muscles_idx_pianist[pianist - 1][0], muscles_idx_pianist[pianist - 1][-1])
    initial_electrode = 0 if pianist == 1 else n_elec_pianist_1
    final_electrode = n_elec_pianist_1 if pianist == 1 else n_elec_pianist_1 + n_elec_pianist_2
    while True:
        client = Client(host_ip, host_port, "TCP")
        data = client.get_data(data=["emg", 'imu'],
                               Nmhe=read_freq,
                               exp_freq=read_freq,
                               nb_of_data=1,
                               raw=True,
                               norm_emg=True,
                               mvc_list=list_mvc
                               )

        data['gyro_proc'] = []
        data['accel_proc'] = []
        data["pianist_player"] = pianist
        EMG = np.array(data['emg'])[initial_electrode:final_electrode, :]
        raw_emg = np.array(data['raw_emg'])[initial_electrode:final_electrode, :]

        if len(np.array(data['imu']).shape) == 3:
            accel_proc = np.array(data['imu'])[initial_electrode:final_electrode, :3, :]
            gyro_proc = np.array(data['imu'])[initial_electrode:final_electrode, 3:6, :]
            raw_accel = np.array(data['raw_imu'])[initial_electrode:final_electrode, :3, :]
            raw_gyro = np.array(data['raw_imu'])[initial_electrode:final_electrode, 3:6, :]
        else:
            accel_proc = np.array(data['imu'])[initial_electrode:final_electrode, :]
            gyro_proc = np.array(data['imu'])[n_electrode + initial_electrode:n_electrode + final_electrode:, :]
            raw_accel = np.array(data['raw_imu'])[initial_electrode:final_electrode, :3, :]
            raw_gyro = np.array(data['raw_imu'])[initial_electrode:final_electrode, 3:6, :]

        if print_data is True:
            print(f"Accel data :\n"
                  f"proc : {accel_proc}\n"
                  f"raw : {raw_accel}\n")
            print(f"Gyro data :\n"
                  f"proc: {gyro_proc}\n"
                  f"raw: {raw_gyro}")
            print(f'EMG data: \n'
                  f'proc: {EMG}\n'
                  f'raw: {raw_emg}\n')

        if OSC_stream is True:
            if get_emg is True:
                EMG_list = EMG[:, -1:].reshape(EMG.shape[0])
                EMG_list = EMG_list.tolist()
                OSC_client.send_message("/EMG/processed", EMG_list)

            if len(np.array(data['imu']).shape) == 3:
                if get_accel is True:
                    for i in range(accel_proc.shape[0]):
                        accel_list = accel_proc[i, :, -1:].reshape(3)
                        accel_list = accel_list.tolist()
                        OSC_client.send_message(f"/accel/{i}", accel_list)
                if get_gyro is True:
                    for i in range(gyro_proc.shape[0]):
                        gyro_list = gyro_proc[i, :, -1:].reshape(3)
                        gyro_list = gyro_list.tolist()
                        OSC_client.send_message(f"/gyro/{i}", gyro_list)
            else:
                if get_accel is True:
                    accel_list = accel_proc[:, -1:].reshape(accel_proc.shape[0])
                    accel_list = accel_list.tolist()
                    OSC_client.send_message("/accel/processed", accel_list)
                if get_gyro is True:
                    gyro_list = gyro_proc[:, -1:].reshape(gyro_proc.shape[0])
                    gyro_list = gyro_list.tolist()
                    OSC_client.send_message("/gyro/processed", gyro_list)

        if save_data is True:
            if count == 0:
                print("Save data starting.")
                count += 1
            for key in data.keys():
                if key == 'imu':
                    if len(np.array(data['imu']).shape) == 3:
                        data[key] = np.array(data[key])
                        data['accel_proc'] = data[key][:n_electrode, :3, :]
                        data['gyro_proc'] = data[key][n_electrode:, 3:6, :]
                    else:
                        data[key] = np.array(data[key])
                        data['accel_proc'] = data[key][:n_electrode, :]
                        data['gyro_proc'] = data[key][n_electrode:, :]
                else:
                    data[key] = np.array(data[key])
            add_data_to_pickle(data, data_path)


