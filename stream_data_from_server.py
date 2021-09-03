import scipy.io as sio
from client import Client
import numpy as np
from time import time
from pythonosc.udp_client import SimpleUDPClient

if __name__ == '__main__':
    # Set program variables
    read_freq = 100  # Be sure that it's the same than server read frequency
    show_data = ["raw_emg"]  # can be ["emg"] to show process EMG
    device_host = "192.168.1.211"  # IP address of computer which run trigno device

    # load MVC data from previous trials.
    file_name = "MVC_xxxx.mat"
    file_dir = "MVC_01_08_2021"
    list_mvc = sio.loadmat(f"{file_dir}/{file_name}")["MVC_list_max"]
    n_electrode = 5
    list_mvc = list_mvc[:, :n_electrode].T
    list_mvc = list_mvc.tolist()

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_accel"

    # Run streaming data
    # muscles_idx = (0, n_electrode - 1)
    host_ip = 'localhost'
    host_port = 50000
    get_accel = True
    get_gyro = True
    get_emg = True
    print_data = True
    OSC_ip = "127.0.0.1"
    OSC_port = 51337
    OSC_stream = True
    if OSC_stream is True:
        OSC_client = SimpleUDPClient(OSC_ip, OSC_port)
        print("Streaming OSC activated")

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

        EMG = np.array(data['emg'])
        raw_emg = np.array(data['raw_emg'])

        if len(np.array(data['imu']).shape) == 3:
            accel_proc = np.array(data['imu'])[:, :3, :]
            gyro_proc = np.array(data['imu'])[:, 3:6, :]
            raw_accel = np.array(data['raw_imu'])[:, :3, :]
            raw_gyro = np.array(data['raw_imu'])[:, 3:6, :]
        else:
            accel_proc = np.array(data['imu'])[:n_electrode, :]
            gyro_proc = np.array(data['imu'])[:n_electrode, :]
            raw_accel = np.array(data['raw_imu'])[:, :3, :]
            raw_gyro = np.array(data['raw_imu'])[:, 3:6, :]

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
                # OSC_client.send_message("/EMG/processed/", emg_proc[:, -1:].tolist())
                EMG_list = EMG[:, -1:].reshape(EMG.shape[0])
                EMG_list = EMG_list.tolist()
                # float_array = EMG[:, -1:].astype(np.float)
                # OSC_client.send_message("/EMG/processed",
                #                         [float(float_array[0]), float(float_array[1]), float(float_array[2]),
                #                          float(float_array[3]), float(float_array[4])])
                OSC_client.send_message("/EMG/processed", EMG_list)

            if len(np.array(data['imu']).shape) == 3:
                if get_accel is True:
                    # OSC_client.send_message("/accel/", accel_proc[:, :, -1:].tolist())
                    # accel_float = accel_proc[:, :, -1:].astype(np.float)
                    # OSC_client.send_message("/accel/0",
                    #                         [float(accel_float[0][0]), float(accel_float[0][1]), float(accel_float[0][2])])
                    # OSC_client.send_message("/accel/1",
                    #                         [float(accel_float[1][0]), float(accel_float[1][1]), float(accel_float[1][2])])
                    # OSC_client.send_message("/accel/2",
                    #                         [float(accel_float[2][0]), float(accel_float[2][1]), float(accel_float[2][2])])
                    # OSC_client.send_message("/accel/3",
                    #                         [float(accel_float[3][0]), float(accel_float[3][1]), float(accel_float[3][2])])
                    # OSC_client.send_message("/accel/4",
                    #                         [float(accel_float[4][0]), float(accel_float[4][1]), float(accel_float[4][2])])
                    for i in range(accel_proc.shape[0]):
                        accel_list = accel_proc[i, :, -1:].reshape(3)
                        accel_list = accel_list.tolist()
                        OSC_client.send_message(f"/accel/{i}", accel_list)

                if get_gyro is True:
                    # OSC_client.send_message("/gyro/", gyro_proc[:, :, -1:].tolist())
                    # gyro_float = gyro_proc[:, :, -1:].astype(np.float)
                    # OSC_client.send_message("/gyro/0",
                    #                         [float(gyro_float[0][0]), float(gyro_float[0][1]), float(gyro_float[0][2])])
                    # OSC_client.send_message("/gyro/1",
                    #                         [float(gyro_float[1][0]), float(gyro_float[1][1]), float(gyro_float[1][2])])
                    # OSC_client.send_message("/gyro/2",
                    #                         [float(gyro_float[2][0]), float(gyro_float[2][1]), float(gyro_float[2][2])])
                    # OSC_client.send_message("/gyro/3",
                    #                         [float(gyro_float[3][0]), float(gyro_float[3][1]), float(gyro_float[3][2])])
                    # OSC_client.send_message("/gyro/4",
                    #                         [float(gyro_float[4][0]), float(gyro_float[4][1]), float(gyro_float[4][2])])

                    for i in range(gyro_proc.shape[0]):
                        gyro_list = gyro_proc[i, :, -1:].reshape(3)
                        gyro_list = gyro_list.tolist()
                        OSC_client.send_message(f"/gyro/{i}", gyro_list)

            else:
                # OSC_client.send_message("/accel/processed/", emg_proc[:, -1:].tolist())
                # accel_float = accel_proc[:, -1:].astype(np.float)
                # OSC_client.send_message("/accel/processed",
                #                         [float(accel_float[0]), float(accel_float[1]), float(accel_float[2]),
                #                          float(accel_float[3]), float(accel_float[4])])
                if get_accel is True:
                    accel_list = accel_proc[:, -1:].reshape(accel_proc.shape[0])
                    accel_list = accel_list.tolist()
                    OSC_client.send_message("/accel/processed", accel_list)

                # OSC_client.send_message("/gyro/processed/", emg_proc[:, -1:].tolist())
                # gyro_float = gyro_proc[:, -1:].astype(np.float)
                # OSC_client.send_message("/gyro/processed",
                #                         [float(gyro_float[0]), float(gyro_float[1]), float(gyro_float[2]),
                #                          float(gyro_float[3]), float(gyro_float[4])])
                if get_gyro is True:
                    gyro_list = gyro_proc[:, -1:].reshape(gyro_proc.shape[0])
                    gyro_list = gyro_list.tolist()
                    OSC_client.send_message("/gyro/processed", gyro_list)


