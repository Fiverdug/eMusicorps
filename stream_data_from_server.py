import scipy.io as sio
from client import Client
import numpy as np

# Set program variables
read_freq = 100  # Be sure that it's the same than server read frequency
n_electrode = 2
show_data = ["raw_emg"]  # can be ["emg"] to show process EMG
device_host = "192.168.1.211"  # IP address of computer which run trigno device

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
muscles_idx = (0, n_electrode - 1)
host_ip = 'localhost'
host_port = 50000
get_accel = True
get_gyro = True
get_emg = True
print_data = True
OSC_ip = "127.0.0.1",
OSC_port = 51337,
OSC_stream = False
if OSC_stream is True:
    from pythonosc.udp_client import SimpleUDPClient
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
    accel_proc = np.array(data['imu'])[:, :3, :]
    gyro_proc = np.array(data['imu'])[:, 3:6, :]
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
            OSC_client.send_message("/EMG/processed/", EMG[:, -1:])
        if get_accel is True:
            OSC_client.send_message("/accel/", accel_proc[:, :, -1:])
        if get_gyro is True:
            OSC_client.send_message("/gyro/", gyro_proc[:, :, -1:])
