import scipy.io as sio
from time import time, sleep
from client import Client
import numpy as np

if __name__ == '__main__':
    while True:
        # Stuff to put in vibration code to get raw EMG data from server.
        # Sleep function will have to be done somewhere to be sure to run at the asked frequency

        # Set program variables
        read_freq = 100
        host_ip = 'localhost'
        host_port = 50001  # be sure this is the same than port_2 of the server
        n_electrodes = 5
        client = Client(host_ip, host_port, "TCP")
        data = client.get_data(data=["emg"], Nmhe=read_freq, exp_freq=read_freq, nb_of_data=1, raw=True, norm_emg=False)
        tic = time()
        raw_emg = np.array(data['raw_emg'])
        raw_emg = raw_emg[:n_electrodes, :]
        print(raw_emg)

