from client import Client
import numpy as np

if __name__ == '__main__':
    while True:
        # Stuff to put in vibration code to get raw EMG data from server.

        # Set program variables
        read_freq = 100
        host_ip = 'localhost'
        host_port = 50001  # be sure this is the same than port_2 of the server
        n_electrodes = 5
        client = Client(host_ip, host_port, "TCP")
        data = client.get_data(data=["emg"], Nmhe=read_freq, exp_freq=read_freq, nb_of_data=1, raw=True, norm_emg=False)
        raw_emg = np.array(data['raw_emg'])
        raw_emg = raw_emg[:n_electrodes, :]
        print(raw_emg)

