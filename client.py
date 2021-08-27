import socket, pickle
import json

Buff_size = 1000000


class Message:
    def __init__(self):
        self.command = []
        self.dic = {}
        self.dic["command"] = []
        self.dic["Nmhe"] = 7
        self.dic["exp_freq"] = 33  # frequency at which the ocp should run
        self.dic["EMG_windows"] = 2000
        self.dic["get_names"] = False
        self.dic["nb_of_data"] = None
        # self.dic["EMG_unit"] = "V"  # or "mV"


class Client:
    def __init__(self, server_address, port, type="TCP", name=None):
        self.name = name if name is not None else "Client"
        self.type = type
        self.address = f"{server_address}:{port}"
        if self.type == "TCP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif self.type == "UDP":
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raise RuntimeError(f"Invalid type of connexion ({ self.type}). Type must be 'TCP' or 'UDP'.")
        self.client.connect((server_address, port))

    def connect(self, server_address, port):
        self.client.connect((server_address, port))

    @staticmethod
    def client_sock(type):
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # TODO: add possibility to ask for some index

    def get_data(
        self,
        data,
        Nmhe=7,
        exp_freq=33,
        EMG_wind=2000,
        nb_of_data=None,
        buff=Buff_size,
        get_kalman=False,
        get_names=False,
        mvc_list=None,
        norm_emg=None,
        raw=False
    ):

        message = Message()
        message.dic["get_names"] = get_names
        # norm_EMG = True if MVC is not None and norm_EMG is None else False
        message.dic["norm_emg"] = norm_emg
        message.dic["mvc_list"] = mvc_list
        message.dic["kalman"] = get_kalman
        message.dic["Nmhe"] = Nmhe
        message.dic["exp_freq"] = exp_freq
        message.dic["EMG_windows"] = EMG_wind
        message.dic["nb_of_data"] = nb_of_data
        message.dic["raw_data"] = raw

        if norm_emg is True and mvc_list is None:
            raise RuntimeError("Define a list of MVC to normalize the EMG data.")
        elif mvc_list is not None and norm_emg is not True:
            print("[WARNING] You have defined a list of MVC but not asked for normalization. " 
                  "Please turn norm_EMG to True tu normalize your data."
                  )

        # message.dic["EMG_unit"] = EMG_unit
        # if len(data) == 0:
        #     raise RuntimeError("Please set at least one data to get.")
        message.dic["command"] = []
        for i in data:
            message.dic["command"].append(i)
            if i != "emg" and i != "markers" and i != "imu" and i != "force_plate":
                raise RuntimeError(f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' ")

        Client.send(self, json.dumps(message.dic).encode(), type="all")
        data = json.loads(self.client.recv(buff))
        return data

    def send(self, data, type=None, IP=None, port=None):
        data = pickle.dumps(data) if not isinstance(data, bytes) else data
        if type == None:
            return self.client.send(data)
        elif type == "all":
            return self.client.sendall(data)
        elif type == "to":
            return self.client.sendto(data, address=(IP, port))

    def close(self):
        return self.client.close


if __name__ == "__main__":
    from time import time, sleep

    host_ip = "192.168.1.211"
    # host_ip = 'localhost'
    host_port = 50000

    tic = time()
    client = Client(host_ip, host_port, "TCP")
    data = client.get_data(
        data=["markers"], Nmhe=7, exp_freq=33, EMG_wind=2000, nb_of_data=8, get_names=True, get_kalman=True
    )
    print(data["kalman"])
    print(time() - tic)
