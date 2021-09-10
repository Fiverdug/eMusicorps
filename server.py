import socket
import pytrigno
import sys
from time import time, sleep, strftime
import scipy.io as sio
import numpy as np
from data_plot import init_plot_emg, update_plot_emg
from data_processing import process_emg_rt, process_IMU, add_data_to_pickle
import multiprocessing as mp
import os
import json

try:
    import biorbd
    biorbd_eigen = True
except ModuleNotFoundError:
    biorbd_eigen = False

try:
    import biorbd_casadi as biorbd
    biorbd_cas = True
except ModuleNotFoundError:
    biorbd_cas = False

# biorbd_pack = True if biorbd_cas or biorbd_eigen else False
biorbd_pack = False

try:
    from vicon_dssdk import ViconDataStream as VDS
except ModuleNotFoundError:
    pass
Buff_size = 100000


class Server:
    def __init__(
        self,
        IP,
        server_ports,
        type=None,
        ocp_freq=100,
        mhe_size=100,
        system_rate=100,
        emg_rate=2000,
        imu_rate=148.1,
        mark_rate=100,
        emg_windows=2000,
        mark_windows=100,
        imu_windows=100,
        recons_kalman=True,
        model_path=None,
        proc_emg=True,
        proc_imu=True,
        mark_dec=4,
        emg_dec=8,
        timeout=-1,
        buff_size=Buff_size,
        device=None,  # 'vicon' or 'pytrigno',
        device_host_ip=None,
        muscle_range=None,
        output_file=None,
        output_dir=None,
        save_data=True
    ):

        # problem variables
        self.IP = IP
        if isinstance(server_ports, list):
            self.ports = server_ports
        else:
            self.ports = [server_ports]
        self.type = type if type is not None else "TCP"
        self.timeout = timeout if timeout != -1 else 10000
        self.system_rate = system_rate
        self.emg_rate = emg_rate
        self.imu_rate = imu_rate
        self.mark_rate = mark_rate
        self.emg_windows = emg_windows
        self.mark_windows = mark_windows
        self.imu_windows = imu_windows
        self.recons_kalman = recons_kalman
        self.model_path = model_path
        self.proc_emg = proc_emg
        self.proc_imu = proc_imu
        self.mark_dec = mark_dec
        self.emg_dec = emg_dec
        self.buff_size = buff_size
        self.Nmhe = mhe_size
        self.ocp_freq = ocp_freq
        self.try_wt_vicon = ()
        self.save_data = save_data
        self.raw_data = False
        self.try_w_connection = True
        if biorbd_pack:
            self.model = biorbd.Model(self.model_path)
        else:
            self.model = ()
        self.device = device if device else "pytrigno"
        self.device_host_ip = device_host_ip if device_host_ip else "localhost"

        current_time = strftime("%Y%m%d-%H%M")
        output_file = output_file if output_file else f"trigno_streaming_{current_time}"

        output_dir = output_dir if output_dir else "live_data"

        if os.path.isdir(output_dir) is not True:
            os.mkdir(output_dir)

        if os.path.isfile(f"{output_dir}/{output_file}"):
            os.remove(f"{output_dir}/{output_file}")

        self.data_path = f"{output_dir}/{output_file}"

        # Init some variables
        self.plot_emg = ()
        self.mvc_list = ()
        self.norm_min_accel_value = ()
        self.norm_max_accel_value = ()
        self.norm_min_gyro_value = ()
        self.norm_max_gyro_value = ()
        self.norm_emg = None
        self.optim = False
        self.stream_emg, self.stream_markers, self.stream_imu, self.stream_force_plate = False, False, False, False
        self.subject_name, self.device_name = None, None
        self.iter = 0
        self.marker_names = ()
        self.nb_of_data = self.Nmhe
        self.emg_empty, self.markers_empty = (), ()
        self.nb_emg, self.nb_marks = 0, 0
        self.nb_electrodes = muscle_range[1] - muscle_range[0] + 1
        self.emg_sample = 0
        self.imu_sample = 0
        self.muscle_range = muscle_range if muscle_range else (0, 15)
        self.imu_range = (self.muscle_range[0], self.muscle_range[0] + (self.nb_electrodes * 9))
        self.output_names = ()
        self.imu_output_names = ()
        self.emg_names = None
        self.imu_names = None

        # Multiprocess stuff
        manager = mp.Manager()
        self.server_queue = []
        for i in range(len(self.ports)):
            self.server_queue.append(manager.Queue())
        self.emg_queue_in = manager.Queue()
        self.emg_queue_out = manager.Queue()
        self.imu_queue_in = manager.Queue()
        self.imu_queue_out = manager.Queue()
        self.kin_queue_in = manager.Queue()
        self.kin_queue_out = manager.Queue()
        self.event_emg = mp.Event()
        self.event_kin = mp.Event()
        self.event_imu = mp.Event()
        self.event_vicon = mp.Event()
        self.process = mp.Process
        self.servers = []
        self.count_server = 0

    @staticmethod
    def __server_sock(type):
        if type == "TCP" or type is None:
            return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        elif type == "UDP":
            return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def accept(self):
        return self.server.accept()

    def run(
        self,
        stream_emg=True,
        stream_markers=True,
        stream_imu=False,
        stream_force_plate=False,
        norm_emg=True,
        optim=False,
        plot_emg=False,
        mvc_list=None,
        norm_min_accel_value=None,
        norm_max_accel_value=None,
        norm_min_gyro_value=None,
        norm_max_gyro_value=None,
        subject_name=None,
        emg_device_name=None,
        imu_device_name=None,
        test_with_connection=True
    ):

        self.device_name = emg_device_name
        self.imu_device_name = imu_device_name
        self.plot_emg = plot_emg
        self.mvc_list = mvc_list
        self.norm_emg = norm_emg
        if self.norm_emg is True and self.mvc_list is None:
            raise RuntimeError("Please define a MVC list to normalize EMG or turn 'norm_emg' to False")
        self.optim = optim
        self.stream_emg = stream_emg
        self.stream_markers = stream_markers
        self.stream_imu = stream_imu
        self.stream_force_plate = stream_force_plate
        self.subject_name = subject_name
        self.norm_min_accel_value = norm_min_accel_value
        self.norm_max_accel_value = norm_max_accel_value
        self.norm_max_gyro_value = norm_max_gyro_value
        self.norm_min_gyro_value = norm_min_gyro_value
        self.try_w_connection = test_with_connection
        if self.try_w_connection is not True:
            print("[Warning] Debug mode without vicon connection.")

        data_type = []
        if self.stream_emg:
            data_type.append("emg")
        if self.stream_markers or self.recons_kalman:
            data_type.append("markers")
        if self.stream_imu:
            data_type.append("imu")
        if self.stream_force_plate:
            raise RuntimeError("Not implemented yet")

        self.imu_sample = int(self.imu_rate / self.system_rate)
        if self.try_w_connection is not True:
            data_exp = sio.loadmat("test_wt_connection.mat")
            self.IM_exp = np.concatenate(
                (data_exp["raw_accel"][:self.nb_electrodes, :, :1000],
                 data_exp["raw_gyro"][:self.nb_electrodes, :, :1000]), axis=1
            )
            self.system_rate = self.system_rate
            self.emg_sample = int(self.emg_rate / self.system_rate)
            self.imu_sample = int(self.imu_rate / self.system_rate)
            self.emg_exp = data_exp["raw_EMG"][:self.nb_electrodes, :]
            if "markers" in data_exp.keys():
                self.markers_exp = data_exp["markers"]
                self.markers_exp = np.nan_to_num(self.markers_exp)
            self.init_frame = 500
            self.last_frame = 1700
            self.m = self.init_frame
            self.c = self.init_frame * 20
            self.marker_names = []

        # Start connexion
        for i in range(len(self.ports)):
            if self.type == "TCP":
                self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            elif self.type == "UDP":
                self.servers.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
            else:
                raise RuntimeError(f"Invalid type of connexion ({type}). Type must be 'TCP' or 'UDP'.")
            try:
                self.servers[i].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.servers[i].bind((self.IP, self.ports[i]))
                if self.type != "UDP":
                    self.servers[i].listen(10)
                    self.inputs = [self.servers[i]]
                    self.outputs = []
                    self.message_queues = {}

            except ConnectionError:
                raise RuntimeError("Unknown error. Server is not listening.")

        # if self.try_w_connection:
        #     if self.device == "vicon":
        #         self._init_vicon_client()
        #     else:
        #         self._init_pytrigno()
        if self.try_w_connection:
            if self.device == "pytrigno":
                self._init_pytrigno()
        open_server = []
        save_data = self.process(name="vicon", target=Server.save_vicon_data, args=(self,))
        save_data.start()
        for i in range(len(self.ports)):
            open_server.append(self.process(name="listen"+f"_{i}", target=Server.open_server, args=(self,)))
            open_server[i].start()
            self.count_server += 1
        if self.stream_emg:
            mproc_emg = self.process(name="process_emg", target=Server.emg_processing, args=(self,))
            mproc_emg.start()
        if self.stream_imu:
            mproc_imu = self.process(name="process_imu", target=Server.imu_processing, args=(self,))
            mproc_imu.start()
        if self.stream_markers:
            mproc_kin = self.process(name="kin", target=Server.recons_kin, args=(self,))
            mproc_kin.start()
        save_data.join()
        for i in range(len(self.ports)):
            open_server[i].join()
            self.count_server += 1
        if self.stream_emg:
            mproc_emg.join()
        if self.stream_markers:
            mproc_kin.join()
        if self.stream_imu:
            mproc_imu.join()

    def open_server(self):
        server_idx = self.count_server
        print(f"{self.type} server {server_idx} is listening on '{self.IP}:{self.ports[server_idx]}' "
              f"and waiting for a client.")
        while 1:
            connection, ad = self.servers[server_idx].accept()
            if self.optim is not True:
                print(f"new connection from {ad}")
            if self.type == "TCP":
                message = json.loads(connection.recv(self.buff_size))
                if self.optim is not True:
                    print(f"client sended {message}")
                data_queue = {}

                while len(data_queue) == 0:
                    try:
                        data_queue = self.server_queue[server_idx].get_nowait()
                        is_working = True
                    except:
                        is_working = False
                        pass
                    if is_working:
                        self.system_rate = data_queue["system_rate"]
                        self.Nmhe = message["Nmhe"]
                        self.ocp_freq = message["exp_freq"]
                        self.raw_data = message["raw_data"]
                        norm_emg = message['norm_emg']
                        mvc_list = message["mvc_list"]
                        self.nb_of_data = (
                            message["nb_of_data"] if message["nb_of_data"] is not None else self.nb_of_data
                        )
                        if self.system_rate < self.ocp_freq:
                            ratio = 1
                        else:
                            ratio = int(self.system_rate / self.ocp_freq)
                        nb_data_ocp_duration = int(ratio * (self.Nmhe + 1))
                        data_to_prepare = {}

                        if len(message["command"]) != 0:
                            for i in message["command"]:
                                if i == "emg":
                                    if self.stream_emg:
                                        if self.raw_data:
                                            raw_emg = data_queue["raw_emg"]
                                            data_to_prepare["raw_emg"] = raw_emg
                                        emg = data_queue["emg_proc"]
                                        if norm_emg:
                                            if isinstance(mvc_list, np.ndarray) is True:
                                                if len(mvc_list.shape) == 1:
                                                    quot = mvc_list.reshape(-1, 1)
                                                else:
                                                    quot = mvc_list
                                            else:
                                                quot = np.array(mvc_list).reshape(-1, 1)
                                        else:
                                            quot = [1]
                                        data_to_prepare["emg"] = emg / quot
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")
                                elif i == "markers":
                                    if self.stream_markers:
                                        markers = data_queue["markers"]
                                        data_to_prepare["markers"] = markers
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")

                                elif i == "imu":
                                    if self.stream_imu:
                                        if self.raw_data:
                                            raw_imu = data_queue["raw_imu"]
                                            data_to_prepare["raw_imu"] = raw_imu
                                        imu = data_queue["imu_proc"]
                                        data_to_prepare["imu"] = imu
                                    else:
                                        raise RuntimeError(f"Data you asking for ({i}) is not streaming")

                                elif i == "force plate":
                                    raise RuntimeError("Not implemented yet.")
                                else:
                                    raise RuntimeError(
                                        f"Unknown command '{i}'. Command must be :'emg', 'markers' or 'imu' "
                                    )

                        if message["kalman"] is True:
                            if self.recons_kalman:
                                angle = data_queue["kalman"]
                                data_to_prepare["kalman"] = angle
                            else:
                                raise RuntimeError(
                                    f"Kalman reconstruction is not activate. "
                                    f"Please turn server flag recons_kalman to True."
                                )

                        # prepare data
                        dic_to_send = self.prepare_data(data_to_prepare, nb_data_ocp_duration, ratio)

                        if message["get_names"] is True:
                            dic_to_send["marker_names"] = data_queue["marker_names"]
                            dic_to_send["emg_names"] = data_queue["emg_names"]

                        if self.optim is not True:
                            print("Sending data to client...")
                            print(f"data sended : {dic_to_send}")
                        encoded_data = json.dumps(dic_to_send).encode()

                        try:
                            connection.send(encoded_data)
                        except:
                            pass

                        if self.optim is not True:
                            print(f"Data of size {sys.getsizeof(dic_to_send)} sent to the client.")

    def prepare_data(self, data_to_prep, nb_data_ocp_duration, ratio):
        for key in data_to_prep.keys():
            nb_of_data = self.nb_of_data
            if len(data_to_prep[key].shape) == 2:
                data_to_prep[key] = data_to_prep[key][:, -nb_data_ocp_duration:]
                if self.raw_data is not True or key != 'raw_emg':
                    data_to_prep[key] = data_to_prep[key][:, ::ratio]
                if self.raw_data is True and key == 'raw_emg':
                    nb_of_data = self.emg_sample
                data_to_prep[key] = data_to_prep[key][:, -nb_of_data:].tolist()
            elif len(data_to_prep[key].shape) == 3:
                data_to_prep[key] = data_to_prep[key][:, :, -nb_data_ocp_duration:]
                if self.raw_data is not True or key != 'raw_imu':
                    data_to_prep[key] = data_to_prep[key][:, :, ::ratio]
                if self.raw_data is True and key == 'raw_imu':
                    nb_of_data = self.imu_sample
                data_to_prep[key] = data_to_prep[key][:, :, -nb_of_data:].tolist()
        return data_to_prep

    def _init_pytrigno(self):
        if self.stream_emg:
            self.emg_sample = int(self.emg_rate / self.system_rate)
            if self.norm_emg is True and len(self.mvc_list) != self.nb_electrodes:
                raise RuntimeError(
                    f"Length of the mvc list ({self.mvc_list}) " f"not consistent with emg number ({self.nb_electrodes})."
                )
            self.dev_emg = pytrigno.TrignoEMG(
                channel_range=self.muscle_range, samples_per_read=self.emg_sample, host=self.device_host_ip
            )
            self.dev_emg.start()

        if self.stream_imu:
            self.imu_sample = int(self.imu_rate / self.system_rate)

            self.dev_imu = pytrigno.TrignoIM(
                channel_range=self.imu_range, samples_per_read=self.imu_sample, host=self.device_host_ip
            )
            self.dev_imu.start()

    def _init_vicon_client(self):
        address = f"{self.device_host_ip}:801"
        print(f"Connection to ViconDataStreamSDK at : {address} ...")
        self.vicon_client = VDS.Client()
        self.vicon_client.Connect(address)
        self.vicon_client.EnableSegmentData()
        self.vicon_client.EnableDeviceData()
        self.vicon_client.EnableMarkerData()
        self.vicon_client.EnableUnlabeledMarkerData()

        a = self.vicon_client.GetFrame()
        while a is not True:
            a = self.vicon_client.GetFrame()

        system_rate = self.vicon_client.GetFrameRate()
        if system_rate != self.system_rate:
            print(
                f"[WARNING] Vicon system rate ({system_rate} Hz) is different than system rate you chosen "
                f"({self.system_rate} Hz). System rate is now set to : {system_rate} Hz."
            )
            self.system_rate = system_rate

        if self.stream_emg:
            self.device_name = self.device_name if self.device_name else self.vicon_client.GetDeviceNames()[2][0]
            self.device_info = self.vicon_client.GetDeviceOutputDetails(self.device_name)
            self.emg_sample = int(self.emg_rate / self.system_rate)
            self.emg_empty = np.zeros((len(self.device_info), self.emg_sample))
            # self.output_names, self.emg_names = self.get_emg(init=True)
            # self.nb_emg = len(self.output_names)
            if self.norm_emg is True and len(self.mvc_list) != self.nb_electrodes:
                raise RuntimeError(
                    f"Length of the mvc list ({self.mvc_list}) " f"not consistent with emg number ({self.nb_electrodes})."
                )

        if self.stream_imu:
            self.imu_device_name = self.imu_device_name if self.imu_device_name else self.vicon_client.GetDeviceNames()[3][0]
            self.imu_device_info = self.vicon_client.GetDeviceOutputDetails(self.imu_device_name)
            self.imu_sample = int(self.imu_rate / self.system_rate)
            self.imu_empty = np.zeros((144, self.imu_sample))
            # self.imu_output_names, self.imu_names = self.get_imu(init=True)
            # self.nb_imu = len(self.imu_output_names)

        if self.stream_markers:
            self.subject_name = self.subject_name if self.subject_name else self.vicon_client.GetSubjectNames()[0]
            self.vicon_client.EnableMarkerData()
            self.vicon_client.EnableUnlabeledMarkerData()
            self.vicon_client.EnableMarkerRayData()
            self.marker_names = self.vicon_client.GetMarkerNames(self.subject_name)
            self.markers_empty = np.ndarray((3, len(self.marker_names), 1))

    def _loop_sleep(self, delta_tmp, delta, tic, print_time=False):
        delta = (delta_tmp + delta) / 2
        toc = time() - tic
        time_to_sleep = (1 / self.system_rate) - toc - delta
        if time_to_sleep > 0:
            sleep(time_to_sleep)
        loop_duration = time() - tic
        delta_tmp = loop_duration - 1 / self.system_rate
        if print_time is True:
            toc = time() - tic
            print(toc)
        return delta, delta_tmp

    def emg_processing(self):
        # self.event_vicon.wait()
        c = 0
        while True:
            try:
                emg_data = self.emg_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False
                pass

            if is_working:
                if self.try_w_connection is not True:
                    if c < self.emg_exp.shape[1]:
                        emg_tmp = self.emg_exp[: self.nb_electrodes, c: c + self.emg_sample]
                        c += self.emg_sample
                    else:
                        c = 0
                else:
                    emg_tmp = emg_data["emg_tmp"]
                ma_win = 200
                raw_emg, emg_proc = emg_data["raw_emg"], emg_data["emg_proc"]
                raw_emg, emg_proc = process_emg_rt(
                    raw_emg,
                    emg_proc,
                    emg_tmp,
                    mvc_list=self.mvc_list,
                    ma_win=ma_win,
                    emg_win=self.emg_windows,
                    emg_freq=self.emg_rate,
                    norm_emg=self.norm_emg,
                    lpf=False,
                )
                self.emg_queue_out.put({"raw_emg": raw_emg, "emg_proc": emg_proc})
                self.event_emg.set()

    def imu_processing(self):
        d = 0
        while True:
            try:
                imu_data = self.imu_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                if self.try_w_connection is not True:
                    if d < self.IM_exp.shape[2]:
                        imu_tmp = self.IM_exp[:self.nb_electrodes, :, d: d + self.imu_sample]
                        d += self.imu_sample
                    else:
                        d = 0
                else:
                    imu_tmp = imu_data["imu_tmp"]

                accel_tmp = imu_tmp[:, :3, :]
                gyro_tmp = imu_tmp[:, 3:6, :]
                if self.device == 'vicon':
                    # convert rad/s into deg/s when vicon is used
                    gyro_tmp = gyro_tmp * (180 / np.pi)

                if self.device == 'pytrigno':
                    # convert data from G into m/s2 when pytrigno is used
                    accel_tmp = accel_tmp * 9.81

                raw_imu, imu_proc = imu_data["raw_imu"], imu_data["imu_proc"]
                if len(raw_imu) != 0:
                    if len(imu_proc.shape) == 3:
                        raw_accel, accel_proc = raw_imu[:self.nb_electrodes, :3, :], imu_proc[:self.nb_electrodes, :3, :]
                        raw_gyro, gyro_proc = raw_imu[:self.nb_electrodes, 3:6, :], imu_proc[:self.nb_electrodes, 3:6, :]
                    else:
                        raw_accel, accel_proc = raw_imu[:self.nb_electrodes, :3, :], imu_proc[:self.nb_electrodes, :]
                        raw_gyro, gyro_proc = raw_imu[:self.nb_electrodes, 3:6, :], imu_proc[:self.nb_electrodes, :]
                else:
                    raw_accel, accel_proc = raw_imu, imu_proc
                    raw_gyro, gyro_proc = raw_imu, imu_proc

                raw_accel, accel_proc = process_IMU(
                    accel_proc,
                    raw_accel,
                    accel_tmp,
                    self.imu_windows,
                    self.imu_sample,
                    ma_win=30,
                    accel=True,
                    norm_min_bound=self.norm_min_accel_value,
                    norm_max_bound=self.norm_max_accel_value,
                    squared=True,
                )
                raw_gyro, gyro_proc = process_IMU(
                    gyro_proc,
                    raw_gyro,
                    gyro_tmp,
                    self.imu_windows,
                    self.imu_sample,
                    ma_win=30,
                    norm_min_bound=self.norm_min_gyro_value,
                    norm_max_bound=self.norm_max_gyro_value,
                    squared=True,
                )
                if len(accel_proc.shape) == 3:
                    raw_imu, imu_proc = np.concatenate((raw_accel, raw_gyro), axis=1), np.concatenate((accel_proc, gyro_proc), axis=1)
                else:
                    raw_imu, imu_proc = np.concatenate((raw_accel, raw_gyro), axis=1), np.concatenate(
                        (accel_proc, gyro_proc), axis=0)
                self.imu_queue_out.put({"raw_imu": raw_imu, "imu_proc": imu_proc})
                self.event_imu.set()

    def recons_kin(self):
        while True:
            try:
                markers_data = self.kin_queue_in.get_nowait()
                is_working = True
            except:
                is_working = False

            if is_working:
                markers = markers_data["markers"]
                states = markers_data["states"]
                model = self.model
                if self.try_w_connection:
                    markers_tmp, self.marker_names, occluded = self.get_markers()
                    if self.iter > 0:
                        for i in range(markers_tmp.shape[1]):
                            if occluded[i] is True:
                                markers_tmp[:, i, :] = markers[:, i, -1:]
                else:
                    markers_tmp = self.markers_exp[:, :, self.m : self.m + 1] * 0.001
                    self.m = self.m + 1 if self.m < self.last_frame else self.init_frame
                    for i in range(markers_tmp.shape[1]):
                        if np.product(markers_tmp[:, i, :] * 0.001) == 0:
                            markers_tmp[:, i, :] = markers[:, i, -1:]

                if len(markers) == 0:
                    markers = markers_tmp
                markers = (
                    markers
                    if markers.shape[2] < self.mark_windows
                    else np.append(markers[:, :, 1:], markers_tmp, axis=2)
                )

                if self.recons_kalman is True:
                    states_tmp = self.kalman_func(markers[:, :, -1:], model)
                    if len(states) == 0:
                        states = states_tmp
                    states = (
                        states if states.shape[1] < self.mark_windows else np.append(states[:, 1:], states_tmp, axis=1)
                    )
                self.kin_queue_out.put({"states": states, "markers": markers})
                self.event_kin.set()

    def save_vicon_data(self):
        emg_dec = self.emg_dec
        mark_dec = self.mark_dec
        raw_emg = []
        raw_imu = []
        imu_proc = []
        emg_proc = []
        markers = []
        states = []

        if self.try_w_connection:
            if self.device == "vicon":
                self._init_vicon_client()
        self.nb_marks = len(self.marker_names)
        if self.plot_emg:
            p, win_emg, app, box = init_plot_emg(self.nb_electrodes)
        delta = 0
        delta_tmp = 0
        self.iter = 0
        dic_to_put = {}
        frame = False
        self.initial_time = time()
        while True:
            tic = time()
            if self.try_w_connection:
                if self.device == "vicon":
                    frame = self.vicon_client.GetFrame()
                    if frame is not True:
                        print("A problem occurred, no frame available.")
                elif self.device == "pytrigno":
                    frame = True
            else:
                frame = True

            if frame:
                if self.stream_emg:
                    if self.try_w_connection:
                        emg_tmp, emg_names = self.get_emg(emg_names=self.emg_names)
                    else:
                        emg_tmp = []
                        emg_names = []
                    self.emg_queue_in.put_nowait({"raw_emg": raw_emg, "emg_proc": emg_proc, "emg_tmp": emg_tmp})
                if self.stream_markers:
                    markers_tmp, self.marker_names, occluded = self.get_markers()
                    if self.iter > 0:
                        for i in range(markers_tmp.shape[1]):
                            if occluded[i] is True:
                                markers_tmp[:, i, :] = markers[:, i, -1:]
                    self.kin_queue_in.put_nowait({"states": states, "markers": markers, "model_path": self.model_path,
                                                  "markers_tmp": markers_tmp})
                if self.stream_imu:
                    if self.try_w_connection:
                        imu_tmp, imu_names = self.get_imu(imu_names=self.imu_names)
                    else:
                        imu_tmp, imu_names = [], []
                    self.imu_queue_in.put_nowait({"raw_imu": raw_imu, "imu_proc": imu_proc, "imu_tmp": imu_tmp})

                if self.stream_emg:
                    self.event_emg.wait()
                    emg_data = self.emg_queue_out.get_nowait()
                    self.event_emg.clear()
                    raw_emg, emg_proc = emg_data["raw_emg"], emg_data["emg_proc"]
                    dic_to_put["emg_names"] = emg_names
                    dic_to_put['raw_emg'] = raw_emg
                    dic_to_put['emg_proc'] = emg_proc

                if self.stream_markers:
                    self.event_kin.wait()
                    kin = self.kin_queue_out.get_nowait()
                    self.event_kin.clear()
                    states, markers = kin["states"], kin["markers"]
                    dic_to_put["markers"] = np.around(markers, decimals=mark_dec)
                    dic_to_put["kalman"] = states
                    dic_to_put["marker_names"] = self.marker_names

                if self.stream_imu:
                    self.event_imu.wait()
                    imu = self.imu_queue_out.get_nowait()
                    self.event_imu.clear()
                    raw_imu, imu_proc = imu["raw_imu"], imu["imu_proc"]
                    dic_to_put["imu_names"] = imu_names
                    dic_to_put['raw_imu'] = raw_imu
                    dic_to_put['imu_proc'] = imu_proc
            dic_to_put['system_rate'] = self.system_rate

            for i in range(len(self.ports)):
                try:
                    self.server_queue[i].get_nowait()
                except:
                    pass
                self.server_queue[i].put_nowait(dic_to_put)

            if self.plot_emg:
                update_plot_emg(raw_emg, p, app, box)

            if self.iter == 0:
                initial_time = time()
                print("Data start streaming")
            self.iter += 1

            # Save data
            if self.save_data is True:
                data_to_save = {
                    "Time": time() - initial_time,
                    "emg_freq": self.emg_rate,
                    "IM_freq": self.imu_rate,
                    "read_freq": self.system_rate,
                }
                if self.stream_emg is True:
                    data_to_save["emg_proc"] = emg_proc[:, -1:]
                    data_to_save["raw_emg"] = raw_emg[:, -self.emg_sample:]

                if self.stream_imu is True:
                    if imu_proc.shape == 3:
                        data_to_save["accel_proc"] = imu_proc[:, 0:3, -1:]
                        data_to_save["raw_accel"] = raw_imu[:, 0:3, -self.imu_sample:]
                        data_to_save["gyro_proc"] = imu_proc[:, 3:6, -1:]
                        data_to_save["raw_gyro"] = raw_imu[:, 3:6, -self.imu_sample:]
                    else:
                        data_to_save["accel_proc"] = imu_proc[:self.nb_electrodes, -1:]
                        data_to_save["raw_accel"] = raw_imu[:, 0:3, -self.imu_sample:]
                        data_to_save["gyro_proc"] = imu_proc[self.nb_electrodes:, -1:]
                        data_to_save["raw_gyro"] = raw_imu[:, 3:6, -self.imu_sample:]

                add_data_to_pickle(data_to_save, self.data_path)
            delta, delta_tmp = self._loop_sleep(delta_tmp, delta, tic)

    @staticmethod
    def init_kalman(model):
        freq = 100  # Hz
        params = biorbd.KalmanParam(freq)
        kalman = biorbd.KalmanReconsMarkers(model, params)

        Q = biorbd.GeneralizedCoordinates(model)
        Qdot = biorbd.GeneralizedVelocity(model)
        Qddot = biorbd.GeneralizedAcceleration(model)
        return kalman, Q, Qdot, Qddot

    def get_markers(self, mark_names=()):
        occluded = []
        markers = self.markers_empty
        subject_name = self.subject_name
        marker_names = mark_names if len(mark_names) != 0 else self.vicon_client.GetMarkerNames(subject_name)
        for i in range(len(marker_names)):
            if mark_names:
                markers[:, i, 0], occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    subject_name, marker_names[i]
                )
            else:
                markers[:, i, 0], occluded_tmp = self.vicon_client.GetMarkerGlobalTranslation(
                    subject_name, marker_names[i][0]
                )
                marker_names[i] = marker_names[i][0]
            occluded.append(occluded_tmp)
        return markers * 0.001, marker_names, occluded

    @staticmethod
    def get_force_plate(vicon_client):
        forceVectorData = []
        forceplates = vicon_client.GetForcePlates()
        for plate in forceplates:
            forceVectorData = vicon_client.GetForceVector(plate)
            momentVectorData = vicon_client.GetMomentVector(plate)
            copData = vicon_client.GetCentreOfPressure(plate)
            globalForceVectorData = vicon_client.GetGlobalForceVector(plate)
            globalMomentVectorData = vicon_client.GetGlobalMomentVector(plate)
            globalCopData = vicon_client.GetGlobalCentreOfPressure(plate)

            try:
                analogData = vicon_client.GetAnalogChannelVoltage(plate)
            except VDS.DataStreamException as e:
                print("Failed getting analog channel voltages")
        return forceVectorData

    def get_emg(self, emg_names=None):# init=False, output_names=None, emg_names=None):
        # output_names = [] if output_names is None else output_names
        names = [] if emg_names is None else emg_names
        if self.device == "vicon":
            emg = np.zeros((16, self.emg_sample))
            # if init is True:
            #     count = 0
            #     for output_name, emg_name, unit in self.device_info:
            #         emg[count, :], occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.device_name, output_name, emg_name
            #         )
            #         if np.mean(emg[count, -self.emg_sample :]) != 0:
            #             output_names.append(output_name)
            #             emg_names.append(emg_name)
            #         count += 1
            # else:
            #     for i in range(len(output_names)):
            #         emg[i, :], occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.device_name, output_names[i], emg_names[i]
            #         )
            count = 0
            for output_name, emg_name, unit in self.device_info:
                emg[count, :], occluded = self.vicon_client.GetDeviceOutputValues(
                    self.device_name, output_name, emg_name
                )
                if emg_names is None:
                    names.append(emg_name)
                count += 1
            emg = emg[:self.nb_electrodes, :]
        else:
            emg = self.dev_emg.read()

        # if init is True:
        #     return output_names, emg_names
        # else:
        return emg, names

    def get_imu(self, imu_names=None): #, init=False, output_names=None, imu_names=None):
        # output_names = [] if output_names is None else output_names
        names = [] if imu_names is None else imu_names
        if self.device == "vicon":
            imu = np.zeros((144, self.imu_sample))
            # if init is True:
            #     count = 0
            #     for output_name, imu_name, unit in self.imu_device_info:
            #         imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(
            #             self.imu_device_name, output_name, imu_name
            #         )
            #         imu[count, :] = imu_tmp[-self.imu_sample:]
            #         if np.mean(imu[count, :, -self.imu_sample:]) != 0:
            #             output_names.append(output_name)
            #             imu_names.append(imu_name)
            #         count += 1
            # else:
            count = 0
            for output_name, imu_name, unit in self.imu_device_info:
                imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(
                    self.imu_device_name, output_name, imu_name
                )
                if imu_names is None:
                    names.append(imu_name)
            # for i in range(self.nb_electrodes):
            #     imu_tmp, occluded = self.vicon_client.GetDeviceOutputValues(
            #         self.imu_device_name, output_names[i], imu_names[i]
            #     )
                imu[count, :] = imu_tmp[-self.imu_sample:]
                count += 1

            imu = imu[:self.nb_electrodes * 9, :]
            imu = imu.reshape(self.nb_electrodes, 9, -1)
        else:
            imu = self.dev_imu.read()
            imu = imu.reshape(self.nb_electrodes, 9, -1)

        return imu, names

    @staticmethod
    def kalman_func(markers, model):
        markers_over_frames = []
        freq = 100  # Hz
        params = biorbd.KalmanParam(freq)
        kalman = biorbd.KalmanReconsMarkers(model, params)

        q = biorbd.GeneralizedCoordinates(model)
        q_dot = biorbd.GeneralizedVelocity(model)
        qd_dot = biorbd.GeneralizedAcceleration(model)
        for i in range(markers.shape[2]):
            markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

        q_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        q_dot_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
        for i, targetMarkers in enumerate(markers_over_frames):
            kalman.reconstructFrame(model, targetMarkers, q, q_dot, qd_dot)
            q_recons[:, i] = q.to_array()
            q_dot_recons[:, i] = q_dot.to_array()
        return q_recons, q_dot_recons

