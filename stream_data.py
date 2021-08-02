from RT_process_Pytrigno import run
import scipy.io as sio

# Set program variables
read_freq = 74
n_electrode = 14
show_data = ["raw_emg"]  # can be ["emg"] to show process EMG
device_host = "192.168.1.211"  # IP address of computer which run trigno device

# load MVC data from previous trials.
file_name = "MVC_xxxx.mat"
file_dir = "MVC_01_08_2021"
list_mvc = sio.loadmat(f"{file_dir}/{file_name}")["MVC_list_max"]
list_mvc = list_mvc.T

# Set file to save data
output_file = "stream_data_xxx"
output_dir = "data_stream_01_08_2021"

# Run streaming data
muscles_idx = (0, n_electrode - 1)
run(
    muscles_idx,
    output_file=output_file,
    output_dir=output_dir,
    read_freq=read_freq,
    host_ip=device_host,
    MVC_list=list_mvc,
    #show_data=show_data,
    print_data=True,
    test_with_connection=False,
    OSC_stream=True,
    OSC_ip="127.0.0.1",
)
