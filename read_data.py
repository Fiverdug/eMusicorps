import data_processing
import scipy.io as sio

# Read pickle file saved with RT_process_pytrigno:
filename = "/home/amedeo/Documents/programmation/eMusicorps/test_accel/stream_data_xxx"  # path of saved file
data = data_processing.read_data(filename)  # return dictionary data type

# save to matlab format file:
# sio.savemat("data_xxx.mat", data)
