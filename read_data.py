import data_processing

# Read pickle file saved with RT_process_pytrigno:
filename = "data_stream_01_08_2021/stream_data_xxx"  # path of saved file
data = data_processing.read_data(filename)
print(data.keys())
