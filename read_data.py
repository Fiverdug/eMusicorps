import data_processing

# Read pickle file saved with RT_process_pytrigno:
filename = "data_streamed/stream_data_01_05_2021"  # path of saved file
data = data_processing.read_data(filename)
print(data.keys())
