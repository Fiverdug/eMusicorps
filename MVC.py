from MVC_pytrigno import ComputeMvc

# number of EMG electrode
n_electrode = 10

# set file and directory to save
file_name = "MVC_xxxx.mat"
file_dir = "MVC_01_08_2021"
device_host = "192.168.1.211"

# Run MVC
muscles_idx = (0, n_electrode - 1)
MVC = ComputeMvc(
    range_muscles=muscles_idx,
    output_dir=file_dir,
    device_host=device_host,
    output_file=file_name,
    test_with_connection=True,
)
list_mvc = MVC.run()
print(list_mvc)
