from server import Server

if __name__ == '__main__':
    # IP_server = '192.168.1.211'
    IP_server = "localhost"
    device_ip = '192.168.1.211'
    server_port = [50000, 50001]

    # Set program variables
    read_freq = 100
    n_electrode = 2

    # Set file to save data
    output_file = "stream_data_xxx"
    output_dir = "test_stream"

    # Run streaming data
    muscles_idx = (0, n_electrode - 1)

    server = Server(
        IP=IP_server,
        server_ports=server_port,
        device="vicon",
        type="TCP",
        muscle_range=muscles_idx,
        host_pytrigno=device_ip,
        system_rate=read_freq,
        output_dir=output_dir,
        output_file=output_file,
    )

    server.run(
        stream_emg=False,
        stream_markers=False,
        stream_imu=True,
        norm_max_accel_value=1,
        norm_min_accel_value=-1,
        norm_min_gyro_value=-500,
        norm_max_gyro_value=500,
        optim=True,
        plot_emg=False,
        norm_emg=False,
        test_with_connection=True
    )
