import acconeer.exptool as et
import csv
import os
import pathlib

#データ取り扱い
import pandas as pd
import numpy as np
import datetime
import time
import math

#描画
import matplotlib
import matplotlib.pyplot as plt


def main():
    args = et.utils.ExampleArgumentParser().parse_args()
    et.utils.config_logging(args)

    if args.socket_addr:
        client = et.SocketClient(args.socket_addr)
    elif args.spi:
        client = et.SPIClient()
    else:
        port = args.serial_port or et.utils.autodetect_serial_port()
        client = et.UARTClient(port)

    client.squeeze = False
    
    range_start = 0.4
    range_end = 1.0
    sensor_config = et.EnvelopeServiceConfig()
    sensor_config.sensor = args.sensors
    sensor_config.range_interval = [range_start, range_end]
    sensor_config.profile = sensor_config.Profile.PROFILE_2
    sensor_config.hw_accelerated_average_samples = 20
    sensor_config.downsampling_factor = 2

    session_info = client.setup_session(sensor_config)
    # pg_updater = PGUpdater(sensor_config, None, session_info)
    # pg_process = et.PGProcess(pg_updater)
    # pg_process.start()
    
    client.start_session()

    #KLダイバージェンス設定
    dx = 0.001 
    
    #移動平均の個数
    num = 500

    #取得するセンサデータの個数とカウンター
    sample = 3250
    counter = 0
    b = np.ones(num)/num
    #事前に保存しておいたcsvファイル読み込み
    # df1 = pd.read_csv("test.csv",usecols=[1])
    df1 = np.loadtxt('test.csv')
    # df2 = np.zeros(len(df1))

    interrupt_handler = et.utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")
    before_time = time.time()
    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        if(counter == 0):
            df2 = data[0]
        else:
            df2 = df2 + data[0]
        counter += 1
        if(counter > sample):
            df2 = df2/sample
            break
    after_time = time.time()
    print(after_time-before_time)
    #データ保存部
    # dir_name = "/Users/sepa/Desktop/センサーの実験/book1/"
    dir_name = "/Users/sepa/Desktop/センサーの実験/book2/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/test/"
    now = datetime.datetime.fromtimestamp(time.time())
    file_name = dir_name + now.strftime("%Y_%m_%d_%H_%M_%S") + ".csv"
    np.savetxt(file_name,df2)

    print("Disconnecting...")
    # pg_process.close()
    client.disconnect()
    




if __name__ == "__main__":
    main()
