import acconeer.exptool as et
import csv

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
    
    range_start = 0.2
    range_end = 0.6
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
    sample = 500
    counter = 0
    b = np.ones(num)/num
    #事前に保存しておいたcsvファイル読み込み
    # df1 = pd.read_csv("test.csv",usecols=[1])
    df1 = np.loadtxt('test.csv')
    # df2 = np.zeros(len(df1))

    interrupt_handler = et.utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")
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

    #データ保存部
    # np.savetxt('test.csv',df2)
    # exit(1)

    # print(df2)
    # print(np.argmax(df2))
    # print(df2[np.argmax(df2)])

    #以前のデータから最大値の山の抽出範囲検索
    df1_max_order = np.argmax(df1)
    df1_max_value = df1[np.argmax(df1)]

    df1_start_loc = df1_max_order-1
    while(df1_start_loc > 0):
        if(df1[df1_start_loc] <= df1[df1_start_loc-1]):
            break
        df1_start_loc = df1_start_loc-1

    df1_finish_loc = df1_max_order+1
    while(df1_finish_loc < len(df1)):
        if(df1[df1_finish_loc] <= df1[df1_finish_loc+1]):
            break
        df1_finish_loc = df1_finish_loc+1
    
    #現在のデータから最大値の山の抽出範囲検索
    df2_max_order = np.argmax(df2)
    df2_max_value = df2[np.argmax(df2)]
    df2_start_loc = df2_max_order-1

    while(df2_start_loc > 0):
        if(df2[df2_start_loc] <= df2[df2_start_loc-1]):
            break
        df2_start_loc = df2_start_loc-1

    df2_finish_loc = df2_max_order+1
    while(df2_finish_loc < len(df2)):
        if(df2[df2_finish_loc] <= df2[df2_finish_loc+1]):
            break
        df2_finish_loc = df2_finish_loc+1


    #ピークの値を合わせる
    # df1_copy = df1.copy()
    # df2_copy = df2.copy()
    df1_diff_peak_start = df1_max_order-df1_start_loc
    df1_diff_peak_finish = df1_finish_loc-df1_max_order
    df2_diff_peak_start = df2_max_order-df2_start_loc
    df2_diff_peak_finish = df2_finish_loc-df2_max_order
    
    #ピークよりも左側の調整
    if(df2_diff_peak_start>df1_diff_peak_start):
        # diff = df2_diff_peak_start-df1_diff_peak_start #スライスを指定するためマイナスの値にする
        start_offset = df1_diff_peak_start
    else:
        start_offset = df2_diff_peak_start

    #ピークよりも右側の調整
    if(df2_diff_peak_finish>df1_diff_peak_finish):
        finish_offset = df1_diff_peak_finish
    else:
        finish_offset = df2_diff_peak_finish
    
    #ピークを合わせるために削除
    df1_copy = np.copy(df1[df1_max_order-start_offset:df1_max_order+finish_offset])
    df2_copy = np.copy(df2[df2_max_order-start_offset:df2_max_order+finish_offset])
    # df1_copy = np.delete(df1_copy,np.s_[df1_max_order-start_offset:df1_max_order+finish_offset])
    # df2_copy = np.delete(df2_copy,np.s_[df2_max_order-start_offset:df2_max_order+finish_offset])

    #ピーク位置調整前
    ax = plt.subplot(1,2,1)
    interval = (int(range_end*100) - int(range_start*100))/len(df1)
    x = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df1))

    # df1_x = np.arange(range_start*100+df1_start_loc*interval,range_start*100+df1_finish_loc*interval-interval,interval)
    plt.plot(x,df1,color='b')  

    # df2_x = np.arange(range_start*100+df2_start_loc*interval,range_start*100+df2_finish_loc*interval-interval,interval)
    # plt.plot(range(df2_start_loc,df2_finish_loc),df2[df2_start_loc:df2_finish_loc],label='Current Data',color='r')  

    plt.xlabel('Distance(cm)')
    plt.ylabel('Amplitude')
    plt.title('Before Extraction')

    #ピーク位置調整後
    ax = plt.subplot(1,2,2)
    df1_x = np.arange(range_start*100+df1_start_loc*interval,range_start*100+df1_finish_loc*interval-interval,interval)

    plt.plot(df1_x,df1[df1_start_loc:df1_finish_loc-1],color='r')  
    # plt.plot(range(df1_max_order-start_offset,df1_max_order+finish_offset),df1[df1_max_order-start_offset:df1_max_order+finish_offset],label='Previous Data',color='b')  
    # plt.plot(range(df2_max_order-start_offset,df2_max_order+finish_offset),df2_copy,label='Current Data',color='r')  

    plt.xlabel('Distance(cm)')
    plt.ylabel('Amplitude')
    plt.title('After Extraction')


    plt.tight_layout()
    plt.show()

    print("Disconnecting...")
    # pg_process.close()
    client.disconnect()
    




if __name__ == "__main__":
    main()
