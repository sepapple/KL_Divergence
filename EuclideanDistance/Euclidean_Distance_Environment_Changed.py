import acconeer.exptool as et
import csv

#データ取り扱い
import pandas as pd
import numpy as np
import datetime
import time
import datetime
import math
from scipy import signal

#描画
import matplotlib
import matplotlib.pyplot as plt

def Euclidean_Distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def Cosine_Similarity(x,y):
    return np.dot(x,y)/(np.sqrt(np.dot(x,x))* np.sqrt(np.dot(y,y)))


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
    
    sensor_config = et.EnvelopeServiceConfig()
    sensor_config.sensor = args.sensors
    range_start = 0.4
    range_end = 1.0
    sensor_config.range_interval = [range_start, range_end]
    sensor_config.profile = sensor_config.Profile.PROFILE_2
    sensor_config.hw_accelerated_average_samples = 20
    sensor_config.downsampling_factor = 2

    session_info = client.setup_session(sensor_config)
    # print("session_info: "+str(session_info))
    
    # pg_updater = PGUpdater(sensor_config, None, session_info)
    # pg_process = et.PGProcess(pg_updater)
    # pg_process.start()
    
    client.start_session()

    #KLダイバージェンス設定
    dx = 0.001 
    
    #移動平均の個数
    num = 500
    #取得するセンサデータの個数とカウンター
    # sample = 2000
    sample = 5000
    counter = 0
    b = np.ones(num)/num
    #事前に保存しておいたcsvファイル読み込み
    # df1 = pd.read_csv("test.csv",usecols=[1])


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
    
    #ダンボールと商品のピーク値を検出
    df1 = np.loadtxt('test.csv')
    df1_maxid = signal.argrelmax(df1,order=10)
    standard = max(df1)/2
    df1_target_peak = []
    for i in df1_maxid[0]:
        if(standard < df1[i]):
            df1_target_peak.append(i)

    df2_maxid = signal.argrelmax(df2,order=10)
    print(df2_maxid)
    standard = max(df2)/2
    df2_target_peak = []
    for i in df2_maxid[0]:
        if(standard < df2[i]):
            df2_target_peak.append(i)

    # print(df2_target_peak)
    # plt.plot(range(0,len(df2)),df2)
    # plt.show()
    # exit(1)
    
    #データ保存部
    # np.savetxt('test.csv',df2)
    # exit(1)
    # print((int(range_end*100) - int(range_start*100))/len(df1))

    # print(df2)
    # print(np.argmax(df2))
    # print(df2[np.argmax(df2)])
    


    #以前のデータから最大値の山の抽出範囲検索
    df1_max_order = df1_target_peak[1]
    df1_max_value = df1[df1_target_peak[1]]
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
    df2_max_order = df2_target_peak[1]
    df2_max_value = df2[df2_target_peak[1]]
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

    #ピーク位置調整後
    # ax = plt.subplot(1,2,1)
    # ax = plt.subplot(1,3,1)
    ax = plt.subplot(2,2,1)

    plt.plot(range(0,len(df1_copy)),df1_copy,label='Previous',color='b')  
    plt.plot(range(0,len(df2_copy)),df2_copy,label='Current',color='r')  

    plt.xlabel('Data Number')
    plt.ylabel('Amplitude')
    plt.title('After Adjustment of Peak Position')
    plt.legend(loc='best')

    #ユークリッド距離,コサイン類似度の算出
    euclidean_distance = Euclidean_Distance(df1_copy,df2_copy)
    cosine_similarity = Cosine_Similarity(df1_copy,df2_copy)
    point_distance = np.sqrt((df1_copy-df2_copy)**2)
    peak_point = np.argmax(df1_copy)

    print("各点におけるユークリッド距離の総和: "+str(euclidean_distance))
    # print("コサイン類似度: "+str(cosine_similarity))
    print("ピーク地点におけるユークリッド距離: "+str(np.sqrt((df1_copy[peak_point]-df2_copy[peak_point])**2)))
    # print("点ごとの距離: "+str(point_distance))


    # ax = plt.subplot(1,2,2)
    # ax = plt.subplot(1,3,2)
    ax = plt.subplot(2,2,2)
    value = np.sqrt(np.square(df1_copy-df2_copy))

    plt.plot(range(0,len(value)),value,label='Euclidean_Distance',color='r')  

    plt.xlabel('Data Number')
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distance')
    
    df1_slope = []
    df2_slope = []
    for i in range(len(df1_copy)-1):
        df1_slope.append(df1_copy[i+1]-df1_copy[i])
        df2_slope.append(df2_copy[i+1]-df2_copy[i])

    # ax = plt.subplot(1,3,3)
    ax = plt.subplot(2,2,3)
    plt.plot(range(0,len(df1_slope)),df1_slope,label='Previous',color='b')  
    plt.plot(range(0,len(df2_slope)),df2_slope,label='Current',color='r')  

    plt.xlabel('Data Number')
    plt.ylabel('Slope')
    plt.legend()
    plt.title('Slope')

    ax = plt.subplot(2,2,4)
    slope_diff = np.sqrt(np.square(np.array(df1_slope)-np.array(df2_slope)))
    plt.plot(range(0,len(slope_diff)),slope_diff,label='Previous',color='b')  
    print("傾きの差のユークリッド距離: "+str(slope_diff[np.argmax(slope_diff)]))

    plt.xlabel('Data Number')
    plt.ylabel('Slope Difference')
    plt.title('Slope Difference')
    
    # dir_name = "/Users/sepa/Desktop/60GHzレーダーの実験/Euclidean_Distance/実験環境変更/空箱同士/"
    dir_name = "/Users/sepa/Desktop/"
    now = datetime.datetime.fromtimestamp(time.time())
    file_name = dir_name + now.strftime("%Y_%m_%d_%H_%M_%S") + ".png"
    
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

    print("Disconnecting...")
    # pg_process.close()
    client.disconnect()

if __name__ == "__main__":
    main()
