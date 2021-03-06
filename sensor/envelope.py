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

# def KLdivergence(p,q):
#     KL=np.sum([p * np.log(p/q) for p,q in zip(p,q)])   
#     return KL

def KLdivergence(p,q):
    print([b * np.log(a/b) for a,b in zip(p,q)])   
    KL=(np.sum([b * np.log(a/b) for a,b in zip(p,q)]))*(-1)
    return KL

def JSdivergence(p,q):
    pq2 = (p + q) / 2
    #print(pq2)
    kl1 =  KLdivergence(p ,pq2)
    kl2 =  KLdivergence(q ,pq2)
    #print(kl1, kl2)
    JS = (kl1 / 2) + (kl2 / 2)
    return JS

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
    start = time.time()
    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        if(counter == 0):
            # df2 = np.delete(data[0],np.s_[-7::])
            df2 = data[0]
        else:
            df2 = df2 + data[0]
        counter += 1
        if(counter > sample):
            df2 = df2/sample
            break
        # print("sensor data: "+str(data[0]))
        # print("number of sensor data: "+str(len(data[0])))
    finish = time.time()
    print("処理時間: " + str(finish-start))
    
    #前に取得したデータと現在取得したデータの処理
    difference = abs(df1-df2) #振幅の差の絶対値を取得
    abs_max = max(difference)
    print(abs_max)

    temp_list = []
    temp_ndarray = np.loadtxt('absvalue_samevalue.csv',delimiter=',')
    temp_ndarray = np.append(temp_ndarray,abs_max)
    temp_list.append(temp_ndarray)
    print(temp_ndarray)
    # temp_list = list(temp_ndarray)
    # temp_list.append()
    # temp_list = np.array(temp_list)
    # print(temp_list)
    # np.savetxt('absvalue_samevalue.csv',temp_ndarray,delimiter=',')
    np.savetxt('absvalue_samevalue.csv',np.array(temp_list),delimiter=',')
    # difference = (df1-df2)**2 #振幅の差の二乗差を取得
    # print(difference)
    
    #合計が1になるように計算
    # df2 = df2/np.sum(df2)
    # print(np.sum(df2))
    
    
    #データ保存部分
    # np.savetxt('test.csv',df2)
    # exit(1)
    # df2 = pd.DataFrame(get_data,columns=['sensor_data'])
    # df2.to_csv('test.csv')
    # df1 = np.delete(df1,np.s_[:300])
    # df2 = np.delete(df2,np.s_[:300])
    # df1 = np.convolve(df1,b, mode = 'same')
    # df2 = np.convolve(df2,b, mode = 'same')
    # KL_U2  = KLdivergence(df1,df2)
    # print(KL_U2)
    # KL_U2  = KLdivergence(df2,df1)
    # print(KL_U2)
    # KL_U2  = JSdivergence(df1,df2)
    # KL_U2  = Pearson(df1,df2)

# デフォルトの色
    clr=plt.rcParams['axes.prop_cycle'].by_key()['color']

#第一引数から第二引数までの範囲で第三引数刻みで数字の配列を作成
    x = np.arange(0,5000+0.25,0.25)
# p(x)
    ax = plt.subplot(1,3,1)
    #第一引数から第二引数までの範囲で第三引数刻みで数字の配列を作成
    x = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df1))

    # print(len(x))
    plt.plot(x,df1,label='Previous Data')  

# 凡例
    # plt.legend(loc=1)
#plt.xticks(np.arange(0,5000+1,500))
    plt.xlabel('$x[cm]$')
    plt.ylabel('Amplitude')
    plt.title('Previous Data')

# q(x)
    qx = plt.subplot(1,3,2)
    plt.plot(x,df2,label='Current Data')  

    # 凡例
    # plt.legend(loc=1)
    #plt.xticks(np.arange(0,5000+1,500))
    plt.xlabel('$x[cm]$')
    plt.ylabel('Amplitude')
    plt.title('Current Data')

    qx = plt.subplot(1,3,3)
    plt.plot(x,difference,label='Square difference')  

    # plt.legend(loc=1)
    #plt.xticks(np.arange(0,5000+1,500))
    plt.xlabel('$x[cm]$')
    plt.ylabel('Amplitude')
    plt.title('Absolute value of difference')

    plt.tight_layout()
    # plt.show()

    print("Disconnecting...")
    # pg_process.close()
    client.disconnect()
    




if __name__ == "__main__":
    main()
