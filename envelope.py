import acconeer.exptool as et
import csv

#データ取り扱い
import pandas as pd
import numpy as np
import datetime
import time

#描画
import matplotlib
import matplotlib.pyplot as plt

# def KLdivergence(p,q):
#     KL=np.sum([p * np.log(p/q) for p,q in zip(p,q)])   
#     return KL

def KLdivergence(p,q):
    # print(p)
    # print(q)
    print([b * np.log(a/b) for a,b in zip(p,q)])   
    KL=(np.sum([b * np.log(a/b) for a,b in zip(p,q)]))*(-1)
    return KL

def JSdivergence(p,q):
    #print(p)
    #print(q)
    #print(p+q)
    pq2 = (p + q) / 2
    #print(pq2)
    kl1 =  KLdivergence(p ,pq2)
    kl2 =  KLdivergence(q ,pq2)
    #print(kl1, kl2)
    JS = (kl1 / 2) + (kl2 / 2)
    return JS

# def Pearson(p,q):
#     KL=np.sum([b*((a/b-1)**2) for a,b in zip(p,q)])   
#     return KL

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
    sample = 300
    counter = 0
    b = np.ones(num)/num
    #事前に保存しておいたcsvファイル読み込み
    # df1 = pd.read_csv("test.csv",usecols=[1])
    df1 = np.loadtxt('test.csv')
    df2 = np.empty(len(df1))

    interrupt_handler = et.utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")
    start = time.time()
    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        if(counter == 0):
            df2 = np.delete(data[0],np.s_[-7::])
        else:
            df2 = df2 + np.delete(data[0],np.s_[-7::])
        counter += 1
        if(counter > sample):
            df2 = df2/sample
            break
        # print("sensor data: "+str(data[0]))
        # print("number of sensor data: "+str(len(data[0])))
    finish = time.time()
    print("処理時間: " + str(finish-start))
    
    #合計が1になるように計算
    df2 = df2/np.sum(df2)
    print(np.sum(df2))
    
    #正規化
    # df2 = (df2-df2.min())/(df2.max() - df2.min())
    # df2[np.argmin(df2)] = df2[np.argmin(df2)] + 0.00001
    # df1[np.argmin(df1)] = df1[np.argmin(df1)] + 0.00001

    # np.savetxt('test.csv',df2)
    # exit(1)
    # df2 = pd.DataFrame(get_data,columns=['sensor_data'])
    # df2.to_csv('test.csv')
    # df1 = np.delete(df1,np.s_[:300])
    # df2 = np.delete(df2,np.s_[:300])
    # df1 = np.convolve(df1,b, mode = 'same')
    # df2 = np.convolve(df2,b, mode = 'same')
    KL_U2  = KLdivergence(df1,df2)
    print(KL_U2)
    KL_U2  = KLdivergence(df2,df1)
    print(KL_U2)
    # KL_U2  = JSdivergence(df1,df2)
    # KL_U2  = Pearson(df1,df2)

# デフォルトの色
    clr=plt.rcParams['axes.prop_cycle'].by_key()['color']

#第一引数から第二引数までの範囲で第三引数刻みで数字の配列を作成
    x = np.arange(0,5000+0.25,0.25)

# p(x)
    ax = plt.subplot(1,2,1)
    #第一引数から第二引数までの範囲で第三引数刻みで数字の配列を作成
    x = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df1))
    # print(len(x))
    plt.plot(x,df1,label='$p(x)$')  

# 凡例
    plt.legend(loc=1,prop={'size': 13})
#plt.xticks(np.arange(0,5000+1,500))
    plt.xlabel('$x[cm]$')

# q(x)
    qx = plt.subplot(1,2,2)
    plt.plot(x,df2,label='$q(x)$')  

# 凡例
    plt.legend(loc=1,prop={'size': 13})
#plt.xticks(np.arange(0,5000+1,500))
    plt.xlabel('$x[cm]$')

    ax.set_title('$KL(p||q)=%.16f$' % KL_U2,fontsize=20)
    print(KL_U2)

    plt.tight_layout()
    plt.show()
    print("Disconnecting...")
    # pg_process.close()
    client.disconnect()
    




if __name__ == "__main__":
    main()
