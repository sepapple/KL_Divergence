import numpy as np
import pyqtgraph as pg

from acconeer.exptool import clients, configs, utils
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess

#描画
import matplotlib
import matplotlib.pyplot as plt

#計算
import math

def normalization(x):
    x_sum = np.sum(x)
    return x/x_sum

def KLdivergence(p,q):
    # print(p)
    # print(q)
    # print([b * np.log(a/b) for a,b in zip(p,q)])   
    KL=(np.sum([b * np.log(a/b) for a,b in zip(p,q)]))*(-1)
    return KL

def main():
    args = utils.ExampleArgumentParser().parse_args()
    utils.config_logging(args)

    if args.socket_addr:
        client = clients.SocketClient(args.socket_addr)
    elif args.spi:
        client = clients.SPIClient()
    else:
        port = args.serial_port or utils.autodetect_serial_port()
        client = clients.UARTClient(port)

    client.squeeze = False

    range_start = 0.18
    range_end = 0.60
    num = int((range_end*100-range_start*100)/6)+1

    sensor_config = configs.SparseServiceConfig()
    sensor_config.sensor = args.sensors
    sensor_config.range_interval = [range_start, range_end]
    sensor_config.sweeps_per_frame = 16
    sensor_config.hw_accelerated_average_samples = 60
    sensor_config.sampling_mode = sensor_config.SamplingMode.A
    sensor_config.profile = sensor_config.Profile.PROFILE_2
    sensor_config.gain = 0.6

    session_info = client.setup_session(sensor_config)

    # pg_updater = PGUpdater(sensor_config, None, session_info)
    # pg_process = PGProcess(pg_updater)
    # pg_process.start()
    client.start_session()

    storage = []
    counter = 0
    sample = 300

    interrupt_handler = utils.ExampleInterruptHandler()
    print("Press Ctrl-C to end session")

    temp = np.zeros(num)
    
    #端末からデータを受け取り、フレームに入っているsweepの平均を取得し、tempに追加
    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        counter += 1

        for sweep in data[0]:
            temp = temp + sweep

        temp = temp/len(data[0])
        storage.append(temp)
        temp = np.zeros(int(num))

        if(counter >= sample):
            break


    #300個のフレームから距離ごとに平均を取得
    result = np.zeros(len(storage[0]))
    for data in storage:
        result = result + data
    result = result/len(storage)
    # print(storage)
    # print(result)


    # clr=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #フレーム内のスイープの平均の遷移の表示   
    # for i in range(num):
    #     show_data = []
    #     plt.subplot(math.ceil(num/2),2,i+1)
    #     plt.title(str(range_start*100+i*6)+"cm") 
    #     for j in range(len(storage)):
    #         show_data.append(storage[j][i])
    #     plt.plot(range(1,sample+1),show_data)  
    
    now_getData = []
    #ある距離の時系列データの遷移とそのデータのヒストグラムの表示   
    for i in range(num):
        show_data = []
        # plt.subplot(math.ceil(num/2),2,i+1)
        plt.subplot(math.ceil(num/2),4,i*2+1)
        plt.title(str(range_start*100+i*6)+"cm") 
        for j in range(len(storage)):
            show_data.append(storage[j][i])

        now_getData.append(show_data)
        # ヒストグラムの作成と表示
        # hist, bins = np.histogram(show_data,density=True)
        plt.plot(range(1,sample+1),show_data,color = "tomato")  
        plt.subplot(math.ceil(num/2),4,i*2+2)
        plt.hist(show_data,bins=10,density=True,color = "aqua")
        plt.title(str(range_start*100+i*6)+"cm") 
        # print("ヒストグラムの度数"+str(hist[0]))
        # print("階級を区切る値"+str(hist[1]))

    plt.tight_layout()
    plt.show()
    
    #ndarray型に変換
    now_getData = np.array(now_getData)
    #データを保存
    # np.savetxt('sparse.csv',now_getData)
    roop_count = 0
    prev_getData = np.loadtxt('sparse.csv')
    #ヒストグラムの度数からKLを算出
    for i in zip(prev_getData,now_getData):
        plt.subplot(math.ceil(num/2),4,roop_count*2+1)
        plt.title(str(range_start*100+roop_count*6)+"cm(Previous)") 
        now_hist = plt.hist(i[0],bins=10,color = "lime")
        now_hist = list(now_hist)
        now_hist = normalization(np.array(now_hist[0]))
        # print(now_hist)
        # now_hist_normalization = normalization(np.array(now_hist[0]))
        # now_hist[0] = now_hist_normalization
        # plt.hist(now_hist)
        # print(now_hist)

        plt.subplot(math.ceil(num/2),4,roop_count*2+2)
        plt.title(str(range_start*100+roop_count*6)+"cm(Now)") 
        prev_hist = plt.hist(i[1],bins=10,color = "deepskyblue")
        prev_hist = normalization(np.array(prev_hist[0]))
        # print("now_histの要素の数: "+str(len(now_hist[0])))
        # print("prev_histの要素の数: "+str(len(now_hist[0])))
        value = KLdivergence(now_hist,prev_hist)
        print(str(range_start*100+roop_count*6)+"cm時のKL_Divergence: "+str(value))
        roop_count += 1
                
    plt.tight_layout()
    plt.show()

    
    # p(x)
    #300個のフレームから距離ごとに平均を表示
    # ax = plt.subplot(1,1,1)
    # x = np.arange(range_start*100,range_end*100+1,6)
    # plt.plot(x,result,label='$p(x)$')  
    # plt.legend(loc=1,prop={'size': 13})
    # plt.xlabel('$x[cm]$')
    # plt.show()

# 凡例

# q(x)
#     qx = plt.subplot(1,2,2)
#     plt.plot(x,df2,label='$q(x)$')  
# 
# # 凡例
#     plt.legend(loc=1,prop={'size': 13})
# #plt.xticks(np.arange(0,5000+1,500))
#     plt.xlabel('$x[cm]$')
# 
#     ax.set_title('$KL(p||q)=%.16f$' % KL_U2,fontsize=20)
#     print(KL_U2)
# 
    # plt.tight_layout()
    print("Disconnecting...")
    client.disconnect()

if __name__ == "__main__":
    main()
