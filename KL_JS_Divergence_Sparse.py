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
    counter = 0

    #0が入っていないかをチェック
    now_zero_location = p==0
    prev_zero_location = q==0
    for i in zip(now_zero_location,prev_zero_location):
        if(i[0] == True or i[1] == True):
            # print("\n0を発見しました隊長！\n")
            p = np.delete(p,counter)
            q = np.delete(q,counter)
        counter += 1

    KL=(np.sum([b * np.log(a/b) for a,b in zip(p,q)]))*(-1)
    return KL

def JSdivergence(p,q):
    pq2 = (p + q) / 2
    kl1 =  KLdivergence(p ,pq2)
    kl2 =  KLdivergence(q ,pq2)
    JS = (kl1 / 2) + (kl2 / 2)
    return JS

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
    # result = np.zeros(len(storage[0]))
    # for data in storage:
    #     result = result + data
    # result = result/len(storage)


    # clr=plt.rcParams['axes.prop_cycle'].by_key()['color']

    roop_count = 0
    prev_getData = np.loadtxt('sparse.csv')
    
    #生データ300個のフレームの平均を表示   
    show_raw_data = np.zeros(int(num))
    for frame in storage:
        show_raw_data += frame    
    show_curr_RawData = show_raw_data/sample
    
    ##保存していたデータの生データ300個のフレームの平均
    show_prev_RawData = []
    for data in prev_getData:
        show_prev_RawData.append(np.sum(data)/sample)

    x = np.arange(range_start*100,range_end*100+1,6)
    plt.subplot(1,3,1)
    plt.plot(x,show_curr_RawData,color='r')  
    plt.title("Raw Data(Current)") 
    plt.xlabel("Frame")
    plt.ylabel("Amptitude")
    plt.subplot(1,3,2)
    plt.plot(x,show_prev_RawData,color='b')  
    plt.title("Raw Data(Previous)") 
    plt.xlabel("Frame")
    plt.ylabel("Amptitude")
    plt.subplot(1,3,3)
    plt.title("Raw Data(Combination)") 
    plt.plot(x,show_curr_RawData,color='r')  
    plt.plot(x,show_prev_RawData,color='b')  
    plt.xlabel("Quantity")
    plt.ylabel("Amptitude")
    
    plt.tight_layout()
    plt.show()
    # exit(1)
    

    #データを別々に表示
    for i in range(num):
        show_data = []
        plt.subplot(math.ceil(num/2),4,roop_count*2+1)
        plt.title(str(range_start*100+i*6)+"cm(Current)") 
        plt.xlabel("Quantity")
        plt.ylabel("Amptitude")
        for j in range(len(storage)):
            show_data.append(storage[j][i])
        plt.plot(range(1,sample+1),show_data,color='r')  

        plt.subplot(math.ceil(num/2),4,roop_count*2+2)
        plt.title(str(range_start*100+i*6)+"cm(Previous)") 
        plt.plot(range(1,sample+1),prev_getData[i],color='b')
        plt.xlabel("Quantity")
        plt.ylabel("Amptitude")
        roop_count += 1

    plt.tight_layout()
    plt.show()

    #データをあわせて表示
    for i in range(num):
        show_data = []
        plt.subplot(math.ceil(num/2),2,i+1)
        for j in range(len(storage)):
            show_data.append(storage[j][i])
        plt.plot(range(1,sample+1),show_data,color='r',label="Current Data")  
        plt.title(str(range_start*100+i*6)+"cm(Combination)") 
        plt.plot(range(1,sample+1),prev_getData[i],color='b',label="Previous Data")
        plt.xlabel("Quantity")
        plt.ylabel("Amptitude")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,fontsize='small')

    plt.tight_layout()
    plt.show()

    #ある距離の時系列データの遷移とそのデータのヒストグラムの表示   
    now_getData = []
    for i in range(num):
        show_data = []
        # plt.subplot(math.ceil(num/2),2,i+1)
        plt.subplot(math.ceil(num/2),4,i*2+1)
        plt.title(str(range_start*100+i*6)+"cm") 
        plt.xlabel("Distance") 
        plt.ylabel("Amptitude") 
        for j in range(len(storage)):
            show_data.append(storage[j][i])

        now_getData.append(show_data)
        # ヒストグラムの作成と表示
        # hist, bins = np.histogram(show_data,density=True)
        plt.plot(range(1,sample+1),show_data,color = "tomato")  
        plt.subplot(math.ceil(num/2),4,i*2+2)
        plt.hist(show_data,bins=10,density=True,color = "aqua")
        plt.title(str(range_start*100+i*6)+"cm") 
        plt.xlabel("Class") 
        plt.ylabel("Frequency") 
        # print("ヒストグラムの度数"+str(hist[0]))
        # print("階級を区切る値"+str(hist[1]))

    plt.tight_layout()
    plt.show()

    #ndarray型に変換し、保存
    # now_getData = np.array(now_getData)
    # np.savetxt('sparse.csv',now_getData)

    roop_count = 0
    prev_getData = np.loadtxt('sparse.csv')
    KL_strage = []
    JS_strage = []

    #ヒストグラムの度数からKLを算出
    for i in zip(prev_getData,now_getData):
        plt.subplot(math.ceil(num/2),4,roop_count*2+1)
        plt.title(str(range_start*100+roop_count*6)+"cm(Previous)") 
        plt.xlabel("Distance") 
        plt.ylabel("Frequency") 
        now_hist = plt.hist(i[0],bins=10,color = "lime")
        now_hist = normalization(np.array(now_hist[0]))

        # now_hist_normalization = normalization(np.array(now_hist[0]))
        # now_hist[0] = now_hist_normalization
        # print(now_hist)

        plt.subplot(math.ceil(num/2),4,roop_count*2+2)
        plt.title(str(range_start*100+roop_count*6)+"cm(Now)") 
        plt.xlabel("Distance") 
        plt.ylabel("Frequency") 
        prev_hist = plt.hist(i[1],bins=10,color = "deepskyblue")
        prev_hist = normalization(np.array(prev_hist[0]))
        # print("now_histの要素の数: "+str(len(now_hist[0])))
        # print("prev_histの要素の数: "+str(len(now_hist[0])))
        KL_value = KLdivergence(now_hist,prev_hist)
        print(str(range_start*100+roop_count*6)+"cm時のKL_Divergence: "+str(KL_value))
        JS_value = JSdivergence(now_hist,prev_hist)
        print(str(range_start*100+roop_count*6)+"cm時のJS_Divergence: "+str(JS_value)+"\n")
        KL_strage.append(KL_value)
        JS_strage.append(JS_value)
        roop_count += 1
    
    
    plt.tight_layout()
    plt.show()
    
    X = list(range(int(range_start*100),int(range_end*100+1),6))
    plt.subplot(1,3,1)
    # plt.plot(range(len(KL_strage)),KL_strage,color='r',marker="o")  
    plt.plot(X,KL_strage,color='r',marker="o")  
    plt.title("KL Divergence") 
    plt.xlabel("Distance") 
    plt.ylabel("Amptitude") 

    plt.subplot(1,3,2)
    plt.plot(X,JS_strage,color='b',marker="o")  
    plt.title("JS Divergence") 
    plt.xlabel("Distance") 
    plt.ylabel("Amptitude") 

    plt.subplot(1,3,3)
    plt.plot(X,KL_strage,color='r',marker="o")  
    plt.plot(X,JS_strage,color='b',marker="o")  
    # plt.plot(X,KL_strage,color='r',marker="o",linewidth=0)  
    # plt.plot(X,JS_strage,color='b',marker="o",linewidth=0)  
    plt.title("Compare") 
    plt.xlabel("Distance") 
    plt.ylabel("Amptitude") 
    
    plt.tight_layout()
    plt.show()

    print("Disconnecting...")
    client.disconnect()

if __name__ == "__main__":
    main()
