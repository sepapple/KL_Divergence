import numpy as np
import pyqtgraph as pg

from acconeer.exptool import clients, configs, utils
from acconeer.exptool.pg_process import PGProccessDiedException, PGProcess

#描画
import matplotlib
import matplotlib.pyplot as plt

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

    sensor_config = configs.SparseServiceConfig()
    sensor_config.sensor = args.sensors
    sensor_config.range_interval = [0.18, 0.60]
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

    while not interrupt_handler.got_signal:
        data_info, data = client.get_next()
        counter += 1

        temp = np.zeros(len(data[0][0]))
        for sweep in data[0]:
            temp = temp + sweep
        temp = temp/len(data[0])
        storage.append(temp)

        if(counter >= sample):
            break

    result = np.zeros(len(storage[0]))
    print(len(storage[0]))
    for data in storage:
        result = result + data
    
    result = result/len(storage)
    # print(storage)
    print(result)
    # print(len(result))

    clr=plt.rcParams['axes.prop_cycle'].by_key()['color']

# p(x)
    ax = plt.subplot(1,1,1)
    #第一引数から第二引数までの範囲で第三引数刻みで数字の配列を作成
    x = np.arange(range_start*100,range_end*100+1,6)
    print(x)
    # print(len(x))
    plt.plot(x,result,label='$p(x)$')  

# 凡例
    plt.legend(loc=1,prop={'size': 13})
    plt.xlabel('$x[cm]$')

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
    plt.show()
    print("Disconnecting...")
    client.disconnect()

if __name__ == "__main__":
    main()
