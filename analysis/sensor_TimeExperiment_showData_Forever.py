import acconeer.exptool as et
import csv
import os


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
    range_start = 0.4
    range_end = 1.0
    colors = ['r','g','b','m','c']
    labels = ['Initial','After 5min','After 10min','After 30min','After 60min']
    # dir_name = "/Users/sepa/Desktop/センサーの実験/forever/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/forever/storage/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/change_angle/空箱/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/change_angle/本1冊/"
    dir_name = "/Users/sepa/Desktop/センサーの実験/change_angle/本2冊/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/shift_cardboard/空箱/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/shift_cardboard/本1冊/"
    # dir_name = "/Users/sepa/Desktop/センサーの実験/shift_cardboard/本2冊/"
    files = os.listdir(dir_name)
    files_path = [dir_name + file for file in files if not file.startswith('.') and not 'png' in file and os.path.isfile(dir_name+file)]
    files_path = sorted(files_path)
    
    df = np.loadtxt(files_path[0])
    x = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df))
    for i,file_path in enumerate(files_path):
        df = np.loadtxt(file_path)
        # plt.plot(x,df,color=colors[i],label= labels[i])
        plt.plot(x,df,label= str(i+1))
        # plt.plot(x,df,label= str(i)+"hours after")

    plt.xlabel('Distance(cm)')
    plt.ylabel('Amplitude')
    plt.title('Sensor Value')
    plt.tight_layout()
    plt.savefig(dir_name+'result')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
