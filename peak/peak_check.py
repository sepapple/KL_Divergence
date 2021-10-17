import acconeer.exptool as et
import csv
import os

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
    if(len(x)>len(y)):
        return (np.sqrt(np.sum((x[:len(y)]-y[:len(y)])**2)))/len(y)
    else:
        return (np.sqrt(np.sum((x[:len(x)]-y[:len(x)])**2)))/len(x)

def main():
    
    df1_target_peak = []
    range_start = 0.4
    range_end = 1.0
    
    #ファイルのディレクトリ
    # file_dir = '/Users/sepa/Desktop/センサーの実験/first/empty/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/first/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/second/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/first/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/second/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/first/book2/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/first/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/second/'

    # file_dir = '/Users/sepa/Desktop/センサーの実験/shift_cardboard/空箱/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/shift_cardboard/本1冊/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/shift_cardboard/本2冊/'

    # file_dir = '/Users/sepa/Desktop/センサーの実験/change_angle/空箱/'
    # file_dir = '/Users/sepa/Desktop/センサーの実験/change_angle/本1冊/'
    file_dir = '/Users/sepa/Desktop/センサーの実験/change_angle/本2冊/'
    
    
    #データ読み込み
    header = []
    row_data = []
    slope_diff = []
    corr_data = []
    header.append("Indictor")
    row_data.append("RowData")
    slope_diff.append("Diff_Slope")
    corr_data.append("Corrrelation")    

    plt.subplot(1,1,1)

    #違うデータでの比較
    df1_files = [file_name for file_name in os.listdir(file_dir) if not file_name.startswith('.') and 'csv' in file_name]
    # df2_files = [file_name for file_name in os.listdir(df2_file_dir) if not file_name.startswith('.') and 'csv' in file_name]
    df1_files = sorted(df1_files)
    
    for file in df1_files:
        df1_target_peak = []
        df1_path = file_dir + file
        df1 = np.loadtxt(df1_path)
        print(df1_path)
        interval = (int(range_end*100) - int(range_start*100))/len(df1)
        # print(str(target1[-3])+'/'+str(target1[-2])+'/'+str(target1[-1])+' : '+str(target1[-3])+'/'+str(target2[-2])+'/'+str(target2[-1]))

        #ピーク検出
        df1_target_peak = signal.argrelmax(df1,order=1)

        temp_peak = []
        standard = max(df1)/2
        for i in df1_target_peak[0]:
            if(standard < df1[i]):
                temp_peak.append(i)
        
        df1_cardboard_peak = list(df1_target_peak[0]).index(temp_peak[0])
        # print(df1_cardboard_peak)

        # tmp = 0
        # for idx in df1_target_peak[0]:
        #     if df1[tmp] < df1[idx]:
        #         tmp = idx
        # df1_cardboard_peak = list(df1_target_peak[0]).index(tmp)

        counter = 1
        for i in range(df1_cardboard_peak,len(df1_target_peak[0])-1):
            print("df1の段ボールと"+str(counter)+"つ目のピークの相対距離: "+str((df1_target_peak[0][df1_cardboard_peak+counter]-df1_target_peak[0][df1_cardboard_peak])*interval)+"cm")
            print("df1の段ボールと"+str(counter)+"つ目のピークとの振幅差: "+str((df1[df1_target_peak[0][df1_cardboard_peak]]-df1[df1_target_peak[0][df1_cardboard_peak+counter]])))
            print("df1の段ボールと"+str(counter)+"つ目のピークとの比: "+str((df1[df1_target_peak[0][df1_cardboard_peak+counter]]/df1[df1_target_peak[0][df1_cardboard_peak]])))
            counter += 1

        df1_X = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df1))
        # df1_X = [x for x in range(len(df1))]
        plt.plot(df1_X,df1,'r')
        plt.plot(df1_X[df1_target_peak[0]],df1[df1_target_peak[0]],'bo')
        plt.xlabel('Distance(cm)')
        plt.ylabel('Amplitude')
        plt.title('Sensor and Peak Value')
        plt.tight_layout()

        if not os.path.exists(file_dir+'result'):#ディレクトリがなかったら
            os.mkdir(file_dir+'result')#作成したいフォルダ名を作成
        plt.savefig(file_dir + 'result/'+str(file)+'.png')
        plt.cla()

    

if __name__ == "__main__":
    main()
