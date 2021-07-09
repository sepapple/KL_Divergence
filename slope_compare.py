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
    if(len(x)>len(y)):
        return (np.sqrt(np.sum((x[:len(y)]-y[:len(y)])**2)))/len(y)
    else:
        return (np.sqrt(np.sum((x[:len(x)]-y[:len(x)])**2)))/len(x)

def Cosine_Similarity(x,y):
    return np.dot(x,y)/(np.sqrt(np.dot(x,x))* np.sqrt(np.dot(y,y)))


def main():
    
    df1_target_peak = []
    df2_target_peak = []
    range_start = 0.4
    range_end = 1.0
    
    #ファイルのディレクトリ
    df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/2021_06_19_17_11_33.csv'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/2021_06_19_17_05_32.csv'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/2021_06_19_17_16_58.csv'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/2021_06_19_17_38_59.csv'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/2021_06_19_18_09_37.csv'
    
    df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book2/2021_06_19_19_12_52.csv'
    
    #データ読み込み
    df1 = np.loadtxt(df1_file_dir)
    df2 = np.loadtxt(df2_file_dir)

    #ピーク検出
    df1_maxid = signal.argrelmax(df1,order=10)
    standard = max(df1)/2
    for i in df1_maxid[0]:
        if(standard < df1[i]):
            df1_target_peak.append(i)

    df2_maxid = signal.argrelmax(df2,order=10)
    standard = max(df1)/2
    for i in df2_maxid[0]:
        if(standard < df2[i]):
            df2_target_peak.append(i)

    #左側の山を抽出(df1)
    df1_start_loc = df1_target_peak[0]-1
    while(df1_start_loc > 0):
        if(df1[df1_start_loc] <= df1[df1_start_loc-1]):
            break
        df1_start_loc = df1_start_loc-1

    #右側の山を抽出(df1)
    df1_finish_loc = df1_target_peak[-1]+1
    while(df1_finish_loc < len(df1)):
        if(df1[df1_finish_loc] <= df1[df1_finish_loc+1]):
            break
        df1_finish_loc = df1_finish_loc+1

    #左側の山を抽出(df2)
    df2_start_loc = df2_target_peak[0]-1
    while(df2_start_loc > 0):
        if(df2[df2_start_loc] <= df2[df2_start_loc-1]):
            break
        df2_start_loc = df2_start_loc-1

    #右側の山を抽出(df2)
    df2_finish_loc = df2_target_peak[-1]+1
    while(df2_finish_loc < len(df2)):
        if(df2[df2_finish_loc] <= df2[df2_finish_loc+1]):
            break
        df2_finish_loc = df2_finish_loc+1
    


    #ピークの値を合わせる
    df1_diff_peak_start = df1_target_peak[0]-df1_start_loc
    df1_diff_peak_finish = df1_finish_loc-df1_target_peak[-1]
    df2_diff_peak_start = df2_target_peak[0]-df2_start_loc
    df2_diff_peak_finish = df2_finish_loc-df2_target_peak[-1]
    
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
    df1_copy = np.copy(df1[df1_target_peak[0]-start_offset:df1_target_peak[-1]+finish_offset])
    df2_copy = np.copy(df2[df2_target_peak[0]-start_offset:df2_target_peak[-1]+finish_offset])
    # df1_copy = np.delete(df1_copy,np.s_[df1_max_order-start_offset:df1_max_order+finish_offset])
    # df2_copy = np.delete(df2_copy,np.s_[df2_max_order-start_offset:df2_max_order+finish_offset])
    result = Euclidean_Distance(df1_copy,df2_copy)
    
    # df1_copy_slope = np.diff(a=df1_copy,n=1)
    # df2_copy_slope = np.diff(a=df2_copy,n=1)
    df1_copy_slope = np.gradient(df1_copy)
    df2_copy_slope = np.gradient(df2_copy)
    # print(df1_copy_slope)
    ax = plt.subplot(1,2,1)
    plt.plot(range(len(df1_copy)),df1_copy,color='r')
    plt.plot(range(len(df2_copy)),df2_copy,color='b')
    plt.title("Original Data")
    ax = plt.subplot(1,2,2)
    plt.plot(range(len(df1_copy_slope)),df1_copy_slope,color='r')
    plt.plot(range(len(df2_copy_slope)),df2_copy_slope,color='b')
    plt.title("Slope Value: %2.5f" %result)
    plt.show()
    exit(1)

    #グラフ表示
    #ピーク位置調整前
    ax = plt.subplot(1,2,1)
    interval = (int(range_end*100) - int(range_start*100))/len(df1)
    
    df1_x = np.arange(range_start*100+df1_start_loc*interval,range_start*100+df1_finish_loc*interval-interval*2,interval)
    plt.plot(df1_x,df1[df1_start_loc:df1_finish_loc-1],label='Data1',color='b') 

    df2_x = np.arange(range_start*100+df2_start_loc*interval,range_start*100+df2_finish_loc*interval-interval*2,interval)
    plt.plot(df2_x,df2[df2_start_loc:df2_finish_loc-1],label='Data1',color='r')  

    # plt.xlabel('Data Number')
    plt.xlabel('Distance(cm)')
    plt.ylabel('Amplitude')
    plt.title('Before Adjustment of Peak Position')
    plt.legend(loc=1)

    #ピーク位置調整後
    ax = plt.subplot(1,2,2)
    # print(start_offset)
    # print(finish_offset)

    plt.plot(range(0,len(df1_copy)),df1_copy,label='Previous',color='b')  
    plt.plot(range(0,len(df2_copy)),df2_copy,label='Current',color='r')  
    # plt.plot(range(df1_max_order-start_offset,df1_max_order+finish_offset),df1[df1_max_order-start_offset:df1_max_order+finish_offset],label='Previous Data',color='b')  
    # plt.plot(range(df2_max_order-start_offset,df2_max_order+finish_offset),df2_copy,label='Current Data',color='r')  

    plt.xlabel('Data Number')
    plt.ylabel('Amplitude')
    plt.title('After Adjustment of Peak Position')


    plt.tight_layout()
    plt.legend(loc=1)
    plt.show()
    

if __name__ == "__main__":
    main()
