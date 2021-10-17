import acconeer.exptool as et
import csv
import os
import itertools

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
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/empty/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book2/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/first/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/second/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/first/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/second/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/first/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/second/'
    
    #データ読み込み
    header = []
    row_data = []
    slope_diff = []
    corr_data = []
    header.append("Indictor")
    row_data.append("RowData")
    slope_diff.append("Diff_Slope")
    corr_data.append("Corrrelation")    
    df1_files = [file_name for file_name in os.listdir(df1_file_dir) if not file_name.startswith('.') and 'csv' in file_name]

    #同じデータでの比較
    for comb in itertools.combinations(df1_files,2):
        header.append(str(comb[0])+' : '+str(comb[1]))
        df1_target_peak = []
        df1_path = df1_file_dir + comb[0]
        df1 = np.loadtxt(df1_path)
        interval = (int(range_end*100) - int(range_start*100))/len(df1)

        #ピーク検出
        df1_maxid = signal.argrelmax(df1,order=10)
        standard = max(df1)/2
        for i in df1_maxid[0]:
            if(standard < df1[i]):
                df1_target_peak.append(i)

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


        df2_target_peak = []
        df2_path = df1_file_dir + comb[1]
        df2 = np.loadtxt(df2_path)

        df2_maxid = signal.argrelmax(df2,order=10)
        standard = max(df2)/2
        for i in df2_maxid[0]:
            if(standard < df2[i]):
                df2_target_peak.append(i)

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
        row_result = Euclidean_Distance(df1_copy,df2_copy)
    
        # df1_copy_slope = np.diff(a=df1_copy,n=1)
        # df2_copy_slope = np.diff(a=df2_copy,n=1)
        df1_copy_slope = np.gradient(df1_copy)
        df2_copy_slope = np.gradient(df2_copy)
        slope_result = Euclidean_Distance(df1_copy_slope,df2_copy_slope)
        
        #相関係数
        i=0
        dataset_normalization = []
        while(i<len(df1_copy_slope) and i<len(df2_copy_slope)):
            temp=[]
            temp.append(df1_copy_slope[i])
            temp.append(df2_copy_slope[i])
            dataset_normalization.append(temp)
            i+=1
        df_normalization = pd.DataFrame(dataset_normalization,columns=['df1','df2'],)
        corr = df_normalization.corr().iat[0,1]

        
        # print("df1の段ボールからの相対距離: "+str((df1_target_peak[1]-df1_target_peak[0])*interval)+"cm")
        # print("df2の段ボールからの相対距離: "+str((df2_target_peak[1]-df2_target_peak[0])*interval)+"cm")
        row_data.append(row_result)
        slope_diff.append(slope_result)
        corr_data.append(corr)

    # print(header)
    # print(row_data)
    # print(slope_diff)
    # print(corr_data)
    df1_split = df1_file_dir.split('/')
    df = pd.DataFrame([row_data,slope_diff,corr_data],columns=header)
    df.to_csv("/Users/sepa/Desktop/センサーの実験/解析データ/"+df1_split[-3]+"-"+df1_split[-2]+".csv",index=False)

    

if __name__ == "__main__":
    main()
