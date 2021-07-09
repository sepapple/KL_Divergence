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
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/empty/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book2/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/first/'
    df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/second/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/first/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/second/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/second/'
    # df1_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/first/'

    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book1/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/first/book2/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/first/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/empty/second/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/first/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book1/second/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/first/'
    # df2_file_dir = '/Users/sepa/Desktop/センサーの実験/second/book2/second/'
    
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
    df1_files = [file_name for file_name in os.listdir(df1_file_dir) if not file_name.startswith('.') and 'csv' in file_name]
    # df2_files = [file_name for file_name in os.listdir(df2_file_dir) if not file_name.startswith('.') and 'csv' in file_name]
    for file in df1_files:
        df1_target_peak = []
        df1_path = df1_file_dir + file
        df1 = np.loadtxt(df1_path)
        print(df1_path)
        interval = (int(range_end*100) - int(range_start*100))/len(df1)
        # print(str(target1[-3])+'/'+str(target1[-2])+'/'+str(target1[-1])+' : '+str(target1[-3])+'/'+str(target2[-2])+'/'+str(target2[-1]))

        #ピーク検出
        df1_target_peak = signal.argrelmax(df1,order=2)
        
        tmp = 0
        for idx in df1_target_peak[0]:
            if df1[tmp] < df1[idx]:
                tmp = idx
        df1_cardboard_peak = list(df1_target_peak[0]).index(tmp)

        counter = 1
        for i in range(df1_cardboard_peak,len(df1_target_peak[0])-1):
            print("df1の段ボールと"+str(counter)+"つ目のピークの相対距離: "+str((df1_target_peak[0][df1_cardboard_peak+counter]-df1_target_peak[0][df1_cardboard_peak])*interval)+"cm")
            counter += 1

        df1_X = np.arange(range_start*100,range_end*100,(int(range_end*100) - int(range_start*100))/len(df1))
        # df1_X = [x for x in range(len(df1))]
        plt.plot(df1_X,df1,'r')
        plt.plot(df1_X[df1_target_peak[0]],df1[df1_target_peak[0]],'bo')
        # plt.plot(df1_target_peak[0],df1[df1_target_peak[0]],'bo')
        plt.xlabel('Distance(cm)')
        plt.ylabel('Amplitude')
        plt.title('Sensor and Peak Value')
        plt.tight_layout()

        if not os.path.exists(df1_file_dir+'result'):#ディレクトリがなかったら
            os.mkdir(df1_file_dir+'result')#作成したいフォルダ名を作成
        plt.savefig(df1_file_dir + 'result/'+str(file)+'.png')
        plt.cla()

        # for j in df2_files:
        #     df2_target_peak = []
        #     df2_path = df2_file_dir + j
        #     df2 = np.loadtxt(df2_path)
        # 
        #     target1 = df1_path.split('/')
        #     target2 = df2_path.split('/')
        #     print(str(target1[-3])+'/'+str(target1[-2])+'/'+str(target1[-1])+' : '+str(target1[-3])+'/'+str(target2[-2])+'/'+str(target2[-1]))
        #     # header.append(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
        # 
        #     df2_maxid = signal.argrelmax(df2,order=5)
        #     df2_target_peak = signal.argrelmax(df2,order=2)
        #     tmp = 0
        #     for idx in df2_target_peak[0]:
        #         if df2[tmp] < df2[idx]:
        #             tmp = idx
        # 
        #     df2_cardboard_peak = list(df2_target_peak[0]).index(tmp)
        #     
        #    
        #     plt.subplot(1,2,1)
        #     df1_X = [x for x in range(len(df1))]
        #     plt.plot(df1_X,df1,'r')
        #     plt.plot(df1_target_peak[0],df1[df1_target_peak[0]],'bo')
        #     plt.title("DataFrame1")
        # 
        #     plt.subplot(1,2,2)
        #     df2_X = [x for x in range(len(df2))]
        #     plt.plot(df2_X,df2,'r')
        #     plt.plot(df2_target_peak[0],df2[df2_target_peak[0]],'bo')
        #     plt.title("DataFrame2")
            # plt.show()
            
            # counter = 1
            # for i in range(df1_cardboard_peak,len(df1_target_peak[0])-1):
            #     print("df1の段ボールと"+str(counter)+"つ目のピークの相対距離: "+str((df1_target_peak[0][df1_cardboard_peak+counter]-df1_target_peak[0][df1_cardboard_peak])*interval)+"cm")
            #     counter += 1

            # counter = 1
            # for i in range(df2_cardboard_peak,len(df2_target_peak[0])-1):
            #     print("df2の段ボールと"+str(counter)+"つ目のピークの相対距離: "+str((df2_target_peak[0][df1_cardboard_peak+counter]-df2_target_peak[0][df1_cardboard_peak])*interval)+"cm")
            #     counter += 1

    #違うデータでの比較
    # print(header)
    # print(row_data)
    # print(slope_diff)
    # print(corr_data)
    # df = pd.DataFrame([row_data,slope_diff,corr_data],columns=header)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/empty_book1_template_matching.csv",index=False)
    # df.to_csv("/Users/sepa/Desktop/template/experiment/result/graph_extract/empty_book2_template_matching.csv",index=False)

    # df1_split = df1_file_dir.split('/')
    # df2_split = df2_file_dir.split('/')
    # print(df1_split)
    # print(df2_split)
    
    # df.to_csv("/Users/sepa/Desktop/センサーの実験/解析データ/"+df1_split[-3]+"-"+df1_split[-2]+"_"+df2_split[-3]+"-"+df2_split[-2]+".csv",index=False)
    

if __name__ == "__main__":
    main()
