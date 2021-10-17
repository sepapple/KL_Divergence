import pandas as pd
import numpy as np
import os

def main():
    dir_name = "/Users/sepa/Desktop/センサーの実験/解析データ/"
    files = [file_name for file_name in os.listdir(dir_name) if not file_name.startswith('.') and 'csv' in file_name]
    for file in files:
        path = dir_name+file
        df = pd.read_csv(path,header=0,index_col=0)
        print(path)
        print("生データ平均")
        print((np.sum(df.values[0]))/df.size)
        print("傾きの差")
        print((np.sum(df.values[1]))/df.size)
        print("相関係数")
        print((np.sum(df.values[2]))/df.size)
        # print(np.sum(df.values.flatten()))
        # print(df.size)
        # print((np.sum(df.values.flatten()))/df.size)

if __name__ == "__main__":
    main()
