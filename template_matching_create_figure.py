import csv
import cv2
import os

#データ取り扱い
import pandas as pd
import numpy as np
import datetime
import time
import datetime
import math

#描画
import matplotlib
import matplotlib.pyplot as plt

def Euclidean_Distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def Cosine_Similarity(x,y):
    return np.dot(x,y)/(np.sqrt(np.dot(x,x))* np.sqrt(np.dot(y,y)))


def main():
    #特定のセンサデータをプロット
    input_dir_name = '/Users/sepa/Desktop/センサーの実験/first/'
    output_dir_name = '/Users/sepa/Desktop/template/experiment/'
    # input_files = 
    for directory in os.listdir(input_dir_name):
        if not directory.startswith('.'):
            target_input_dir = [input_dir_name+directory+'/'  ]
            for path in target_input_dir:
                for filename in os.listdir(path):
                    if (not filename.startswith('.') and not 'png' in filename):
                        target_input_file = path + filename
                        target_output_file =  output_dir_name+directory+'/'+filename
                        target_output_file = target_output_file.replace('.csv','')
                        
                        df = np.loadtxt(target_input_file)
                        plt.plot(range(len(df)),df,color='b')
                        plt.savefig(target_output_file+'.png')
                        plt.cla()

if __name__ == "__main__":
    main()
