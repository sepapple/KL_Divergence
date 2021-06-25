import acconeer.exptool as et
import csv
import cv2
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

def main():
    #グラフ全体を切り抜く設定
    beneath_height = 55
    top_height = 70
    left_width = 80
    right_width = 65

    #特徴量の部分を切り抜く設定
    beneath_height = 55
    top_height = 70
    left_width = 210
    right_width = 100
    
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    #特定のセンサデータをプロット
    dir_name = '/Users/sepa/Desktop/template/experiment/'
    
    targets = [target_dir for target_dir in os.listdir(dir_name) if not target_dir.startswith('.')]
    dir_paths = [dir_name+target+'/' for target in targets]
    file_paths = []
    for dir_path in dir_paths:
        temp=[]
        for filename in os.listdir(dir_path):
            temp.append(dir_path+filename)
        file_paths.append(temp)
    
    for i in file_paths[0]:
        for j in file_paths[1]:
            target1 = i.split('/')
            target2 = j.split('/')
            print(str(target1[-2])+'/'+str(target1[-1])+' : '+str(target2[-2])+'/'+str(target2[-1]))
            img1 = cv2.imread(i,0)
            img2 = cv2.imread(j,0)
            for meth in methods:
                # method = eval(meth)
                # res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[top_height:-beneath_height,left_width:-right_width],method)               
                # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
                # print(str(meth)+'を用いた類似度: '+str(max_val))
                print(str(meth)+'を用いた類似度: ')


    # for i in range(len(file_paths)):
    #     for 
    exit(1)
    # for i in range()
    # filenames = []
    # filenames.append(filename_1)
    # filenames.append(filename_2)
    # filenames.append(filename_3)
    # 
    # for filename in filenames:
    #     df = np.loadtxt(filename+'.csv')
    #     plt.plot(range(len(df)),df,color='b')  
    #     plt.savefig(filename)
    #     plt.cla()
    


    img1 = cv2.imread(filename_1+'.png',0)
    img2 = cv2.imread(filename_2+'.png',0)

    #表示方法(ESCでウィンドウを閉じる)
    # cv2.imshow('image',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.waitKey(0)

    #テンプレートマッチング
    for meth in methods:
        method = eval(meth)
        res = cv2.matchTemplate(img1[top_height:-beneath_height,left_width:-right_width],img2[top_height:-beneath_height,left_width:-right_width],method)               
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) 
        print(str(meth)+'を用いた類似度: '+str(max_val))

    #保存
    # cv2.imwrite(filename_1+'_Allextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Allextraction.png',img2[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_1+'_Featureextraction.png',img1[top_height:-beneath_height,left_width:-right_width])
    # cv2.imwrite(filename_2+'_Featureextraction.png',img2[top_height:-beneath_height,left_width:-right_width])


if __name__ == "__main__":
    main()
